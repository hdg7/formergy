#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSGA-II energy+semantics optimizer for TinyLlama with robust Ollama handling.

What you get
------------
1) Energy:
 - GPU: NVML cumulative energy (mJ since driver load); compute before/after delta
   and sum across user-selected GPUs (e.g., --gpu-indexes 0,1).
 - CPU: CodeCarbon estimator (GPU disabled to avoid double-counting).

2) Semantics:
 - Cosine similarity (BAAI/bge-large-en-v1.5).
 - NLI entailment probability (FacebookAI/roberta-large-mnli).
 - Fluency via GPT-2 perplexity (exp(cross-entropy)).

3) NSGA-II:
 - Minimize energy per token and semantic delta (1 - score).
 - Candidates come from Qwen (via Ollama) + genetic crossover/mutation.

4) Robust Ollama integration:
 - --test-ollama mode, warmup wait, exponential backoff for chat calls,
   in-loop recovery (wait & retry) and safe mutation fallback.

Run (example)
-------------
python nsga2_ollama_energy_refactor_sem.py \
  --prompt-file prompt.txt \
  --gpu-indexes 0,1 \
  --ollama-warmup-wait 90 \
  --ollama-retries 12 \
  --ollama-initial-delay 1.5 \
  --ollama-backoff 1.7 \
  --ollama-max-delay 15 \
  --pop-size 6 --generations 3 --offspring-per-parent 1 \
  --plot
"""
import os
import re
import json
import time
import math
import csv
import random
import argparse
import traceback
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel,
    pipeline,
)

# ------------ Energy: GPU via NVML (nvidia-ml-py exposes 'pynvml') ------------
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTotalEnergyConsumption, NVMLError
)

# ------------ Energy: CPU via CodeCarbon (estimator; works in VMs) ------------
from codecarbon import EmissionsTracker

# ------------ Ollama client (Qwen proposals) ----------------------------------
import ollama

# ------------ Plotting & data -------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

KWH_TO_J = 3.6e6  # 1 kWh = 3.6e6 Joules


# ========================== Utility: Read prompts =============================
def read_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                prompts.append(t)
    return prompts


# ========================== TinyLlama Wrappers ================================
def _maybe_head_mask(x: torch.Tensor, mask: Optional[Set[int]], num_heads: int):
    """Zero selected heads for tensors shaped [B,S,H,D] or [B,H,S,D]."""
    if not mask or x.dim() != 4:
        return x
    if x.size(2) == num_heads:  # (B, S, H, D)
        x[:, :, list(mask), :] = 0
    elif x.size(1) == num_heads:  # (B, H, S, D)
        x[:, list(mask), :, :] = 0
    return x


class EnergyAwareLlamaAttention(nn.Module):
    """
    Wrap an attention module and apply a per-layer head mask
    to the per-head output (best-effort for different layouts).
    """

    def __init__(self, attn_module: nn.Module, head_mask: Optional[Set[int]] = None):
        super().__init__()
        self.attn = attn_module
        self.head_mask = head_mask
        self.num_heads = getattr(attn_module, "num_heads", None) or getattr(attn_module, "n_heads", None)

    def forward(self, *args, **kwargs):
        out = self.attn(*args, **kwargs)
        y = out[0] if isinstance(out, tuple) else out
        if isinstance(y, torch.Tensor) and self.num_heads is not None:
            y = _maybe_head_mask(y, self.head_mask, self.num_heads)
        if isinstance(out, tuple):
            out = (y,) + out[1:]
        else:
            out = y
        return out


class EnergyAwareLlamaMLP(nn.Module):
    """
    Wrap LLaMA MLP (SwiGLU) and apply a structured channel mask to up/gate projections.
    Works with sharded models (device_map='auto') by moving the mask to the input device.
    """

    def __init__(self, mlp_module: nn.Module, neuron_mask: Optional[Set[int]] = None):
        super().__init__()
        self.mlp = mlp_module
        self.neuron_mask = neuron_mask
        self.gate_proj = getattr(mlp_module, "gate_proj", None)
        self.up_proj = getattr(mlp_module, "up_proj", None)
        self.down_proj = getattr(mlp_module, "down_proj", None)
        self.act = getattr(mlp_module, "act_fn", getattr(torch.nn, "SiLU")())

        if not (self.gate_proj and self.up_proj and self.down_proj):
            raise RuntimeError("Unexpected MLP structure; check Transformers/model version.")

        hidden_size = self.up_proj.out_features
        if neuron_mask:
            mask = torch.ones(hidden_size, dtype=torch.bool)  # created on CPU by default
            idx = torch.tensor(sorted(list(neuron_mask)), dtype=torch.long)
            mask.index_fill_(0, idx, False)
            # CPU buffer is fine; we'll move it on-the-fly in forward()
            self.register_buffer("binary_mask", mask, persistent=False)
        else:
            self.binary_mask = None

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.binary_mask is not None:
            mask = self.binary_mask
            # Ensure mask lives on the same device as the compute
            if mask.device != up.device:
                mask = mask.to(up.device, non_blocking=True)
            up = up * mask
            gate = gate * mask
        return self.down_proj(self.act(gate) * up)


def apply_wrappers(model, head_masks: Dict[int, Set[int]], neuron_masks: Dict[int, Set[int]]):
    """
    Apply EnergyAwareLlamaAttention and EnergyAwareLlamaMLP to each decoder layer.
    """
    layers = getattr(model, "model", model).layers  # LLaMA-style
    for i, layer in enumerate(layers):
        # Wrap attention
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if attn is not None:
            wrapped = EnergyAwareLlamaAttention(attn, head_mask=head_masks.get(i, set()))
            if hasattr(layer, "self_attn"):
                layer.self_attn = wrapped
            else:
                layer.attention = wrapped

        # Wrap MLP
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            layer.mlp = EnergyAwareLlamaMLP(mlp, neuron_mask=neuron_masks.get(i, set()))


def load_tinyllama(model_id: str,
                   attn_impl: str,
                   use_4bit: bool = False,
                   dtype: torch.dtype = torch.bfloat16) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load TinyLlama with optional NF4 4-bit quantization and chosen attention backend.
    Defaults to SDPA; will fall back automatically if FlashAttention-2 isn't available.
    """
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
            quantization_config=quant,
            attn_implementation=attn_impl,
        )
    except ImportError as e:
        if "flash_attn" in str(e) or "FlashAttention2" in str(e):
            print("[warn] FlashAttention-2 not available; falling back to SDPA.")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
                quantization_config=quant,
                attn_implementation="sdpa",
            )
        else:
            raise

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure pad_token_id is defined to avoid generation warnings and enable min tokens enforcement
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def generate_once(model, tok, prompt: str,
                  max_new_tokens: int = 128,
                  min_new_tokens: int | None = None) -> Tuple[str, float, int]:
    """
    Returns text, latency_seconds, num_generated_tokens.
    Respects a minimum number of new tokens if supported by the installed version of
    Transformers. If `min_new_tokens` is not supported, falls back to `min_length`.
    """
    # Guard: define pad token if missing to keep generate() happy
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    # Prefer min_new_tokens if provided
    if min_new_tokens is not None and min_new_tokens > 0:
        try:
            gen_kwargs["min_new_tokens"] = int(min_new_tokens)
        except Exception:
            pass

    with torch.no_grad():
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        try:
            out = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            # Older Transformers may not have `min_new_tokens`; approximate via min_length
            if min_new_tokens is not None and min_new_tokens > 0:
                total_min_len = int(inputs["input_ids"].shape[1] + int(min_new_tokens))
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    min_length=total_min_len
                )
            else:
                out = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if k != "min_new_tokens"})
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.time() - t0

    gen_toks = out.shape[1] - inputs["input_ids"].shape[1]
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, dt, gen_toks


# ======================= Semantic Evaluation Suite ============================
class SemanticEvaluator:
    """
    Loads:
    - BGE embeddings (BAAI/bge-large-en-v1.5) for cosine similarity,
    - RoBERTa MNLI (FacebookAI/roberta-large-mnli) for entailment probability,
    - GPT-2 for fluency (perplexity).
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # BGE: embedding model (Transformer encoder). We'll mean-pool last_hidden_state.
        self.bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        self.bge = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5").to(self.device).eval()

        # NLI pipeline for entailment
        self.nli = pipeline("text-classification", model="FacebookAI/roberta-large-mnli",
                            device=0 if self.device.startswith("cuda") else -1)

        # Fluency via GPT-2 perplexity
        self.gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
        self.gpt2_tok.pad_token = self.gpt2_tok.eos_token  # safe default if batching later
        self.gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device).eval()

    @torch.no_grad()
    def _bge_embed(self, texts: List[str]) -> torch.Tensor:
        # Mean-pool outputs and L2-normalize for cosine similarity
        enc = self.bge_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        out = self.bge(**enc, output_hidden_states=False, return_dict=True)
        last = out.last_hidden_state  # [B, T, H]
        attn_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (last * attn_mask).sum(dim=1)
        counts = attn_mask.sum(dim=1).clamp(min=1)
        emb = summed / counts
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb  # [B, H]

    def cosine_sim(self, a_texts: List[str], b_texts: List[str]) -> float:
        a = self._bge_embed(a_texts)
        b = self._bge_embed(b_texts)
        sims = (a * b).sum(dim=1)  # cosine since normalized
        return float(sims.mean().item())

    # def entail_prob(self, premise: List[str], hypothesis: List[str]) -> float:
    #     # Compute mean entailment probability (label 'ENTAILMENT')
    #     print(premise)
    #     print(hypothesis)
    #     probs = []
    #     for p, h in zip(premise, hypothesis):
    #         print(p)
    #         print(h)
    #         out = self.nli({"text": p, "text_pair": h})[0]
    #         if out["label"].upper().startswith("ENTAIL"):
    #             print("FLOAT1")
    #             probs.append(float(out["score"]))
    #         else:
    #             print("FLOAT2")
    #             probs.append(float(out["score"]) if out["label"].upper().startswith("NEUTRAL") else 0.0)
    #     print("DIV")
    #     return float(sum(probs) / max(1, len(probs)))

    def entail_prob(self, premise: List[str], hypothesis: List[str]) -> float:
        """
        Compute mean entailment probability with robust batching & guarding.
        - Accepts list[str] or str; coerces to list[str].
        - Filters out empty/whitespace-only pairs.
        - Runs ONE batched call to the NLI pipeline with truncation.
        - Uses return_all_scores=True to extract the 'ENTAILMENT' score per pair.
        - Returns 0.0 if no valid pairs remain or if the pipeline fails.
        """
        # --------- normalize & quick diagnostics ----------
        if isinstance(premise, str):
            premise = [premise]
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis]
        # Defensive: show shapes/types once
        try:
            print(f"[entail] types: premise={type(premise).__name__}, hypothesis={type(hypothesis).__name__}")
            print(f"[entail] lengths: len(premise)={len(premise)}, len(hypothesis)={len(hypothesis)}")
        except Exception as _:
            pass  # keep going

        # --------- build aligned, non-empty pairs ----------
        pairs: List[Tuple[str, str]] = []
        for p, h in zip(premise, hypothesis):
            p = (p or "").strip()
            h = (h or "").strip()
            if p and h:
                pairs.append((p, h))

        if not pairs:
            print("[entail] no non-empty (premise, hypothesis) pairs -> entail_prob = 0.0")
            return 0.0

        # --------- prepare batched inputs ----------
        # HF sequence-classification supports dicts {"text": p, "text_pair": h}
        # Using truncation avoids length errors on very long strings
        batch_inputs = [{"text": p, "text_pair": h} for (p, h) in pairs]

        try:
            # return_all_scores=True gives all label probs (ENTAILMENT/NEUTRAL/CONTRADICTION)
            outputs = self.nli(
                batch_inputs,
                return_all_scores=True,
                truncation=True
            )
        except Exception as e:
            # Log once and return neutral score
            print(f"[entail] pipeline call failed: {e!r} -> entail_prob = 0.0")
            return 0.0

        # --------- extract 'ENTAILMENT' scores ----------
        probs: List[float] = []
        for out in outputs:
            # 'out' is a list like [{"label": "ENTAILMENT", "score": ...}, {...}, {...}]
            entail_score = 0.0
            try:
                for item in out:
                    if str(item.get("label", "")).upper().startswith("ENTAIL"):
                        entail_score = float(item.get("score", 0.0))
                        break
            except Exception:
                entail_score = 0.0
            probs.append(entail_score)

        if not probs:
            print("[entail] empty probs after extraction -> entail_prob = 0.0")
            return 0.0

        mean_prob = float(sum(probs) / len(probs))
        print(f"[entail] mean entailment probability over {len(probs)} valid pairs: {mean_prob:.4f}")
        return mean_prob

    @torch.no_grad()
    def gpt2_perplexity(self, texts: List[str]) -> float:
        """
        Robust PPL on single texts:
        - If a text is empty/whitespace, substitute a minimal fallback.
        - Ensure the sequence has >=2 tokens so LM loss is defined.
        - Compute CE on labels==input_ids, then PPL=exp(CE).
        """
        ppl_list = []
        for t in texts:
            # sanitize / minimal fallback
            t = t if isinstance(t, str) else ""
            t = t.strip()
            if len(t) == 0:
                t = " ."

            # tokenize WITHOUT padding; batch size = 1
            enc = self.gpt2_tok(t, return_tensors="pt", truncation=True, padding=False, max_length=1024)
            ids = enc["input_ids"].to(self.device)  # [1, L]
            attn = torch.ones_like(ids, device=self.device)  # [1, L]

            # ensure L >= 2
            if ids.size(1) < 2:
                eos = torch.tensor([[self.gpt2_tok.eos_token_id]], device=self.device)
                if ids.size(1) == 0:
                    ids = eos.repeat(1, 2)
                    attn = torch.ones_like(ids, device=self.device)
                else:
                    ids = torch.cat([ids, eos], dim=1)
                    attn = torch.ones_like(ids, device=self.device)

            out = self.gpt2(input_ids=ids, attention_mask=attn, labels=ids)
            ce = float(out.loss.item())
            ppl_list.append(math.exp(ce))
        return float(sum(ppl_list) / max(1, len(ppl_list)))


def composite_semantic_score(sim_cos: float, entail_prob: float, fluency_ppl: float,
                             ppl_cap: float = 100.0) -> float:
    """
    Combine metrics into a single scalar in [0,1] approx:
    score = 0.5 * sim_cos + 0.5 * entail_prob, then decayed by fluency penalty.
    Fluency penalty = min(1, ppl_cap / max(ppl_cap, ppl))
    """
    base = 0.5 * sim_cos + 0.5 * entail_prob
    penalty = min(1.0, ppl_cap / max(ppl_cap, fluency_ppl))
    return max(0.0, min(1.0, base * penalty))


# =========================== Energy Measurement ===============================
def codecarbon_start(cc_dir: str) -> Tuple[EmissionsTracker, int, str]:
    os.makedirs(cc_dir, exist_ok=True)
    csv_path = os.path.join(cc_dir, "emissions.csv")
    tracker = EmissionsTracker(
        measure_power_secs=1,
        tracking_mode="process",
        save_to_file=True,
        output_dir=cc_dir,
        gpu_ids=""  # disable GPU tracking in CodeCarbon (we measure via NVML)
    )
    start_row = 0
    if os.path.exists(csv_path):
        try:
            start_row = len(pd.read_csv(csv_path))
        except Exception:
            start_row = 0
    tracker.start()
    return tracker, start_row, csv_path


def codecarbon_stop(tracker: EmissionsTracker, csv_path: str, start_row: int) -> float:
    tracker.stop()
    if not os.path.exists(csv_path):
        return 0.0
    df = pd.read_csv(csv_path)
    if df.empty:
        return 0.0
    cols = ["energy_consumed", "energy_consumed (kWh)", "energy_consumed_kwh", "energy_kwh"]
    col = next((c for c in cols if c in df.columns), None)
    if col is None:
        col = next((c for c in df.columns if "kwh" in c.lower()), None)
    if col is None:
        return 0.0
    delta_kwh = float(df.iloc[start_row:][col].fillna(0.0).sum())
    return delta_kwh * KWH_TO_J


# =============================== Baseline Run =================================
def run_model_for_prompts(model, tok, prompts: List[str], max_new_tokens: int,
                          min_new_tokens: int | None = None) -> Tuple[List[str], float, int]:
    """Generate output for all prompts (concatenated timing), return (texts, total_latency_s, total_tokens)."""
    outputs, total_dt, total_tokens = [], 0.0, 0
    for p in prompts:
        text, dt, gen_toks = generate_once(model, tok, p,
                                           max_new_tokens=max_new_tokens,
                                           min_new_tokens=min_new_tokens)
        outputs.append(text)
        total_dt += dt
        total_tokens += gen_toks
    return outputs, total_dt, total_tokens


# =============================== Ollama helpers ===============================
SYSTEM_JSON_PROMPT = """You are an energy-optimization assistant for TinyLlama (LLaMA architecture).
Return a compact JSON with keys:
- "attn_impl": one of ["sdpa","eager","flash_attention_2"]
- "head_masks": {layer_index: [head_ids...], ...}
- "neuron_masks": {layer_index: [neuron_ids...], ...}
Constraints:
- Keep accuracy reasonable; small sparsity (e.g., 5-20%) on some early layers.
- If unsure, prefer 'sdpa'.
JSON ONLY.
"""


def ollama_wait_until_ready(timeout_s: float = 60.0, probe_interval: float = 2.0) -> bool:
    """
    Poll Ollama until it responds or timeout elapses.
    """
    t0 = time.time()
    while time.time() - t0 <= timeout_s:
        try:
            ollama.list()
            return True
        except KeyboardInterrupt:
            raise
        except Exception:
            time.sleep(probe_interval)
    return False


def test_ollama_minimal(model_name: str = "qwen3-coder-next") -> bool:
    """
    One-shot chat to confirm Ollama + model respond.
    """
    try:
        msgs = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Say 'OK' and stop."},
        ]
        resp = ollama.chat(model=model_name, messages=msgs, stream=False)
        txt = (resp.get("message", {}) or {}).get("content", "").strip().lower()
        return "ok" in txt
    except KeyboardInterrupt:
        raise
    except Exception:
        return False


def ollama_chat_with_retry(model: str,
                           messages: list,
                           retries: int = 10,
                           initial_delay: float = 1.5,
                           backoff: float = 1.6,
                           max_delay: float = 12.0):
    """
    Call ollama.chat() with exponential backoff. Returns response dict on success,
    raises the last exception if all retries fail.
    """
    delay = max(0.1, float(initial_delay))
    attempt = 0
    last_err = None
    while attempt <= retries:
        try:
            return ollama.chat(model=model, messages=messages, stream=False)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            print(f"[ollama] chat failed (attempt {attempt+1}/{retries+1}): {e}")
            time.sleep(delay)
            delay = min(max_delay, delay * backoff)
            attempt += 1
    if last_err is not None:
        print("[ollama] giving up after retries; last error:\n",
              "".join(traceback.format_exception_only(type(last_err), last_err)).strip())
        raise last_err if last_err else RuntimeError("Unknown Ollama failure")


def ask_qwen_for_config(ollama_model: str, parent_cfg: Optional[dict] = None,
                        retries: int = 10, initial_delay: float = 1.5,
                        backoff: float = 1.6, max_delay: float = 12.0) -> dict:
    user_inst = "Propose a modest sparsity config for TinyLlama." if parent_cfg is None else \
        ("Mutate this parent config slightly while keeping sparsity modest:\n" +
         json.dumps(parent_cfg))
    messages = [
        {"role": "system", "content": SYSTEM_JSON_PROMPT},
        {"role": "user", "content": user_inst},
    ]
    resp = ollama_chat_with_retry(
        model=ollama_model, messages=messages,
        retries=retries, initial_delay=initial_delay,
        backoff=backoff, max_delay=max_delay
    )
    text = resp["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("Ollama did not return a valid JSON block.")
    cfg = json.loads(m.group(0))
    cfg.setdefault("attn_impl", "sdpa")
    cfg.setdefault("head_masks", {})
    cfg.setdefault("neuron_masks", {})
    return cfg


# ============================== NSGA-II Primitives ============================
@dataclass
class Candidate:
    attn_impl: str = "sdpa"
    head_masks: Dict[int, Set[int]] = field(default_factory=dict)  # {layer: set(head_ids)}
    neuron_masks: Dict[int, Set[int]] = field(default_factory=dict)

    # evaluated stats
    energy_j_per_token: float = float("inf")
    semantic_score: float = 0.0
    latency_s: float = 0.0
    tokens: int = 0
    cpu_j: float = 0.0
    gpu_j: float = 0.0
    ppl: float = 1e9

    def clone(self) -> "Candidate":
        return Candidate(
            attn_impl=self.attn_impl,
            head_masks={k: set(v) for k, v in self.head_masks.items()},
            neuron_masks={k: set(v) for k, v in self.neuron_masks.items()},
        )


def json_to_candidate(cfg: dict) -> Candidate:
    hm = {int(k): set(map(int, v)) for k, v in cfg.get("head_masks", {}).items()}
    nm = {int(k): set(map(int, v)) for k, v in cfg.get("neuron_masks", {}).items()}
    return Candidate(attn_impl=cfg.get("attn_impl", "sdpa"), head_masks=hm, neuron_masks=nm)


def candidate_to_json(c: Candidate) -> dict:
    return {
        "attn_impl": c.attn_impl,
        "head_masks": {k: sorted(list(v)) for k, v in c.head_masks.items()},
        "neuron_masks": {k: sorted(list(v)) for k, v in c.neuron_masks.items()},
    }


def fast_non_dominated_sort(pop: List[Candidate]) -> List[List[int]]:
    """Return fronts as lists of indices."""
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    rank = [0 for _ in pop]
    fronts = [[]]

    def dominates(a: Candidate, b: Candidate) -> bool:
        # Minimize f1=energy_j_per_token and f2=(1 - semantic_score)
        f1a, f2a = a.energy_j_per_token, (1.0 - a.semantic_score)
        f1b, f2b = b.energy_j_per_token, (1.0 - b.semantic_score)
        return (f1a <= f1b and f2a <= f2b) and (f1a < f1b or f2a < f2b)

    for p in range(len(pop)):
        Sp = set()
        np = 0
        for q in range(len(pop)):
            if dominates(pop[p], pop[q]):
                Sp.add(q)
            elif dominates(pop[q], pop[p]):
                np += 1
        S[p] = Sp
        n[p] = np
        if np == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(pop: List[Candidate], idxs: List[int]) -> Dict[int, float]:
    if not idxs:
        return {}
    distances = {i: 0.0 for i in idxs}
    # f1
    f1 = [(i, pop[i].energy_j_per_token) for i in idxs]
    f2 = [(i, 1.0 - pop[i].semantic_score) for i in idxs]
    for arr in [f1, f2]:
        arr.sort(key=lambda x: x[1])
        distances[arr[0][0]] = float("inf")
        distances[arr[-1][0]] = float("inf")
        minv, maxv = arr[0][1], arr[-1][1]
        denom = max(1e-12, maxv - minv)
        for j in range(1, len(arr) - 1):
            distances[arr[j][0]] += (arr[j + 1][1] - arr[j - 1][1]) / denom
    return distances


def select_next_population(pop: List[Candidate], pop_size: int) -> List[Candidate]:
    fronts = fast_non_dominated_sort(pop)
    new_pop = []
    for front in fronts:
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend([pop[i] for i in front])
        else:
            dist = crowding_distance(pop, front)
            sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
            need = pop_size - len(new_pop)
            new_pop.extend([pop[i] for i in sorted_front[:need]])
            break
    return new_pop


def crossover(a: Candidate, b: Candidate) -> Candidate:
    child = a.clone()
    if random.random() < 0.5:
        child.attn_impl = b.attn_impl
    # merge/random masks
    for layer in set(list(a.head_masks.keys()) + list(b.head_masks.keys())):
        set_a = a.head_masks.get(layer, set())
        set_b = b.head_masks.get(layer, set())
        child.head_masks[layer] = set_a.union(
            set(random.sample(list(set_b), k=max(0, len(set_b) // 2))) if set_b else set()
        )
    for layer in set(list(a.neuron_masks.keys()) + list(b.neuron_masks.keys())):
        set_a = a.neuron_masks.get(layer, set())
        set_b = b.neuron_masks.get(layer, set())
        child.neuron_masks[layer] = set_a.union(
            set(random.sample(list(set_b), k=max(0, len(set_b) // 2))) if set_b else set()
        )
    return child


def mutate(c: Candidate, max_heads_drop: int = 2, max_neurons_drop: int = 4, attn_choices=None) -> Candidate:
    attn_choices = attn_choices or ["sdpa", "eager"]
    child = c.clone()
    if random.random() < 0.2:
        child.attn_impl = random.choice(attn_choices)

    # randomly drop/add a few entries
    def jit_drop_add(d: Dict[int, Set[int]], drop_k: int, add_prob=0.2):
        if not d:
            return
        layer = random.choice(list(d.keys()))
        s = set(d[layer])
        # drop
        for _ in range(random.randint(0, drop_k)):
            if s:
                s.remove(random.choice(list(s)))
        # add small id
        if random.random() < add_prob:
            s.add(random.randint(0, 15))
        d[layer] = s

    jit_drop_add(child.head_masks, max_heads_drop)
    jit_drop_add(child.neuron_masks, max_neurons_drop)
    return child


# =============================== Evaluation ===================================
def evaluate_candidate(cand: Candidate,
                       model_id: str,
                       base_outputs: List[str],
                       prompts: List[str],
                       max_new_tokens: int,
                       min_new_tokens: int | None,
                       gpu_indexes: List[int],
                       cc_dir: str,
                       sem: 'SemanticEvaluator',
                       use_4bit: bool) -> Candidate:
    # Load model with candidate's attn_impl
    model, tok = load_tinyllama(model_id, attn_impl=cand.attn_impl, use_4bit=use_4bit)
    # apply wrappers
    apply_wrappers(model, cand.head_masks, cand.neuron_masks)

    print("ENERGY")
    # Energy start
    nvmlInit()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in gpu_indexes]
    e0_mJ = [nvmlDeviceGetTotalEnergyConsumption(h) for h in handles]
    nvmlShutdown()
    tracker, start_row, csv_path = codecarbon_start(cc_dir)

    print("TRANSFORMERS")
    # Generate transformed outputs
    texts, dt, toks = run_model_for_prompts(model, tok, prompts, max_new_tokens, min_new_tokens)

    # Energy end
    nvmlInit()
    e1_mJ = [nvmlDeviceGetTotalEnergyConsumption(h) for h in handles]
    nvmlShutdown()

    gpu_J = 0.0
    for a, b in zip(e1_mJ, e0_mJ):
        delta = (a - b) / 1000.0
        gpu_J += max(0.0, delta)
    cpu_J = codecarbon_stop(tracker, csv_path, start_row)
    print("ENERGY END")

    # Semantics and fluency
    sim = sem.cosine_sim(base_outputs, texts)  # cosine similarity
    print("SIM")
    entail = sem.entail_prob(base_outputs, texts)  # entailment probability
    print("ENTAIL")
    ppl = sem.gpt2_perplexity(texts)  # fluency
    print("FLUENCY")
    sem_score = composite_semantic_score(sim, entail, ppl)
    print("SEMANTICS")

    cand.latency_s = dt
    cand.tokens = toks
    cand.cpu_j = cpu_J
    cand.gpu_j = gpu_J
    total_j = cpu_J + gpu_J
    cand.energy_j_per_token = total_j / max(1, toks)
    cand.semantic_score = sem_score
    cand.ppl = ppl
    return cand


# =================================== Main =====================================
def main():
    ap = argparse.ArgumentParser(description="NSGA-II optimizer for TinyLlama energy & semantics (resilient Ollama)")

    # Models & backends
    ap.add_argument("--ollama-model", default="qwen3-coder-next", help="Ollama model for proposals")
    ap.add_argument("--tinyllama-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF id for TinyLlama")
    ap.add_argument("--attn-impl", default="sdpa", help="Default attention backend (sdpa\neager\nflash_attention_2)")
    ap.add_argument("--use-4bit", action="store_true", help="Load TinyLlama in 4-bit NF4 (bitsandbytes)")

    # Prompts
    ap.add_argument("--prompt-file", default="prompt.txt", help="Path to a file with one prompt per line")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--min-new-tokens", type=int, default=160,
                    help="Minimum number of new tokens to generate before allowing EOS (fallbacks to min_length on older transformers)")

    # Energy
    ap.add_argument("--gpu-indexes", default="0", help="Comma-separated GPU indexes to sum (e.g., 0,1)")
    ap.add_argument("--cc-dir", default="cc_logs", help="Directory for CodeCarbon CSV")

    # NSGA-II
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--offspring-per-parent", type=int, default=1)

    # Ollama health & retries
    ap.add_argument("--test-ollama", action="store_true",
                    help="Run a quick Ollama health check and exit (0 on success).")
    ap.add_argument("--ollama-warmup-wait", type=int, default=60,
                    help="Max seconds to wait for Ollama to become ready at start.")
    ap.add_argument("--ollama-retries", type=int, default=10,
                    help="Number of retries for a single Ollama request.")
    ap.add_argument("--ollama-initial-delay", type=float, default=1.5,
                    help="Initial sleep (seconds) before first retry.")
    ap.add_argument("--ollama-backoff", type=float, default=1.6,
                    help="Multiplicative backoff factor between retries.")
    ap.add_argument("--ollama-max-delay", type=float, default=12.0,
                    help="Max sleep (seconds) between retries.")

    # Output
    ap.add_argument("--results-csv", default="nsga2_results.csv")
    ap.add_argument("--plot", action="store_true", help="Plot Pareto frontier of final population")
    ap.add_argument("--compile", action="store_true", help="Use torch.compile(reduce-overhead, dynamic=True)")

    args = ap.parse_args()

    # Ensure sensible generation bounds
    if args.max_new_tokens is not None and args.min_new_tokens is not None and args.max_new_tokens < args.min_new_tokens:
        print(f"[warn] --max-new-tokens ({args.max_new_tokens}) < --min-new-tokens ({args.min_new_tokens}); raising max to min.")
        args.max_new_tokens = args.min_new_tokens

    # Optional health-only mode
    if args.test_ollama:
        ready = ollama_wait_until_ready(timeout_s=max(5.0, args.ollama_warmup_wait / 2.0))
        ok = (ready and test_ollama_minimal(args.ollama_model))
        print("[ollama] ready:", ready, "\n minimal chat:", ok)
        raise SystemExit(0 if (ready and ok) else 1)

    # Warmup wait before the main loop (helpful in containerized starts)
    print("[ollama] waiting for server...")
    if not ollama_wait_until_ready(timeout_s=float(args.ollama_warmup_wait), probe_interval=2.0):
        print("[ollama] server not reachable in warmup window; will still try with retries later.")
    else:
        print("[ollama] server reachable.")

    # Read prompts
    prompts = read_prompts(args.prompt_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompt_file}")

    # GPUs to track
    gpu_indexes = [int(x) for x in args.gpu_indexes.split(",") if x.strip() != ""]
    print(f"[energy] Tracking NVML GPUs: {gpu_indexes}")

    # Baseline: plain TinyLlama (no masks), attention as args.attn_impl
    print("[baseline] loading baseline TinyLlama...")
    base_model, base_tok = load_tinyllama(args.tinyllama_id, attn_impl=args.attn_impl, use_4bit=args.use_4bit)
    if args.compile:
        base_model = torch.compile(base_model, mode="reduce-overhead", dynamic=True)

    print("[baseline] generating outputs for prompts...")
    base_outputs, base_dt, base_toks = run_model_for_prompts(
        base_model, base_tok, prompts, args.max_new_tokens, args.min_new_tokens
    )
    print(f"[baseline] latency={base_dt:.3f}s tokens={base_toks}")

    # Semantic evaluator (loads BGE, RoBERTa-MNLI, GPT-2)
    sem = SemanticEvaluator()
    print("[baseline] computing fluency (ppl) on baseline outputs...")
    ppl_base = sem.gpt2_perplexity(base_outputs)

    # Initialize population: include baseline candidate (no masking) + Qwen proposals
    population: List[Candidate] = []
    base_cand = Candidate(attn_impl=args.attn_impl, head_masks={}, neuron_masks={})
    # Baseline semantics vs itself: sim=1, entail=1, composite near 1 minus fluency penalty
    base_cand.semantic_score = composite_semantic_score(1.0, 1.0, ppl_base)
    base_cand.ppl = ppl_base
    base_cand.energy_j_per_token = float("inf")  # we don't evaluate baseline energy
    population.append(base_cand)

    need = max(0, args.pop_size - 1)
    for _ in range(need):
        try:
            cfg = ask_qwen_for_config(
                args.ollama_model, parent_cfg=None,
                retries=args.ollama_retries,
                initial_delay=args.ollama_initial_delay,
                backoff=args.ollama_backoff,
                max_delay=args.ollama_max_delay,
            )
            population.append(json_to_candidate(cfg))
        except Exception as e:
            print("[warn] Ollama proposal failed; inserting a light random candidate:", e)
            population.append(Candidate(attn_impl="sdpa"))

    # Evolution loop
    results_rows = []
    for gen in range(args.generations):
        print(f"\n=== Generation {gen+1}/{args.generations} ===")

        # Evaluate all candidates that lack energy/semantics (skip baseline energy if inf)
        for i, cand in enumerate(population):
            if math.isinf(cand.energy_j_per_token):
                print(f" [eval] candidate {i}: attn_impl={cand.attn_impl}, "
                      f"heads={ {k:sorted(list(v)) for k,v in cand.head_masks.items()} }, "
                      f"neurons={ {k:sorted(list(v)) for k,v in cand.neuron_masks.items()} }")
                try:
                    print("EVALUATING CANDIDATE")
                    evaluated = evaluate_candidate(
                        cand, args.tinyllama_id, base_outputs, prompts,
                        args.max_new_tokens, args.min_new_tokens,
                        gpu_indexes, args.cc_dir, sem, args.use_4bit
                    )
                    population[i] = evaluated
                    print("CONSTRUCTING ROW")
                    row = {
                        "gen": gen,
                        "idx": i,
                        "attn_impl": evaluated.attn_impl,
                        "head_masks": json.dumps({k: sorted(list(v)) for k, v in evaluated.head_masks.items()}),
                        "neuron_masks": json.dumps({k: sorted(list(v)) for k, v in evaluated.neuron_masks.items()}),
                        "energy_j_per_token": evaluated.energy_j_per_token,
                        "semantic_score": evaluated.semantic_score,
                        "latency_s": evaluated.latency_s,
                        "tokens": evaluated.tokens,
                        "cpu_J": evaluated.cpu_j,
                        "gpu_J": evaluated.gpu_j,
                        "ppl": evaluated.ppl,
                    }
                    results_rows.append(row)
                    print(row)
                except Exception as e:
                    print("THE EXCEPTION IS ACTIVATED")
                    msg = str(e) if not isinstance(e, BaseException) else repr(e)
                    print(msg)
                    transient = ("Failed to connect to Ollama" in msg) or ("Connection refused" in msg) \
                                or ("Read timed out" in msg) or ("Max retries exceeded" in msg)
                    if transient:
                        print("[warn] Ollama seems unavailable during evaluation; waiting to recover...")
                        if ollama_wait_until_ready(timeout_s=min(30.0, float(args.ollama_warmup_wait)), probe_interval=2.0):
                            try:
                                evaluated = evaluate_candidate(
                                    cand, args.tinyllama_id, base_outputs, prompts,
                                    args.max_new_tokens, args.min_new_tokens,
                                    gpu_indexes, args.cc_dir, sem, args.use_4bit
                                )
                                population[i] = evaluated
                                row = {
                                    "gen": gen, "idx": i,
                                    "attn_impl": evaluated.attn_impl,
                                    "head_masks": json.dumps({k: sorted(list(v)) for k, v in evaluated.head_masks.items()}),
                                    "neuron_masks": json.dumps({k: sorted(list(v)) for k, v in evaluated.neuron_masks.items()}),
                                    "energy_j_per_token": evaluated.energy_j_per_token,
                                    "semantic_score": evaluated.semantic_score,
                                    "latency_s": evaluated.latency_s,
                                    "tokens": evaluated.tokens,
                                    "cpu_J": evaluated.cpu_j,
                                    "gpu_J": evaluated.gpu_j,
                                    "ppl": evaluated.ppl,
                                }
                                results_rows.append(row)
                                continue
                            except Exception as e2:
                                print("[error] evaluation retry still failed:", e2)
                        else:
                            print("[warn] Ollama did not recover within the short window; will fallback.")
                    # Final fallback: mark as dominated (keeps loop going)
                    cand.energy_j_per_token = 1e9
                    cand.semantic_score = 0.0

        # Selection
        population = select_next_population(population, args.pop_size)

        # Produce offspring via crossover/mutation + Qwen proposals guided by parents
        offspring: List[Candidate] = []
        parents = population[:]
        random.shuffle(parents)
        for p in parents:
            if random.random() < 0.5:
                try:
                    cfg = ask_qwen_for_config(
                        args.ollama_model,
                        parent_cfg=candidate_to_json(p),
                        retries=args.ollama_retries,
                        initial_delay=args.ollama_initial_delay,
                        backoff=args.ollama_backoff,
                        max_delay=args.ollama_max_delay,
                    )
                    offspring.append(json_to_candidate(cfg))
                except Exception as e:
                    print("[warn] Ollama mutate failed; attempting quick recovery:", e)
                    if ollama_wait_until_ready(timeout_s=30.0, probe_interval=2.0):
                        try:
                            cfg = ask_qwen_for_config(
                                args.ollama_model,
                                parent_cfg=candidate_to_json(p),
                                retries=max(2, args.ollama_retries // 2),
                                initial_delay=args.ollama_initial_delay,
                                backoff=args.ollama_backoff,
                                max_delay=args.ollama_max_delay,
                            )
                            offspring.append(json_to_candidate(cfg))
                            continue
                        except Exception as e2:
                            print("[warn] Ollama mutate retry failed; falling back to local mutation:", e2)
                    offspring.append(mutate(p))
            else:
                q = random.choice(population)
                child = crossover(p, q)
                child = mutate(child)
                offspring.append(child)

            if len(offspring) >= args.offspring_per_parent * len(parents):
                break

        # Add offspring to population for next generation evaluation
        for child in offspring:
            child.energy_j_per_token = float("inf")
            population.append(child)

        # Save results
        if results_rows:
            new = not os.path.exists(args.results_csv)
            with open(args.results_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
                if new:
                    w.writeheader()
                for r in results_rows:
                    w.writerow(r)
            print(f"[results] appended {len(results_rows)} rows to {args.results_csv}")
        else:
            print("Results row is empty")

    # Plot Pareto on final evaluated subset
    if args.plot and results_rows:
        df = pd.DataFrame(results_rows)
        last_gen = df["gen"].max()
        dfg = df[df["gen"] == last_gen].copy()

        def pareto_front(dfx: pd.DataFrame) -> pd.DataFrame:
            arr = dfx[["energy_j_per_token", "semantic_score"]].values
            keep = [True] * len(arr)
            for i, (e_i, s_i) in enumerate(arr):
                if not keep[i]:
                    continue
                for j, (e_j, s_j) in enumerate(arr):
                    if i == j:
                        continue
                    if (e_j <= e_i and s_j >= s_i) and (e_j < e_i or s_j > s_i):
                        keep[i] = False
                        break
            return dfx[keep].sort_values(["energy_j_per_token", "semantic_score"], ascending=[True, False])

        pf = pareto_front(dfg)
        sns.set_context("talk")
        sns.set_style("whitegrid")
        plt.figure(figsize=(9, 7))
        plt.scatter(dfg["energy_j_per_token"], 1.0 - dfg["semantic_score"], s=35, c="#888", label="All (last gen)")
        plt.plot(pf["energy_j_per_token"], 1.0 - pf["semantic_score"], "-o", color="#d62728",
                 label="Pareto (last gen)")
        plt.xlabel("Energy per token (J/token)")
        plt.ylabel("Semantic delta (1 - score)")
        plt.title("NSGA-II: Energy vs Semantic Delta (last generation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("nsga2_pareto.png", dpi=150)
        print("[plot] saved nsga2_pareto.png")


if __name__ == "__main__":
    main()
    print("Execution finished")
