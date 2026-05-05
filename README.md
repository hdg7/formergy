# Formegy

**Formegy** is a research-oriented repository exploring how **evolutionary algorithms (NSGA-II)** can be used to **optimize the energy consumption of Large Language Models (LLMs)** while preserving semantic quality. The framework jointly considers *energy efficiency* and *output semantics* as multi-objective optimization targets.

The current implementation focuses on LLaMA-style models (e.g. TinyLlama) and integrates **GPU/CPU energy measurement**, **semantic evaluation**, and **LLM-assisted candidate generation** using Ollama.

---

## Repository Contents

### `formegy.py`

The core Python implementation of the Formegy framework. This script implements an **NSGA-II multi-objective evolutionary optimizer** that searches for model-level configurations reducing energy usage while maintaining acceptable semantic fidelity.

Key responsibilities of this file include:

- **Energy Measurement**
  - GPU energy tracking via NVIDIA NVML (per-GPU cumulative energy deltas).
  - CPU energy estimation via CodeCarbon (GPU tracking disabled to avoid double counting).

- **Semantic Evaluation Suite**
  - Cosine similarity using *BAAI/bge-large-en-v1.5* embeddings.
  - Natural Language Inference (entailment probability) using *roberta-large-mnli*.
  - Fluency estimation via GPT‑2 perplexity.
  - A composite semantic score combining the above metrics.

- **Model Instrumentation for Energy Control**
  - Wrappers around LLaMA attention and MLP modules to:
    - Mask attention heads.
    - Mask MLP neurons.
    - Switch attention implementations (`sdpa`, `eager`, `flash_attention_2`).

- **LLM-Assisted Search (Ollama integration)**
  - Uses Qwen models hosted in Ollama to propose and mutate candidate sparsity configurations.
  - Robust retry logic with exponential backoff to handle Ollama instability.

- **NSGA-II Optimization**
  - Multi-objective fitness:
    - Minimize *energy per generated token*.
    - Minimize *semantic degradation*.
  - Implements fast non-dominated sorting, crowding distance, crossover, and mutation.

This script is intended to be run from the command line and produces CSV logs and optional plots showing Pareto fronts.

---

### `runningScript.bash`

A convenience Bash script for running multiple Formegy experiments using **TinyLlama**.

It launches a series of NSGA-II runs that isolate different families of optimization knobs:

- **All knobs enabled** (precision, memory, structure, algorithm).
- **Precision-only optimizations**.
- **Memory-only optimizations**.
- **Structural optimizations** (attention heads / neurons).
- **Algorithmic knobs only**.
- **No optimization knobs** (baseline evolutionary search).

Each run:

- Uses the same prompt set.
- Fixes population size and number of generations.
- Writes results to a separate CSV file.
- Optionally produces plots of the Pareto front.

This script is useful for **ablation studies** comparing the impact of different optimization families on energy–semantic trade-offs.

---

### `runningScriptQwen.bash`

A variant of `runningScript.bash` configured to run the same experimental protocol but targeting **Qwen-based models** instead of TinyLlama.

Differences relative to `runningScript.bash`:

- Uses `Qwen/Qwen2.5-Coder-3B-Instruct` as the evaluation model.
- Still relies on Qwen models served via Ollama for evolutionary proposals.
- Generates separate result CSVs for Qwen experiments.

This script enables **cross-model comparison**, allowing analysis of how evolutionary energy optimization behaves across different LLM architectures and scales.

---

## Typical Workflow

1. Define a prompt set (e.g. `promptsShort.txt`).
2. Choose the target model (TinyLlama or Qwen).
3. Run one of the provided Bash scripts.
4. Inspect generated CSV files and Pareto plots to study energy vs. semantic trade-offs.

---

## Research Use Case

Formegy is primarily intended for:

- Green AI and energy-aware ML research.
- Studying structural sparsity and algorithmic knobs in LLMs.
- Exploring LLM-in-the-loop evolutionary optimization.
- Reproducible experiments on energy–quality trade-offs.

---

## Status

This repository is **experimental and research-focused**. APIs, metrics, and optimization strategies may evolve as the project matures.

---

## License

Apache 2.0

