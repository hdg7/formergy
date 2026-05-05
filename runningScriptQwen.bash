python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-precision-knobs \
  --enable-memory-knobs \
  --enable-structure-knobs \
  --enable-algorithm-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexAll1.csv --plot

python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-precision-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexPrec1.csv --plot

python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-memory-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexMemory1.csv --plot

python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-structure-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexStructure1.csv --plot

python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-algorithm-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexAlgorithm1.csv --plot

python formegy.py \
  --model-id Qwen/Qwen2.5-Coder-3B-Instruct \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_qwenComplexNone1.csv --plot
