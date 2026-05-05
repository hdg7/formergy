python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-precision-knobs \
  --enable-memory-knobs \
  --enable-structure-knobs \
  --enable-algorithm-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexAll1.csv --plot

python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-precision-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexPrec1.csv --plot

python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-memory-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexMemory1.csv --plot

python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-structure-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexStructure1.csv --plot

python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --enable-algorithm-knobs \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexAlgorithm1.csv --plot

python formegy.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --ollama-model qwen3-coder-next \
  --prompt-file promptsShort.txt \
  --max-new-tokens 260 --min-new-tokens 160 \
  --gpu-indexes 0,1 \
  --pop-size 20 --generations 32 \
  --results-csv nsga2_results_tinyllamaComplexNone1.csv --plot

