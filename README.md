# HW2 Multi-Agent Werewolf Prediction

## Overview

This system predicts each character's role and wolf_score from Werewolf game logs.

The system uses a multi-agent pipeline:

```
hw2_<student-id>/
├── main.py
├── requirements.txt
├── README
└── src/
    ├── llm_client.py
    ├── parser.py
    ├── prompts.py
    ├── agents.py
    └── solver.py
```

## Prepare environment
```
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
<!-- conda install -c nvidia cuda-runtime -y -->
pip install -U \
  nvidia-cuda-runtime-cu12 \
  nvidia-cublas-cu12 \
  nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
```
pip install -U huggingface_hub
mkdir models
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q5_k_m*.gguf" \
  --local-dir models
<!-- cd models
./llama-gguf-split --merge qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf qwen2.5-7b-instruct-q5_k_m.gguf -->
```
## Usage

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/public/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/public/ \
  --model_path models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
  --output public_submission.csv \
  --n_ctx 32768 \
  --n_gpu_layers -1 \
  --debug_dir debug_public

python main.py \
  --roles_csv Werewolf_Prediction_Dataset/private/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/private/ \
  --model_path models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
  --output private_submission.csv \
  --n_ctx 32768 \
  --n_gpu_layers -1 \
  --debug_dir debug_private
```

## Evaluate
```
python assert/evaluate.py public_submission.csv
```# AI_2026_hw2
