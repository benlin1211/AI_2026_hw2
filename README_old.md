# HW2 Multi-Agent Werewolf Prediction

This version implements the proposed multi-agent pipeline for the Werewolf role/wolf-score task.

## What changed

The pipeline is now:

1. **Parser / deterministic extractor**
   - Splits logs by `===== Day X =====`.
   - Parses message-level records.
   - Extracts first-person CO / non-CO, hard divination results, soft reads, votes, and deaths.

2. **Agent 1: EventExtractionAgent**
   - Runs over day-level chunks instead of sending the full game log at once.
   - Converts noisy roleplay/chat into structured events.
   - Keeps CO, results, votes, deaths, attacks, suspicions, townreads, formations, and contradictions.
   - Ignores food/RP/greetings/jokes unless strategically relevant.

3. **Agent 2: StateTrackerAgent**
   - Merges extracted events into a cumulative public board state.
   - Tracks formation, claimers, hard results, deaths, votes, conflicts, and gray players.

4. **Agent 3: FormationAgent**
   - Analyzes latest Seer/Medium formation and claimant alignment patterns.
   - Its output is now actually passed into the role and wolf agents.

5. **Agent 4: RoleReasoningAgent + WolfReasoningAgent**
   - Estimates role probabilities and Werewolf probability.
   - Treats Madman as wolf-aligned but not a Werewolf for `wolf_score`.
   - Separates hard Seer/Medium results from soft white/black reads.

6. **Final solver**
   - Produces valid `role` and continuous `wolf_score`.
   - Keeps `Optimist Gerd` / `Gerd` as a fixed non-wolf baseline.

## Files

```text
main.py
src/
  __init__.py
  agents.py
  debug_logger.py
  llm_client.py
  parser.py
  prompts.py
  solver.py
  solver_all_possible.py
```

## Environment

```bash
pip install pandas tqdm huggingface_hub
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

Download a local model, for example Qwen2.5 7B Instruct GGUF:

```bash
mkdir -p models
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m*.gguf" \
  --local-dir models
```

Q4_K_M is recommended for the 12GB VRAM restriction. Q5_K_M can work only when memory allows.

## Usage

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/public/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/public \
  --model_path models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
  --output public_submission.csv \
  --n_ctx 16384 \
  --n_gpu_layers -1 \
  --debug_dir debug_public
```

Private set:

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/private/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/private \
  --model_path models/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
  --output private_submission.csv \
  --n_ctx 16384 \
  --n_gpu_layers -1 \
  --debug_dir debug_private
```

Rule-only fallback, mainly for debugging:

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/public/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/public \
  --no_llm \
  --output rule_only_submission.csv
```

## Output

The output schema is fixed:

```csv
id,index,character,role,wolf_score
```

Allowed roles:

```text
Villager, Werewolf, Seer, Medium, Madman, Hunter
```
