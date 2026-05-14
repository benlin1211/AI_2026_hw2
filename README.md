# Werewolf Fast v4 Hard-Fact Fix

This is the fast deterministic multi-agent Werewolf prediction pipeline.

The current version keeps the original fast pipeline design, but adds a hard-fact repair layer for three high-impact failure modes:

1. public event extraction errors,
2. noisy claim / CO parsing,
3. illegal role assignments that violate public game facts.

`main.py` does not call an LLM. The system is intended to run quickly and produce reproducible submissions from deterministic parsing, scoring, constrained world search, and final audit rules.

## Current pipeline overview

For each game, the system runs the following stages.

1. **ParserAgent** — `src/parser.py`
   - Splits the raw log into days and message-level records.
   - Extracts initial claims, votes, soft reads, and rough death mentions.
   - Builds player evidence cards, baseline daily states, compact objective packs, and strategic snippets.

2. **ObjectiveEventAgent** — `src/events.py`
   - Repairs noisy claims from the parser.
   - Extracts public executions, night kills, sudden deaths, GJ/no-attack signals, claim withdrawals, hunter claims, guard reports, formation hints, and ability-result candidates.
   - Uses stricter rules for hard results so ordinary soft reads are less likely to become fake Seer/Medium results.

3. **HardConstraintAgent** — `src/hard_constraints.py`
   - Converts public rules into hard constraints.
   - Forces Gerd to Villager.
   - Marks night-killed players as `impossible_wolves` because wolves kill human players at night.
   - Applies these constraints to role scores, wolf priors, beam candidates, aggregate wolf scores, and final audit checks.

4. **BoardStateAgent** — `src/board_state.py`
   - Builds cumulative board states by day.
   - Tracks alive/dead players, role counts, claimers, gray players, formations, hard results, executions, night kills, GJ days, hunter claims, and guard reports.
   - Uses formation hints when raw claim parsing is incomplete.

5. **FormationPolicyAgent** — `src/formation_policy.py`
   - Classifies the latest board formation, such as `3-1`, `2-2`, or `3-2`.
   - Produces a small policy summary that tells downstream scoring which reasoning mode matters most, such as seer-credit comparison, medium-axis play, line battle, or gray LW search.

6. **InteractionAgent** — `src/interaction.py` + `src/stepwise_scorer.py`
   - Builds a rule-based interaction graph from votes, soft reads, and strategic snippets.
   - Converts pressure, vote pushes, claim challenges, defenses, townreads, and check preferences into player-level features.
   - Produces a continuous wolf prior for ranking, not a final role decision.

7. **GuidelineScorerAgent** — `src/guideline_scorer_fast.py`
   - Adds small deterministic score deltas for practical Werewolf heuristics.
   - Handles contested claims, hard black/white results, panda targets, medium results, push chains, GJ/hunter signals, and formation-specific adjustments.

8. **BeamRoleSolver** — `src/beam_solver.py`
   - Runs candidate-pruned top-K world search under fixed role counts.
   - Combines role scores, wolf priors, hard results, objective events, guideline deltas, and hard constraints.
   - Produces legal role worlds and role marginals without exhaustive enumeration of all 16-player assignments.

9. **WolfScoreAggregator** — `src/aggregator.py`
   - Aggregates continuous `wolf_score` from world marginals, wolf priors, base wolf scores, guideline scores, hard results, and final role labels.
   - Treats `wolf_score` as an AP/ranking score, not merely as the final `Werewolf` role assignment.
   - Preserves an overbooked suspicious band while still respecting hard non-wolf constraints.

10. **Final Audit** — `src/audit.py`
    - Repairs final role labels to legal role counts.
    - Enforces Gerd and `impossible_wolves` constraints.
    - Emits debug reports showing whether final assignments violate hard public facts.

## Mandatory per-game pipeline

1. ParserAgent: `src/parser.py`
2. ObjectiveEventAgent: `src/events.py`
3. HardConstraintAgent: `src/hard_constraints.py`
4. BoardStateAgent: `src/board_state.py`
5. FormationPolicyAgent: `src/formation_policy.py`
6. InteractionAgent: `src/interaction.py` + `src/stepwise_scorer.py`
7. GuidelineScorerAgent: `src/guideline_scorer_fast.py`
8. BeamRoleSolver: `src/beam_solver.py`
9. WolfScoreAggregator: `src/aggregator.py`
10. Final Audit: `src/audit.py`

No LLM calls are made by `main.py`.

## What changed in v4

### 1. Objective event extraction

Public death parsing now supports full names, bracketed aliases, upper-case aliases, underscore names, and translated forms such as `[LIZA]`, `[Friedel the Sister]`, and `[FATHER_JIMZON]`.

Sudden-death parsing is stricter and no longer scans loosely across an entire day for any `sudden death` mention.

Ability-result extraction is stricter. It prefers compact result lines and explicit report formats, and it avoids treating ordinary discussion, GS lists, hypothetical statements, or soft reads as hard Seer/Medium results.

### 2. Claim extraction and repair

Bare short self-claims such as `Seer claim` are accepted only when the message is effectively a self-claim.

Third-party confirmations are only used when they are compact explicit confirmation blocks.

`Not Seer/Medium` repair now catches translated “I can’t see the Seer / Medium” forms.

Conflict resolution removes role COs that contradict explicit self-denial unless the later CO is an explicit self-claim.

Hypothetical, parenthetical, or section-header role mentions are rejected when they do not represent the speaker's own role claim.

### 3. Hard constraints

`src/hard_constraints.py` derives public-rule constraints from objective events.

Night-killed players are marked as `impossible_wolves`.

Role scores, wolf priors, beam candidate sets, aggregate wolf scores, and final audit all respect `impossible_wolves`.

Debug output now writes:

- `hard_constraints_fast_parsed.json`
- `hard_constraint_report_fast_parsed.json`

## Debug outputs

When debug mode is enabled, the pipeline writes intermediate files for each game. The most useful files are:

- `parsed_game.json` — raw parser output by day and message.
- `objective_events_fast_parsed.json` — public deaths, ability results, hunter/GJ events, and formation hints.
- `hard_constraints_fast_parsed.json` — forced roles and forbidden roles derived from public facts.
- `hard_constraint_report_fast_parsed.json` — final hard-constraint validation report.
- `board_states_fast_parsed.json` — cumulative board state after each day.
- `objective_pack_fast_parsed.json` — compact facts used by scorers and solvers.
- `interaction_graph_fast_parsed.json` — rule-based interaction edges.
- `wolf_prior_fast_parsed.json` — stepwise wolf prior.
- `guideline_scores_fast_parsed.json` — deterministic guideline deltas.
- `world_result_fast_parsed.json` — top-K legal worlds and role marginals.
- `assigned_roles_fast_parsed.json` — final role labels.
- `final_wolf_scores_fast_parsed.json` — final submitted wolf scores.

## Validation notes

The hard-fact fix was first validated on the provided `01.txt` sample.

Expected improvements:

- Executions are extracted correctly.
- Night kills are extracted correctly.
- Night-killed players are no longer assigned Werewolf.
- Gerd remains forced to Villager.
- The final audit can catch impossible-wolf assignments before submission.

The current version also improves public leaderboard performance compared with the earlier baseline, mainly because hard public facts now constrain the role solver and wolf-score aggregator.

## Known limitations

Claim extraction is still sensitive to translated phrasing. Some logs use role words as section headers, direct-address labels, or quoted discussion rather than self-claims.

Ability-result extraction is conservative, but it can still confuse a true result with a perspective statement when translated text is ambiguous.

Formation hints can recover formations such as `3-1` even when claimers are incomplete, but missing claimant identities can still affect role assignment.

Soft reads and interaction edges are weak evidence. They are useful for ranking, but they should not override deaths, claims, hard results, and role-count constraints.

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

## Run

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/public/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/public \
  --output public_submission.csv \
  --debug_dir debug_runs_v4
```

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/private/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/private \
  --output private_submission.csv \
  --debug_dir debug_runs_v4
```

Debug mode:

```bash
python main.py \
  --roles_csv Werewolf_Prediction_Dataset/public/roles.csv \
  --corpus_dir Werewolf_Prediction_Dataset/public \
  --output submission_fast.csv \
  --debug_dir debug_runs_fast
```

Speed knobs:

```bash
--beam_top_k 128
--max_role_candidates 5
--max_wolf_candidates 8
--max_hunter_candidates 5
```

Higher values are slower but may improve role consistency.

## Design notes

- `wolf_score` is a marginal/ranking score, not just a byproduct of final role assignment.
- Fake claimers are not automatically scored as Werewolf because Madman can fake claims.
- Night-killed players are hard non-wolves under the public rules.
- Seer human and Medium human mean non-Werewolf, not necessarily Villager.
- GJ, claim withdrawal, panda targets, medium results, and push chains are converted into small deterministic score deltas.
- Top-K worlds are candidate-pruned; this avoids exhaustive 16-player enumeration.
- Final role assignment obeys legal role counts; `wolf_score` is allowed to preserve a broader suspicious band for ranking metrics.
