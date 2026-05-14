"""Fast deterministic multi-agent Werewolf predictor.

Mandatory pipeline per game:
1. ParserAgent              -> parser.py
2. ObjectiveEventAgent      -> events.py
3. BoardStateAgent          -> board_state.py
4. FormationPolicyAgent     -> formation_policy.py
5. InteractionAgent         -> interaction.py + stepwise_scorer.py
6. GuidelineScorerAgent     -> guideline_scorer_fast.py
7. WolfScoreAggregator      -> aggregator.py
8. BeamRoleSolver           -> beam_solver.py

No LLM calls are made in this entrypoint. This is intended as the fast default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.parser import (
    parse_game_log,
    build_evidence_cards,
    build_daily_states,
    build_objective_pack,
    build_strategic_snippets,
)
from src.events import extract_objective_events, extract_repaired_claims, apply_claim_repairs_to_parsed
from src.board_state import build_fast_board_states
from src.formation_policy import analyze_formation_policy
from src.interaction import build_interaction_graph
from src.stepwise_scorer import score_interactions, compute_wolf_feature_prior
from src.agents import fallback_role_prior, fallback_wolf_scores
from src.solver import constrained_role_assignment
from src.beam_solver import solve_topk_worlds, best_assignment_from_worlds
from src.guideline_scorer_fast import score_guidelines
from src.aggregator import aggregate_wolf_scores, assign_roles_from_marginals
from src.audit import audit_and_fix
from src.hard_constraints import (
    derive_hard_constraints,
    attach_constraints_to_pack,
    apply_hard_constraints_to_role_scores,
    apply_hard_constraints_to_wolf_scores,
    hard_constraint_report,
)
from src.debug_logger import DebugLogger


def load_game_log(corpus_dir: Path, game_index: str) -> str:
    path = corpus_dir / f"{game_index}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing game log: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def infer_one_game_fast(
    game_index: str,
    game_rows: pd.DataFrame,
    corpus_dir: Path,
    debug_logger: Optional[DebugLogger] = None,
    beam_top_k: int = 256,
    max_role_candidates: int = 6,
    max_wolf_candidates: int = 9,
    max_hunter_candidates: int = 6,
) -> pd.DataFrame:
    players = game_rows["character"].tolist()
    raw_text = load_game_log(corpus_dir, game_index)

    # 1. ParserAgent.
    parsed = parse_game_log(raw_text, players)

    # Claim repair is still deterministic. It fixes common translated-log issues:
    # bare "Seer claim:" self-claims and false positives from discussing another
    # player's claim.
    repaired_claims = extract_repaired_claims(raw_text, players)
    parsed = apply_claim_repairs_to_parsed(parsed, repaired_claims)

    evidence_cards = build_evidence_cards(parsed, players)
    daily_states_baseline = build_daily_states(parsed, players)
    objective_pack = build_objective_pack(parsed, players)
    strategic_snippets = build_strategic_snippets(parsed, players)

    # 2. ObjectiveEventAgent.
    objective_events = extract_objective_events(raw_text, parsed, players)
    objective_events["repaired_claims"] = repaired_claims

    # Merge event-derived fields into objective_pack for downstream scorer.
    objective_pack = dict(objective_pack)
    objective_pack["hunter_claimers"] = list(dict.fromkeys([x.get("player") for x in objective_events.get("hunter_claims", []) if x.get("player")]))
    if objective_events.get("seer_results"):
        objective_pack["hard_results"] = objective_events["seer_results"]

    # Public-rule hard constraints must be attached before board state, scoring, and beam search.
    hard_constraints = derive_hard_constraints(players, objective_events, objective_pack)
    objective_events["hard_constraints"] = hard_constraints
    objective_pack = attach_constraints_to_pack(objective_pack, hard_constraints)

    # 3. BoardStateAgent.
    board_states = build_fast_board_states(parsed, objective_events, players)

    # 4. FormationPolicyAgent.
    formation_policy = analyze_formation_policy(board_states)

    # 5. InteractionAgent.
    interaction_graph = build_interaction_graph(strategic_snippets, players, objective_pack=objective_pack)
    stepwise_scores = score_interactions(players, interaction_graph, objective_pack)
    wolf_prior = compute_wolf_feature_prior(players, stepwise_scores, objective_pack)

    # 6. GuidelineScorerAgent.
    guideline_scores = score_guidelines(
        players=players,
        board_states=board_states,
        objective_events=objective_events,
        interaction_graph=interaction_graph,
        objective_pack=objective_pack,
    )

    # Fast deterministic role/wolf priors.
    role_scores = fallback_role_prior(players, parsed)
    base_wolf_scores = fallback_wolf_scores(players, parsed)

    # Apply hard public constraints before the beam solver.
    role_scores = apply_hard_constraints_to_role_scores(players, role_scores, hard_constraints)
    base_wolf_scores = apply_hard_constraints_to_wolf_scores(players, base_wolf_scores, hard_constraints)
    wolf_prior = apply_hard_constraints_to_wolf_scores(players, wolf_prior, hard_constraints)

    # 8. BeamRoleSolver / world marginals.
    world_result = solve_topk_worlds(
        players=players,
        role_scores=role_scores,
        wolf_scores=base_wolf_scores,
        wolf_prior=wolf_prior,
        objective_pack=objective_pack,
        objective_events=objective_events,
        guideline_scores=guideline_scores,
        top_k=beam_top_k,
        max_role_candidates=max_role_candidates,
        max_wolf_candidates=max_wolf_candidates,
        max_hunter_candidates=max_hunter_candidates,
    )

    fallback_assignment = constrained_role_assignment(players=players, role_scores=role_scores, parsed=parsed)
    world_assignment = best_assignment_from_worlds(world_result, players)
    assigned_roles = assign_roles_from_marginals(players, world_result, fallback_assignment=world_assignment or fallback_assignment)

    # 7. WolfScoreAggregator.
    final_wolf_scores = aggregate_wolf_scores(
        players=players,
        role_scores=role_scores,
        base_wolf_scores=base_wolf_scores,
        wolf_prior=wolf_prior,
        guideline_scores=guideline_scores,
        world_result=world_result,
        objective_pack=objective_pack,
        objective_events=objective_events,
        assigned_roles=assigned_roles,
    )

    assigned_roles, final_wolf_scores = audit_and_fix(players, assigned_roles, final_wolf_scores, role_scores, hard_constraints=hard_constraints)
    constraint_report = hard_constraint_report(players, assigned_roles, hard_constraints)

    if debug_logger:
        debug_logger.save_parsed_game(game_index, parsed)
        debug_logger.save_evidence_cards(game_index, evidence_cards)
        debug_logger.save_json(game_index, "daily_states_baseline", daily_states_baseline)
        debug_logger.save_json(game_index, "objective_pack_fast", objective_pack)
        debug_logger.save_json(game_index, "objective_events_fast", objective_events)
        debug_logger.save_json(game_index, "hard_constraints_fast", hard_constraints)
        debug_logger.save_json(game_index, "hard_constraint_report_fast", constraint_report)
        debug_logger.save_json(game_index, "board_states_fast", board_states)
        debug_logger.save_json(game_index, "formation_policy_fast", formation_policy)
        debug_logger.save_json(game_index, "strategic_snippets_fast", {"items": strategic_snippets})
        debug_logger.save_json(game_index, "interaction_graph_fast", interaction_graph)
        debug_logger.save_json(game_index, "stepwise_scores_fast", stepwise_scores)
        debug_logger.save_json(game_index, "wolf_prior_fast", wolf_prior)
        debug_logger.save_json(game_index, "guideline_scores_fast", guideline_scores)
        debug_logger.save_json(game_index, "world_result_fast", world_result)
        debug_logger.save_json(game_index, "assigned_roles_fast", assigned_roles)
        debug_logger.save_json(game_index, "final_wolf_scores_fast", final_wolf_scores)

    out = game_rows.copy()
    out["role"] = out["character"].map(assigned_roles)
    out["wolf_score"] = out["character"].map(final_wolf_scores)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles_csv", type=str, default="Werewolf_Prediction_Dataset/public/roles.csv")
    parser.add_argument("--corpus_dir", type=str, default="Werewolf_Prediction_Dataset/public")
    parser.add_argument("--output", type=str, default="submission_fast.csv")
    parser.add_argument("--debug_dir", type=str, default="debug_runs_fast")
    parser.add_argument("--disable_debug", action="store_true")
    parser.add_argument("--beam_top_k", type=int, default=256)
    parser.add_argument("--max_role_candidates", type=int, default=6)
    parser.add_argument("--max_wolf_candidates", type=int, default=9)
    parser.add_argument("--max_hunter_candidates", type=int, default=6)
    args = parser.parse_args()

    roles_df = pd.read_csv(args.roles_csv)
    corpus_dir = Path(args.corpus_dir)
    debug_logger = DebugLogger(debug_dir=args.debug_dir, enabled=not args.disable_debug)

    outputs = []
    for game_index, game_rows in tqdm(roles_df.groupby("index")):
        game_index = str(game_index).zfill(2)
        try:
            result = infer_one_game_fast(
                game_index=game_index,
                game_rows=game_rows,
                corpus_dir=corpus_dir,
                debug_logger=debug_logger,
                beam_top_k=args.beam_top_k,
                max_role_candidates=args.max_role_candidates,
                max_wolf_candidates=args.max_wolf_candidates,
                max_hunter_candidates=args.max_hunter_candidates,
            )
            outputs.append(result)
        except Exception as e:
            print(f"[WARN] Game {game_index} failed: {e}")
            if debug_logger:
                debug_logger.save_error(game_index, "infer_one_game_fast", e)
            fallback = game_rows.copy()
            fallback["role"] = "Villager"
            fallback["wolf_score"] = 0.20
            mask_gerd = fallback["character"].astype(str).str.strip().str.lower().isin(["optimist gerd", "gerd"])
            fallback.loc[mask_gerd, "wolf_score"] = 0.01
            outputs.append(fallback)

    submission = pd.concat(outputs, ignore_index=True)
    submission = submission[["id", "index", "character", "role", "wolf_score"]]
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
