import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.parser import parse_game_log, build_evidence_cards
from src.llm_client import LocalLLM
from src.agents import RoleReasoningAgent, WolfReasoningAgent
from src.solver import constrained_role_assignment, normalize_wolf_scores
from src.debug_logger import DebugLogger

def load_game_log(corpus_dir: Path, game_index: str) -> str:
    """
    Expected filename example:
    Werewolf_Prediction_Dataset/01.txt
    """
    path = corpus_dir / f"{game_index}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing game log: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def infer_one_game(
    game_index: str,
    game_rows: pd.DataFrame,
    corpus_dir: Path,
    role_agent: RoleReasoningAgent,
    wolf_agent: WolfReasoningAgent,
    debug_logger=None,
    use_llm: bool = True,
) -> pd.DataFrame:
    players = game_rows["character"].tolist()
    raw_text = load_game_log(corpus_dir, game_index)

    parsed = parse_game_log(raw_text, players)
    evidence_cards = build_evidence_cards(parsed, players)
    if debug_logger:
        debug_logger.save_parsed_game(game_index, parsed)
        debug_logger.save_evidence_cards(game_index, evidence_cards)
    if debug_logger and isinstance(evidence_cards, dict):
        debug_logger.save_json(game_index, "global_card", evidence_cards.get("_global", {}))
        
    # Default probabilities from rule-based evidence.
    role_scores = {}
    wolf_scores = {}

    if use_llm:
        role_scores = role_agent.predict(players, evidence_cards, parsed, game_index=game_index)
        wolf_scores = wolf_agent.predict(players, evidence_cards, parsed, game_index=game_index)
    else:
        role_scores = {}
        wolf_scores = {}

    assigned_roles = constrained_role_assignment(
        players=players,
        role_scores=role_scores,
        parsed=parsed,
    )

    final_wolf_scores = normalize_wolf_scores(
        players=players,
        wolf_scores=wolf_scores,
        role_scores=role_scores,
        assigned_roles=assigned_roles,
        parsed=parsed,
    )

    if debug_logger:
        debug_logger.save_json(game_index, "solver_assigned_roles", assigned_roles)
        debug_logger.save_json(game_index, "final_wolf_scores", final_wolf_scores)

    out = game_rows.copy()
    out["role"] = out["character"].map(assigned_roles)
    out["wolf_score"] = out["character"].map(final_wolf_scores)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles_csv", type=str, default="Werewolf_Prediction_Dataset/public/roles.csv")
    parser.add_argument("--corpus_dir", type=str, default="Werewolf_Prediction_Dataset")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="submission_public.csv")
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--n_ctx", type=int, default=8192)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--debug_dir", type=str, default="debug_runs")
    parser.add_argument("--disable_debug", action="store_true")
    args = parser.parse_args()

    print(args.roles_csv)
    roles_df = pd.read_csv(args.roles_csv)
    corpus_dir = Path(args.corpus_dir)

    llm = LocalLLM(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )

    debug_logger = DebugLogger(
        debug_dir=args.debug_dir,
        enabled=not args.disable_debug,
    )

    role_agent = RoleReasoningAgent(llm, debug_logger=debug_logger)
    wolf_agent = WolfReasoningAgent(llm, debug_logger=debug_logger)

    outputs = []
    for game_index, game_rows in tqdm(roles_df.groupby("index")):
        game_index = str(game_index).zfill(2)

        try:
            result = infer_one_game(
                game_index=game_index,
                game_rows=game_rows,
                corpus_dir=corpus_dir,
                role_agent=role_agent,
                wolf_agent=wolf_agent,
                debug_logger=debug_logger,
                use_llm=not args.no_llm,
            )
            outputs.append(result)
        except Exception as e:
            raise NotImplementedError(f"Game {game_index} failed: {e}")
            print(f"[WARN] Game {game_index} failed: {e}")
            fallback = game_rows.copy()
            fallback["role"] = "Villager"
            fallback["wolf_score"] = 0.2
            outputs.append(fallback)

    submission = pd.concat(outputs, ignore_index=True)
    submission = submission[["id", "index", "character", "role", "wolf_score"]]
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
    