import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.parser import parse_game_log, build_evidence_cards, build_daily_states, chunk_messages_by_day
from src.llm_client import LocalLLM
from src.agents import RoleReasoningAgent, WolfReasoningAgent, FormationAgent, EventExtractionAgent, StateTrackerAgent, fallback_role_prior, fallback_wolf_scores
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
    formation_agent: FormationAgent,
    event_agent: EventExtractionAgent = None,
    state_agent: StateTrackerAgent = None,
    debug_logger=None,
    use_llm: bool = True,
) -> pd.DataFrame:
    players = game_rows["character"].tolist()
    raw_text = load_game_log(corpus_dir, game_index)

    parsed = parse_game_log(raw_text, players)
    evidence_cards = build_evidence_cards(parsed, players)
    daily_states = build_daily_states(parsed, players)

    # Agent 1 + Agent 2: optional LLM extraction/state tracking over day chunks.
    # Regex parsing remains as deterministic backing evidence.
    llm_event_outputs = []
    llm_state = {
        "alive_players": players,
        "dead_players": [],
        "seer_claimers": [],
        "medium_claimers": [],
        "formation": "0-0",
        "gray_players": players,
        "notes": ["initial empty state"],
    }

    if use_llm and event_agent is not None and state_agent is not None:
        chunks_by_day = chunk_messages_by_day(parsed, chunk_size=100, overlap=12)
        for day, chunks in chunks_by_day.items():
            for chunk_i, messages in enumerate(chunks):
                day_key = f"{day}_{chunk_i:02d}" if len(chunks) > 1 else str(day)
                extracted = event_agent.extract(
                    game_index=game_index,
                    day=day_key,
                    players=players,
                    messages=messages,
                    previous_state=llm_state,
                )
                llm_event_outputs.append(extracted)
                llm_state = state_agent.update(
                    game_index=game_index,
                    day=day_key,
                    players=players,
                    previous_state=llm_state,
                    extracted_events=extracted,
                )

        parsed["llm_events"] = llm_event_outputs
        parsed["llm_state"] = llm_state
        # Make the LLM board state visible to downstream prompts without replacing deterministic states.
        daily_states["llm_final"] = llm_state

    if debug_logger:
        debug_logger.save_parsed_game(game_index, parsed)
        debug_logger.save_evidence_cards(game_index, evidence_cards)
        debug_logger.save_json(game_index, "daily_states", daily_states)
        debug_logger.save_json(game_index, "llm_events", {"items": llm_event_outputs})
        debug_logger.save_json(game_index, "llm_state", llm_state)

    formation_analysis = {}
    if use_llm and formation_agent is not None:
        formation_analysis = formation_agent.predict(
            players=players,
            daily_states=daily_states,
            game_index=game_index,
        )

    # Default probabilities from evidence.
    if use_llm:
        role_scores = role_agent.predict(
            players,
            evidence_cards,
            parsed,
            daily_states=daily_states,
            formation_analysis=formation_analysis,
            game_index=game_index,
        )

        wolf_scores = wolf_agent.predict(
            players,
            evidence_cards,
            parsed,
            daily_states=daily_states,
            formation_analysis=formation_analysis,
            game_index=game_index,
        )
    else:
        role_scores = fallback_role_prior(players, parsed)
        wolf_scores = fallback_wolf_scores(players, parsed)

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

    # Gerd is the fixed system victim/optimist in these logs, not an ordinary hidden-role candidate.
    for p in players:
        if p == "Optimist Gerd" or p.strip().lower() == "gerd":
            assigned_roles[p] = "Villager"
            final_wolf_scores[p] = 0.01

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
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission_public.csv")
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--debug_dir", type=str, default="debug_runs")
    parser.add_argument("--disable_debug", action="store_true")
    args = parser.parse_args()

    print(args.roles_csv)
    roles_df = pd.read_csv(args.roles_csv)
    corpus_dir = Path(args.corpus_dir)

    if not args.no_llm and not args.model_path:
        raise ValueError("--model_path is required unless --no_llm is set")

    llm = None
    if not args.no_llm:
        llm = LocalLLM(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            max_tokens=args.max_tokens,
        )

    debug_logger = DebugLogger(
        debug_dir=args.debug_dir,
        enabled=not args.disable_debug,
    )

    role_agent = RoleReasoningAgent(llm, debug_logger=debug_logger) if llm else None
    wolf_agent = WolfReasoningAgent(llm, debug_logger=debug_logger) if llm else None
    formation_agent = FormationAgent(llm, debug_logger=debug_logger) if llm else None
    event_agent = EventExtractionAgent(llm, debug_logger=debug_logger) if llm else None
    state_agent = StateTrackerAgent(llm, debug_logger=debug_logger) if llm else None

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
                formation_agent=formation_agent,
                event_agent=event_agent,
                state_agent=state_agent,
                debug_logger=debug_logger,
                use_llm=not args.no_llm,
            )
            outputs.append(result)
        except Exception as e:
            print(f"[WARN] Game {game_index} failed: {e}")
            fallback = game_rows.copy()
            fallback["role"] = "Villager"
            fallback["wolf_score"] = 0.2
            mask_gerd = fallback["character"].astype(str).str.strip().str.lower().isin(["optimist gerd", "gerd"])
            fallback.loc[mask_gerd, "wolf_score"] = 0.01
            outputs.append(fallback)

    submission = pd.concat(outputs, ignore_index=True)
    submission = submission[["id", "index", "character", "role", "wolf_score"]]
    submission.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
    