import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--out", type=str, default="error_analysis.csv")
    args = parser.parse_args()

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(args.gt)

    # Normalize column names
    pred = pred.rename(columns={
        "role": "pred_role",
        "wolf_score": "pred_wolf_score",
    })

    gt = gt.rename(columns={
        "role": "true_role",
        "wolf_score": "true_wolf_score",
    })

    # Keep only useful GT columns if present
    merge_keys = ["id", "index", "character"]
    missing_pred = [c for c in merge_keys if c not in pred.columns]
    missing_gt = [c for c in merge_keys if c not in gt.columns]

    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {missing_pred}")
    if missing_gt:
        raise ValueError(f"GT file missing columns: {missing_gt}")

    merged = pred.merge(
        gt,
        on=merge_keys,
        how="left",
        suffixes=("_pred", "_gt"),
    )

    if "pred_wolf_score" not in merged.columns:
        raise ValueError(
            f"pred_wolf_score not found after merge. Columns are: {list(merged.columns)}"
        )

    merged["role_correct"] = merged["pred_role"] == merged["true_role"]
    merged["true_is_wolf"] = merged["true_role"] == "Werewolf"
    merged["pred_is_wolf_role"] = merged["pred_role"] == "Werewolf"

    # Rank by predicted wolf score within each game
    merged = merged.sort_values(
        ["index", "pred_wolf_score"],
        ascending=[True, False],
    )

    merged["wolf_rank_in_game"] = (
        merged.groupby("index")["pred_wolf_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    # Basic diagnostic columns
    keep_cols = [
        "id",
        "index",
        "character",
        "true_role",
        "pred_role",
        "role_correct",
        "true_is_wolf",
        "pred_is_wolf_role",
        "pred_wolf_score",
        "wolf_rank_in_game",
    ]

    if "true_wolf_score" in merged.columns:
        keep_cols.append("true_wolf_score")

    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(args.out, index=False)

    print(f"Saved error analysis to {args.out}")
    print()
    print("Role accuracy:")
    print(merged["role_correct"].mean())
    print()
    print("Confusion matrix:")
    print(pd.crosstab(merged["true_role"], merged["pred_role"], margins=True))
    print()
    print("True wolves by predicted rank:")
    print(
        merged[merged["true_is_wolf"]]
        [["index", "character", "true_role", "pred_role", "pred_wolf_score", "wolf_rank_in_game"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()