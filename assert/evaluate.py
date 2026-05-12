#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np


GT_PATH = "/home/pywu/Downloads/zhong/AI_2026/hw2/Werewolf_Prediction_Dataset/public/roles_with_gt.csv"


def normalize_index(x):
    """
    Normalize game index.
    Examples:
      1  -> "1"
      01 -> "1"
    """
    try:
        return str(int(x))
    except Exception:
        return str(x).strip()


def macro_f1_score(y_true, y_pred, labels=None):
    """
    Compute macro-F1 across all roles.

    F1_t = 2 * P_t * R_t / (P_t + R_t)
    Macro-F1 = average over all roles
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    f1s = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1s.append(f1)

    return float(np.mean(f1s))


def average_precision_score_binary(y_true, y_score):
    """
    Compute Average Precision for binary classification.

    AP = (1 / N_pos) * sum_k P(k) * rel(k)

    where:
      P(k) = precision among top k samples
      rel(k) = 1 if the k-th sample is positive, else 0
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = np.sum(y_true)

    if n_pos == 0:
        return 0.0

    # Sort by wolf_score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp_cumsum = np.cumsum(y_true_sorted)
    ranks = np.arange(1, len(y_true_sorted) + 1)

    precision_at_k = tp_cumsum / ranks
    ap = np.sum(precision_at_k * y_true_sorted) / n_pos

    return float(ap)

def evaluate(pred_path, gt_path=GT_PATH):
    pred = pd.read_csv(pred_path)
    gt = pd.read_csv(gt_path)

    required_pred_cols = {"index", "character", "role", "wolf_score"}
    required_gt_cols = {"index", "character", "role"}

    missing_pred = required_pred_cols - set(pred.columns)
    missing_gt = required_gt_cols - set(gt.columns)

    if missing_pred:
        raise ValueError(f"Prediction CSV missing columns: {missing_pred}")

    if missing_gt:
        raise ValueError(f"Ground truth CSV missing columns: {missing_gt}")

    pred = pred.copy()
    gt = gt.copy()

    pred["index_norm"] = pred["index"].apply(normalize_index)
    gt["index_norm"] = gt["index"].apply(normalize_index)

    pred["character_norm"] = pred["character"].astype(str).str.strip()
    gt["character_norm"] = gt["character"].astype(str).str.strip()

    merged = gt.merge(
        pred,
        on=["index_norm", "character_norm"],
        suffixes=("_true", "_pred"),
        how="left",
    )

    if merged["role_pred"].isna().any():
        missing = merged[merged["role_pred"].isna()][
            ["index_true", "character_true", "role_true"]
        ]
        raise ValueError(
            "Some ground-truth rows do not have predictions.\n"
            f"Missing rows:\n{missing.to_string(index=False)}"
        )

    if merged["wolf_score_pred"].isna().any():
        raise ValueError("Some rows have missing wolf_score values.")

    y_true_role = merged["role_true"].astype(str).values
    y_pred_role = merged["role_pred"].astype(str).values

    labels = sorted(gt["role"].astype(str).unique())

    macro_f1 = macro_f1_score(y_true_role, y_pred_role, labels=labels)

    y_true_wolf = (merged["role_true"].astype(str) == "Werewolf").astype(int).values
    wolf_score = merged["wolf_score_pred"].astype(float).values

    wolf_ap = average_precision_score_binary(y_true_wolf, wolf_score)

    final_score = 0.4 * macro_f1 + 0.6 * wolf_ap

    return {
        "macro_f1_all_roles": macro_f1,
        "wolf_only_ap": wolf_ap,
        "final_score": final_score,
        "num_samples": len(merged),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_csv",
        help="Path to prediction CSV. Must contain: index, character, role, wolf_score",
    )
    parser.add_argument(
        "--gt_csv",
        default=GT_PATH,
        help=f"Path to ground truth CSV. Default: {GT_PATH}",
    )

    args = parser.parse_args()

    result = evaluate(args.pred_csv, args.gt_csv)

    print(f"num_samples: {result['num_samples']}")
    print(f"macro_f1_all_roles: {result['macro_f1_all_roles']:.6f}")
    print(f"wolf_only_ap: {result['wolf_only_ap']:.6f}")
    print(f"final_score: {result['final_score']:.6f}")


if __name__ == "__main__":
    main()