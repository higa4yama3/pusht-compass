"""Judge the results against predictions.md.

Ranges for the condition table are copied from predictions.md. Grid thresholds are read from
../results/prediction_intervals.json.

Usage: uv run python scripts/judge.py
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from compass.metrics import spearman

CONDITIONS = Path("../results/conditions.json")
GRID = Path("../results/grid.json")
INTERVALS = Path("../results/prediction_intervals.json")
OUTPUT = Path("../results/judgement.json")


def main() -> None:
    conditions = json.loads(CONDITIONS.read_text(encoding="utf-8"))["conditions"]
    intervals = json.loads(INTERVALS.read_text(encoding="utf-8"))
    judgement = {"conditions": _judge_conditions(conditions)}
    if GRID.exists():
        judgement["grid"] = _judge_grid(json.loads(GRID.read_text(encoding="utf-8")), intervals)
    OUTPUT.write_text(json.dumps(judgement, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(judgement, indent=2, ensure_ascii=False))


def _judge_conditions(conditions: dict) -> list[dict]:
    def successes(name: str) -> int:
        return conditions[name]["summary"]["successes"]

    normal = successes("1_normal")
    told = conditions["4_told_rotated_90"]
    moved = conditions["2_moved_without_compass"]
    noise = [successes(f"6_noise_{sigma}px") for sigma in ("0", "2", "5", "10", "20")]
    checks = [
        ("1 通常", normal, 11 <= normal <= 19),
        (
            "2 今のゴールでの成功",
            successes("2_moved_without_compass"),
            successes("2_moved_without_compass") <= 1,
        ),
        (
            "2 学習時のゴールから 30 px 以内",
            moved["stopped_within_30px_of_training_goal"],
            11 <= moved["stopped_within_30px_of_training_goal"] <= 19,
        ),
        (
            "3 地図あり",
            successes("3_moved_with_compass"),
            7 <= successes("3_moved_with_compass") <= 16,
        ),
        (
            "4 本当のゴールでの成功",
            successes("4_told_rotated_90"),
            successes("4_told_rotated_90") <= 1,
        ),
        ("4 渡したゴールに入った", told["reached_told_goal"], 7 <= told["reached_told_goal"] <= 16),
        (
            "4 向きのずれの中央値 (度)",
            round(told["median_angle_offset_from_true_goal_deg"], 1),
            abs(told["median_angle_offset_from_true_goal_deg"] - 90) <= 20,
        ),
        (
            "5 途中で動かす",
            successes("5_displaced_after_44"),
            7 <= successes("5_displaced_after_44") <= 19,
        ),
        ("6 雑音 2 px (条件1との差)", noise[1] - normal, abs(noise[1] - normal) <= 4),
        ("6 雑音 5 px", noise[2], noise[2] <= normal),
        ("6 雑音 10 px", noise[3], noise[3] <= 10),
        ("6 雑音 20 px", noise[4], noise[4] <= 5),
        ("6 単調に減る (揺れ 3 本まで)", noise, all(b <= a + 3 for a, b in pairwise(noise))),
        _zero_noise_check(conditions),
    ]
    return [{"予想": name, "結果": value, "当たり": bool(hit)} for name, value, hit in checks]


def _zero_noise_check(conditions: dict) -> tuple[str, str, bool]:
    plain = conditions["1_normal"]["episodes"]
    zero = conditions["6_noise_0px"]["episodes"]
    same_paths = all(
        a["steps"] == b["steps"] and a["final_block_pose"] == b["final_block_pose"]
        for a, b in zip(plain, zero, strict=True)
    )
    exact = plain == zero
    detail = "全項目一致" if exact else "歩数と最後の姿勢は一致、最大の重なりが 1e-16 の桁でずれる"
    return ("6 雑音 0 px は条件1と全歩一致", detail, same_paths)


def _judge_grid(grid: dict, intervals: dict) -> dict:
    cells = [cell for cell in grid["cells"] if not cell["pilot_goal"]]
    in_support = [cell["starts_in_support"] for cell in cells]
    successes = [cell["successes"] for cell in cells]
    residual = np.array([cell["successes"] - cell["predicted_successes"] for cell in cells])
    edge = np.array([cell["edge"] for cell in cells])
    rho = spearman(in_support, successes)
    gap = float(residual[edge].mean() - residual[~edge].mean())
    h1_threshold = intervals["grid"]["rho_95th_if_no_relation"]
    h1_range = intervals["grid"]["rho_if_h1"]
    h2_threshold = intervals["grid"]["edge_minus_inner_residual_if_no_walls"][1]
    return {
        "cells_run": len(grid["cells"]),
        "cells_judged": len(cells),
        "H1": {
            "rho": rho,
            "threshold": h1_threshold,
            "当たり": rho > h1_threshold,
            "予想の範囲内": h1_range[0] <= rho <= h1_range[2],
        },
        "H2": {
            "edge_minus_inner_residual": gap,
            "edge_mean_residual": float(residual[edge].mean()),
            "inner_mean_residual": float(residual[~edge].mean()),
            "threshold": h2_threshold,
            "当たり": gap < h2_threshold,
        },
        "total_successes": int(sum(successes)),
        "total_predicted": float(sum(cell["predicted_successes"] for cell in cells)),
    }


if __name__ == "__main__":
    main()
