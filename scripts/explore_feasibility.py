"""Post-hoc, not pre-registered: separate grid goals the T cannot physically fit into.

predictions.md did not consider that a goal near an edge may put part of the T outside the walls.
Such a goal cannot reach 95% coverage whatever the policy does. This script reports the
pre-registered H1/H2 quantities again with those cells set aside.

Usage: uv run python scripts/explore_feasibility.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compass.geometry import fits_inside_walls
from compass.metrics import spearman

GRID = Path("../results/grid.json")
OUTPUT = Path("../results/grid_feasibility.json")


def main() -> None:
    cells = json.loads(GRID.read_text(encoding="utf-8"))["cells"]
    judged = [cell for cell in cells if not cell["pilot_goal"]]
    for_table = [{**cell, "fits": fits_inside_walls(tuple(cell["goal"]))} for cell in judged]
    infeasible = [cell for cell in for_table if not cell["fits"]]
    feasible = [cell for cell in for_table if cell["fits"]]

    residual = np.array([cell["successes"] - cell["predicted_successes"] for cell in feasible])
    edge = np.array([cell["edge"] for cell in feasible])
    report = {
        "note": "post-hoc analysis, not in predictions.md",
        "infeasible_cells": len(infeasible),
        "infeasible_successes": sum(cell["successes"] for cell in infeasible),
        "infeasible_goals": [cell["goal"] for cell in infeasible],
        "feasible_cells": len(feasible),
        "feasible_edge_cells": int(edge.sum()),
        "feasible_inner_cells": int((~edge).sum()),
        "rho_feasible": spearman(
            [cell["starts_in_support"] for cell in feasible],
            [cell["successes"] for cell in feasible],
        ),
        "edge_minus_inner_residual_feasible": float(residual[edge].mean() - residual[~edge].mean()),
        "feasible_edge_mean_residual": float(residual[edge].mean()),
        "feasible_inner_mean_residual": float(residual[~edge].mean()),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
