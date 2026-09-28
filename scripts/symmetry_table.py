"""Counts, for each element of D4, how many feature tuples it leaves unchanged.

This is the Burnside computation behind the orbit count: every symmetry class
contributes exactly |G| fixed pairs, so the number of classes is the average
of the per-element counts. Running this is what keeps the table in the report
in step with the code.

Group elements are named by how they act, not by their position in GROUP, so
the rows come out in a fixed order regardless of how the module enumerates
them.

Usage, from the repository root:

    python scripts/symmetry_table.py
    python scripts/symmetry_table.py --latex --json results/symmetry_table.json

Requires verify_symmetry.py in the same directory; RANGES is taken from there
so the two scripts cannot disagree about the size of the feature space.
"""

import argparse
import itertools
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from verify_symmetry import (AGENT_DIR_CANDIDATES, RANGES, canonical_key,
                                 load_agent_modules, resolve_api)
except ImportError:
    raise SystemExit("verify_symmetry.py must sit next to this script")

# Unit vectors for direction indices 0..3, in the order the feature code uses.
DIRECTION_VECTORS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # up right down left

# Row order for the printed table, keyed by the label classify() returns.
ROW_ORDER = [
    "identity",
    "90 degree rotation",
    "180 degree rotation",
    "270 degree rotation",
    "reflection across the vertical axis",
    "reflection across the horizontal axis",
    "reflection across the main diagonal",
    "reflection across the anti-diagonal",
]

LATEX_LABELS = {
    "identity": "Identity",
    "90 degree rotation": r"90\textdegree{} rotation",
    "180 degree rotation": r"180\textdegree{} rotation",
    "270 degree rotation": r"270\textdegree{} rotation",
    "reflection across the vertical axis": "Reflection across the vertical axis",
    "reflection across the horizontal axis": "Reflection across the horizontal axis",
    "reflection across the main diagonal": "Reflection across the main diagonal",
    "reflection across the anti-diagonal": "Reflection across the anti-diagonal",
}


def direction_permutation(transform, g, n_components):
    """Recover how g permutes direction indices, using only transform().

    Probes with a tuple whose first component is a direction and whose other
    components are zero, so the answer does not depend on the layout beyond
    component 0 being direction-valued.
    """
    perm = []
    for i in range(4):
        probe = [0] * n_components
        probe[0] = i
        perm.append(tuple(transform(tuple(probe), g))[0])
    return tuple(perm)


def classify(perm):
    """Name a group element from its action on the four directions."""
    up, right, down, left = 0, 1, 2, 3

    if perm == (up, right, down, left):
        return "identity"
    # A rotation sends up to the next direction clockwise, and so on.
    if perm == (right, down, left, up):
        return "90 degree rotation"
    if perm == (down, left, up, right):
        return "180 degree rotation"
    if perm == (left, up, right, down):
        return "270 degree rotation"
    # Reflections: name the axis that stays put.
    if perm == (up, left, down, right):
        return "reflection across the vertical axis"
    if perm == (down, right, up, left):
        return "reflection across the horizontal axis"
    if perm == (left, down, right, up):
        return "reflection across the main diagonal"
    if perm == (right, up, left, down):
        return "reflection across the anti-diagonal"
    return "unrecognised " + str(perm)


def count_fixed(api, n_components):
    """Fixed-tuple count per element, plus orbit count and size distribution."""
    transform, group, canonicalize = api["transform"], api["group"], api["canonicalize"]

    labels = {}
    for g in group:
        perm = direction_permutation(transform, g, n_components)
        label = classify(perm)
        if label in labels:
            raise SystemExit(
                f"two group elements classify as '{label}'; the probe in "
                "direction_permutation() is not distinguishing them"
            )
        labels[label] = g

    fixed = Counter()
    orbit_sizes = Counter()
    keys = set()
    total = 0

    for tup in itertools.product(*RANGES):
        total += 1
        views = set()
        for label, g in labels.items():
            moved = tuple(transform(tup, g))
            views.add(moved)
            if moved == tup:
                fixed[label] += 1
        key = canonical_key(canonicalize, tup)
        if key not in keys:
            keys.add(key)
            orbit_sizes[len(views)] += 1

    return {
        "tuples": total,
        "group_size": len(group),
        "fixed_per_element": dict(fixed),
        "total_fixed": sum(fixed.values()),
        "orbits_by_burnside": sum(fixed.values()) // len(group),
        "orbits_by_enumeration": len(keys),
        "orbit_size_distribution": {str(k): v for k, v in sorted(orbit_sizes.items())},
        "reduction_factor": round(total / len(keys), 3) if keys else None,
    }


def print_table(result):
    fixed = result["fixed_per_element"]
    width = max(len(r) for r in ROW_ORDER) + 2
    print(f"{'transformation':<{width}}{'tuples left unchanged':>22}")
    print("-" * (width + 22))
    for label in ROW_ORDER:
        if label in fixed:
            print(f"{label:<{width}}{fixed[label]:>22,}")
    for label in sorted(set(fixed) - set(ROW_ORDER)):
        print(f"{label:<{width}}{fixed[label]:>22,}")
    print("-" * (width + 22))
    print(f"{'total':<{width}}{result['total_fixed']:>22,}")
    print()
    print(f"tuples enumerated        {result['tuples']:,}")
    print(f"orbits (Burnside)        {result['orbits_by_burnside']:,}")
    print(f"orbits (enumeration)     {result['orbits_by_enumeration']:,}")
    print(f"reduction factor         {result['reduction_factor']}")
    print("orbit sizes              "
          + ", ".join(f"{k}:{v}" for k, v in
                      sorted(result['orbit_size_distribution'].items(), key=lambda kv: int(kv[0]))))
    if result["orbits_by_burnside"] != result["orbits_by_enumeration"]:
        print("\nFAIL Burnside and direct enumeration disagree; "
              "canonicalize() and transform() are inconsistent")


def print_latex(result):
    fixed = result["fixed_per_element"]

    def num(v):
        return "{:,}".format(v).replace(",", "{,}")

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"\textbf{Transformation} & \textbf{Tuples left unchanged} \\")
    print(r"\midrule")
    for label in ROW_ORDER:
        if label in fixed:
            print(f"{LATEX_LABELS[label]} & ${num(fixed[label])}$ \\\\")
    print(r"\midrule")
    print(f"Total & ${num(result['total_fixed'])}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Number of feature tuples left unchanged by each element of "
          r"the dihedral group $D_4$. Dividing the total by $8$ gives "
          f"${num(result['orbits_by_burnside'])}$ symmetry classes from "
          f"${num(result['tuples'])}$ tuples, a reduction by a factor of "
          f"{result['reduction_factor']}" r".}")
    print(r"\label{tab:symmetry_counts}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent-dir", default=None)
    ap.add_argument("--latex", action="store_true", help="also emit the LaTeX table")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    agent_dir = args.agent_dir
    if agent_dir is None:
        for candidate in AGENT_DIR_CANDIDATES:
            if os.path.isdir(candidate):
                agent_dir = candidate
                break
    if agent_dir is None or not os.path.isdir(agent_dir):
        raise SystemExit("agent folder not found; pass --agent-dir")

    features, symmetry = load_agent_modules(agent_dir)
    api = resolve_api(features, symmetry)
    print(f"agent folder: {agent_dir}\n")

    result = count_fixed(api, len(RANGES))
    print_table(result)
    if args.latex:
        print()
        print_latex(result)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwritten to {args.json}")

    return 0 if result["orbits_by_burnside"] == result["orbits_by_enumeration"] else 1


if __name__ == "__main__":
    sys.exit(main())
