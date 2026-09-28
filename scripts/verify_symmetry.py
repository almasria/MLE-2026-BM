"""Verification of the D4 symmetry machinery.

Produces the two numbers the report cites for symmetry, and fails loudly if
either claim is untrue.

Check 1 (exhaustive, deterministic). For every tuple in the enumerated
feature space and every group element g, canonicalize(g . x) must return the
same key as canonicalize(x). Also reports the orbit count and the
distribution of orbit sizes, which is where the reduction factor comes from.

Check 2 (sampled). The feature map is equivariant only up to tie-breaking in
the breadth-first searches. This compares state_to_features(g . s) against
g . state_to_features(s) on random boards, counts the disagreements, and
records which components ever differ. A disagreement confined to the
directional components means the two views agree about danger, escape, bomb
value, mobility and engagement, and differ only in which of several equally
short paths they name.

Usage, from the repository root:

    python scripts/verify_symmetry.py
    python scripts/verify_symmetry.py --states 5000 --seed 1 \
        --json results/symmetry_verification.json

Exit status is non-zero if check 1 fails or if check 2 finds a disagreement
outside the directional components, so this is safe to run in CI.
"""

import argparse
import importlib
import inspect
import itertools
import json
import os
import sys
import types
from collections import Counter

import numpy as np

# Value ranges per component, in tuple order. Check these against the feature
# module before quoting the orbit count: a wrong range silently changes it.
# Index 7 omits 1 because the "survivable but worthless" level is disabled.
RANGES = (
    (0, 1, 2, 3, 4),        # 0 objective direction
    (0, 1), (0, 1), (0, 1), (0, 1),   # 1-4 neighbour enterable
    (0, 1, 2),              # 5 urgency
    (0, 1, 2, 3, 4),        # 6 escape direction
    (0, 2, 3, 4),           # 7 bomb opportunity
    (0, 1, 2),              # 8 mobility
    (0, 1, 2),              # 9 engagement
)

# Components whose value is a direction index. Used only to classify
# disagreements in check 2, not to compute anything.
DIRECTION_COMPONENTS = (0, 6)
NEIGHBOUR_COMPONENTS = (1, 2, 3, 4)

AGENT_DIR_CANDIDATES = (
    os.path.join("agent_code", "RL-Team"),
    os.path.join("agent_code", "q_agent"),
)


def load_agent_modules(agent_dir):
    """Import the agent's feature and symmetry modules.

    The agent folder name contains a hyphen and the modules use relative
    imports, so neither a plain import nor a sys.path entry works. Register a
    synthetic package whose __path__ points at the folder instead.
    """
    package = "_agent_under_test"
    pkg = types.ModuleType(package)
    pkg.__path__ = [os.path.abspath(agent_dir)]
    pkg.__package__ = package
    sys.modules[package] = pkg

    try:
        features = importlib.import_module(package + ".features")
        symmetry = importlib.import_module(package + ".symmetry")
        return features, symmetry
    except Exception:
        # Older layouts used absolute imports and worked off sys.path.
        del sys.modules[package]
        sys.path.insert(0, os.path.abspath(agent_dir))
        features = importlib.import_module("features")
        symmetry = importlib.import_module("symmetry")
        return features, symmetry


def discover_layout(features, symmetry):
    """The (direction indices, neighbour slice) description of the tuple.

    transform() and canonicalize() in symmetry.py are generic and take this as
    an argument; features.py binds it for the active feature version.
    """
    for attr in ("_layout", "LAYOUT", "layout"):
        if hasattr(features, attr):
            return getattr(features, attr)
    layouts = getattr(symmetry, "LAYOUTS", None)
    version = getattr(features, "FEATURE_VERSION", None) \
        or os.environ.get("Q_AGENT_FEATURE_VERSION")
    if layouts and version in layouts:
        return layouts[version]
    return None


def required_positional(fn):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    return sum(1 for p in params
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
               and p.default is p.empty)


def bind_layout(fn, layout, base_args, name):
    """Return fn with the layout appended, if fn asks for it.

    The generic functions in symmetry.py take the layout as a trailing
    argument; the wrappers re-exported from features.py do not.
    """
    n = required_positional(fn)
    if n is None or n <= base_args:
        return fn
    if layout is None:
        raise SystemExit(
            f"{name}() expects {n} arguments but the tuple layout could not be "
            "found; looked for features._layout and symmetry.LAYOUTS[version]"
        )
    return lambda *args: fn(*args, layout)


def resolve_api(features, symmetry):
    """Pick out the callables this script needs, whatever they are named."""

    def first(*candidates):
        for module, name in candidates:
            if hasattr(module, name):
                return getattr(module, name)
        return None

    layout = discover_layout(features, symmetry)

    raw_canonicalize = first((features, "canonicalize"), (symmetry, "canonicalize"))
    raw_transform = first((features, "transform"), (symmetry, "transform"),
                          (symmetry, "transform_features"))

    api = {
        "state_to_features": first((features, "state_to_features")),
        "group": first((symmetry, "GROUP"), (symmetry, "D4_GROUP"),
                       (symmetry, "ELEMENTS")),
        "transform_game_state": first((symmetry, "transform_game_state")),
        "layout": layout,
    }
    missing = [k for k, v in api.items() if v is None and k != "layout"]
    if raw_canonicalize is None:
        missing.append("canonicalize")
    if raw_transform is None:
        missing.append("transform")
    if missing:
        raise SystemExit(
            "could not resolve: " + ", ".join(missing)
            + "\nlooked in features.py and symmetry.py; adjust resolve_api() "
              "if these are named differently"
        )

    # transform(tuple, element[, layout]); canonicalize(tuple[, layout])
    api["transform"] = bind_layout(raw_transform, layout, 2, "transform")
    api["canonicalize"] = bind_layout(raw_canonicalize, layout, 1, "canonicalize")
    return api


def canonical_key(canonicalize, tup):
    """canonicalize() returns (key, element); tolerate a bare key as well."""
    out = canonicalize(tup)
    return out[0] if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], tuple) else out


# --- check 1: canonicalisation over the whole feature space ------------------

def check_canonicalisation(api, verbose=True):
    canonicalize, transform, group = api["canonicalize"], api["transform"], api["group"]

    mismatches = []
    orbit_sizes = Counter()
    keys = set()
    total = 0

    for tup in itertools.product(*RANGES):
        total += 1
        key = canonical_key(canonicalize, tup)
        views = set()
        for g in group:
            moved = tuple(transform(tup, g))
            views.add(moved)
            if canonical_key(canonicalize, moved) != key:
                if len(mismatches) < 10:
                    mismatches.append({"tuple": list(tup), "element": str(g)})
        if key not in keys:
            keys.add(key)
            orbit_sizes[len(views)] += 1

    result = {
        "tuples": total,
        "group_size": len(group),
        "mismatches": len(mismatches),
        "examples": mismatches,
        "orbits": len(keys),
        "orbit_size_distribution": {str(k): v for k, v in sorted(orbit_sizes.items())},
        "reduction_factor": round(total / len(keys), 3) if keys else None,
    }

    if verbose:
        print("check 1: canonicalisation over the enumerated feature space")
        print(f"  tuples enumerated      {total}")
        print(f"  distinct keys (orbits) {len(keys)}")
        print(f"  reduction factor       {result['reduction_factor']}")
        print("  orbit sizes            "
              + ", ".join(f"{k}:{v}" for k, v in sorted(orbit_sizes.items())))
        if mismatches:
            print(f"  FAIL {len(mismatches)} tuples canonicalise inconsistently")
            for m in mismatches:
                print("    ", m)
        else:
            print("  OK all group images share one key")
    return result


# --- check 2: equivariance of the feature map on random boards ---------------

def random_state(rng, cols=17, rows=17, crate_density=0.75, n_coins=9,
                 n_opponents=3, bomb_chance=0.4):
    """A board with the real wall lattice and randomly placed contents.

    Not a faithful replica of the generator in the framework; it only has to
    produce states the feature code accepts, with enough variety that ties
    between equidistant targets actually occur.
    """
    field = np.zeros((cols, rows), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(cols):
        for y in range(rows):
            if x % 2 == 0 and y % 2 == 0:
                field[x, y] = -1

    free = [(x, y) for x in range(cols) for y in range(rows) if field[x, y] == 0]
    for tile in free:
        if rng.random() < crate_density:
            field[tile] = 1

    open_tiles = [t for t in free if field[t] == 0]
    rng.shuffle(open_tiles)
    if len(open_tiles) < 1 + n_opponents + n_coins + 2:
        return None

    cursor = 0

    def take(k):
        nonlocal cursor
        got = open_tiles[cursor:cursor + k]
        cursor += k
        return got

    own = take(1)[0]
    opponents = take(n_opponents)
    coins = take(n_coins)

    bombs = []
    explosion_map = np.zeros((cols, rows), dtype=int)
    if rng.random() < bomb_chance:
        for tile in take(int(rng.integers(1, 3))):
            bombs.append((tile, int(rng.integers(0, 4))))
    if rng.random() < 0.25:
        for tile in take(1):
            explosion_map[tile] = int(rng.integers(1, 3))

    return {
        "round": 1,
        "step": int(rng.integers(1, 200)),
        "field": field,
        "bombs": bombs,
        "explosion_map": explosion_map,
        "coins": coins,
        "self": ("me", int(rng.integers(0, 20)), bool(rng.random() < 0.7), own),
        "others": [(f"opp{i}", int(rng.integers(0, 20)),
                    bool(rng.random() < 0.5), tile)
                   for i, tile in enumerate(opponents)],
        "user_input": None,
    }


def check_equivariance(api, n_states, seed, verbose=True):
    to_features = api["state_to_features"]
    transform, group = api["transform"], api["group"]
    transform_state = api["transform_game_state"]

    rng = np.random.default_rng(seed)
    comparisons = 0
    disagreements = 0
    differing_components = Counter()
    directional_only = 0
    examples = []
    out_of_range = Counter()
    sampled = 0

    while sampled < n_states:
        state = random_state(rng)
        if state is None:
            continue
        base = to_features(state)
        if base is None:
            continue
        sampled += 1

        for i, value in enumerate(base):
            if i < len(RANGES) and value not in RANGES[i]:
                out_of_range[i] += 1

        for g in group:
            moved = to_features(transform_state(state, g))
            if moved is None:
                continue
            expected = tuple(transform(base, g))
            comparisons += 1
            if tuple(moved) == expected:
                continue
            disagreements += 1
            diff = [i for i, (a, b) in enumerate(zip(moved, expected)) if a != b]
            for i in diff:
                differing_components[i] += 1
            if set(diff) <= set(DIRECTION_COMPONENTS):
                directional_only += 1
            elif len(examples) < 10:
                examples.append({
                    "element": str(g),
                    "components": diff,
                    "observed": list(moved),
                    "expected": list(expected),
                })

    result = {
        "states_sampled": sampled,
        "comparisons": comparisons,
        "disagreements": disagreements,
        "disagreement_rate": round(disagreements / comparisons, 5) if comparisons else None,
        "directional_only": directional_only,
        "structural": disagreements - directional_only,
        "differing_components": {str(k): v for k, v in sorted(differing_components.items())},
        "examples": examples,
        "values_outside_declared_ranges": {str(k): v for k, v in out_of_range.items()},
    }

    if verbose:
        print()
        print("check 2: equivariance of the feature map on random boards")
        print(f"  states sampled         {sampled}")
        print(f"  comparisons            {comparisons}")
        print(f"  disagreements          {disagreements} "
              f"({result['disagreement_rate']})")
        print(f"    directional only     {directional_only}")
        print(f"    other components     {result['structural']}")
        if differing_components:
            print("  components that ever differ  "
                  + ", ".join(f"{k}:{v}" for k, v in sorted(differing_components.items())))
        if out_of_range:
            print("  WARNING values outside RANGES: " + str(dict(out_of_range))
                  + "  -- the orbit count in check 1 is wrong until RANGES is fixed")
        if result["structural"]:
            print("  FAIL disagreements outside the directional components")
            for e in examples:
                print("    ", e)
        else:
            print("  OK every disagreement is confined to the directional components")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent-dir", default=None,
                    help="path to the agent folder (default: autodetect)")
    ap.add_argument("--states", type=int, default=2000,
                    help="random states for check 2 (default 2000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-exhaustive", action="store_true")
    ap.add_argument("--json", default=None, help="write results to this file")
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
    print(f"agent folder: {agent_dir}")
    version = getattr(features, "FEATURE_VERSION", None) or \
        os.environ.get("Q_AGENT_FEATURE_VERSION", "default")
    print(f"feature version: {version}")
    print(f"group size: {len(api['group'])}")
    print(f"tuple layout: {api['layout']}")

    declared = getattr(features, "FEATURE_LENGTH", None)
    if declared is not None and declared != len(RANGES):
        raise SystemExit(
            f"the module reports FEATURE_LENGTH={declared} but RANGES declares "
            f"{len(RANGES)} components; fix RANGES before quoting any count"
        )
    print()

    results = {"agent_dir": agent_dir, "feature_version": str(version),
               "seed": args.seed}
    failed = False

    if not args.skip_exhaustive:
        results["canonicalisation"] = check_canonicalisation(api)
        failed |= results["canonicalisation"]["mismatches"] > 0

    results["equivariance"] = check_equivariance(api, args.states, args.seed)
    failed |= results["equivariance"]["structural"] > 0

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwritten to {args.json}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
