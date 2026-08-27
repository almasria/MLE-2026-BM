"""
symmetry.py — The dihedral group D4 for Bomberman, generalized.

Two layers:

1. THE GROUP (general, never changes): 8 elements, each parameterized as
   "mirror left-right first (m in {0,1}), then rotate 90° clockwise k times
   (k in {0,1,2,3})". Every element knows how to act on the four primitive
   objects of this game:
       - direction indices   (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT; 4 = "none")
       - action indices      (directions permute, WAIT/BOMB are fixed)
       - board coordinates   (x, y) on a square board
       - board arrays        field / explosion_map, indexed [x, y]
   Convention checks (screen coordinates: x right, y down, origin top-left):
       rotation cw on vectors: (dx, dy) -> (-dy, dx), so UP -> RIGHT   ✓
       rotation cw on points:  (x, y) -> (S-1-y, x)                    ✓
       mirror LR on vectors:   (dx, dy) -> (-dx, dy), so RIGHT -> LEFT ✓
       mirror LR on points:    (x, y) -> (S-1-x, y)                    ✓

2. CANONICALIZATION (feature-version-specific): map a feature tuple to a
   canonical representative of its D4 orbit. For features v1 this is
   `canonicalize_v1`. When features grow (v2: danger/escape directions...),
   write the new transform out of the same primitives and add
   `canonicalize_v2` — the group layer is untouched.

Group facts used:
   - inverse of a pure rotation (0, k) is (0, (4-k) % 4)
   - every mirrored element (1, k) is a reflection, hence ITS OWN inverse
"""

import numpy as np

N_DIRS = 4
NO_DIR = 4                      # "no target reachable / standing on it"
MIRROR_DIR = (0, 3, 2, 1)       # UP fixed, RIGHT<->LEFT, DOWN fixed


class D4Element:
    """One of the 8 symmetries: mirror (optional), then k clockwise rotations."""

    def __init__(self, m: int, k: int):
        self.m, self.k = m, k

    # --- primitive actions -------------------------------------------------

    def apply_dir(self, d: int) -> int:
        """Transform a direction index (0..3); NO_DIR (4) is invariant."""
        if d == NO_DIR:
            return d
        if self.m:
            d = MIRROR_DIR[d]
        return (d + self.k) % N_DIRS

    def apply_action(self, a: int) -> int:
        """Transform an action index: directions permute, WAIT/BOMB fixed."""
        return self.apply_dir(a) if a < N_DIRS else a

    def apply_coord(self, x: int, y: int, size: int):
        """Transform a board coordinate on a size x size board."""
        if self.m:
            x = size - 1 - x
        for _ in range(self.k):
            x, y = size - 1 - y, x
        return x, y

    def apply_array(self, arr: np.ndarray) -> np.ndarray:
        """Transform a board array indexed [x, y] (framework convention).
        Satisfies: new[apply_coord(x, y)] == old[x, y]."""
        assert arr.shape[0] == arr.shape[1], "square boards only"
        if self.m:
            arr = arr[::-1, :]
        for _ in range(self.k):
            arr = arr.T[::-1, :]
        return arr

    # --- group structure ---------------------------------------------------

    def inverse(self) -> "D4Element":
        if self.m:
            return self                     # reflections are involutions
        return D4Element(0, (N_DIRS - self.k) % N_DIRS)

    def __repr__(self):
        return f"D4(m={self.m}, k={self.k})"


GROUP = [D4Element(m, k) for m in (0, 1) for k in range(4)]
IDENTITY = GROUP[0]


# ---------------------------------------------------------------------------
# Layer 2: feature canonicalization (version-specific)
# ---------------------------------------------------------------------------

def transform_v1(features: tuple, g: D4Element) -> tuple:
    """Apply g to a v1 feature tuple (coin_dir, up, right, down, left).
    The neighbor flag that WAS in direction d is NOW in direction g(d)."""
    coin_dir, *nb = features
    new_nb = [0] * N_DIRS
    for d in range(N_DIRS):
        new_nb[g.apply_dir(d)] = nb[d]
    return (g.apply_dir(coin_dir), *new_nb)


def canonicalize_v1(features: tuple):
    """Map features to their orbit's canonical representative.

    Returns (canonical_features, g) where g is the group element with
    g(features) == canonical_features. To USE it:
      - act():   a_hat = argmax Q[canonical];  execute g.inverse().apply_action(a_hat)
      - update:  index Q[canonical] at action g.apply_action(a_taken)
    """
    best, best_g = features, IDENTITY
    for g in GROUP:
        t = transform_v1(features, g)
        if t < best:
            best, best_g = t, g
    return best, best_g


# ---------------------------------------------------------------------------
# Full game_state transform — for DQN data augmentation (and future features)
# ---------------------------------------------------------------------------

def transform_game_state(gs: dict, g: D4Element) -> dict:
    """Return a copy of game_state with the whole world transformed by g.
    Push all 8 transforms of each transition into the DQN replay buffer to
    get 8x the training data from every real experience."""
    size = gs['field'].shape[0]
    tc = lambda pos: g.apply_coord(pos[0], pos[1], size)
    tagent = lambda a: (a[0], a[1], a[2], tc(a[3]))
    return {
        **gs,
        'field': g.apply_array(gs['field']),
        'explosion_map': g.apply_array(gs['explosion_map']),
        'coins': [tc(c) for c in gs['coins']],
        'bombs': [(tc(pos), t) for pos, t in gs['bombs']],
        'self': tagent(gs['self']),
        'others': [tagent(o) for o in gs['others']],
    }


# ---------------------------------------------------------------------------
# Self-tests: run `python symmetry.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from itertools import product

    all_tuples = [(d, *nb) for d in range(5) for nb in product((0, 1), repeat=4)]
    assert len(all_tuples) == 80

    # 1. Group axioms on directions: R^4 = id, M^2 = id, inverses invert
    R, M = D4Element(0, 1), D4Element(1, 0)
    for d in range(5):
        x = d
        for _ in range(4):
            x = R.apply_dir(x)
        assert x == d
        assert M.apply_dir(M.apply_dir(d)) == d
    for g in GROUP:
        gi = g.inverse()
        for d in range(5):
            assert gi.apply_dir(g.apply_dir(d)) == d

    # 2. Each element is a bijection on the 80 feature tuples
    for g in GROUP:
        assert len({transform_v1(f, g) for f in all_tuples}) == 80

    # 3. THE invariance test: all 8 views share one canonical representative,
    #    and the returned g really maps input to canonical
    for f in all_tuples:
        canon, g = canonicalize_v1(f)
        assert transform_v1(f, g) == canon
        for h in GROUP:
            assert canonicalize_v1(transform_v1(f, h))[0] == canon

    # 4. Orbit count (compare with Burnside's lemma by hand for the report)
    orbits = {canonicalize_v1(f)[0] for f in all_tuples}
    print(f"80 feature tuples collapse into {len(orbits)} orbits")

    # 5. Coordinates <-> arrays consistency: new[g(x,y)] == old[x,y]
    rng = np.random.default_rng(0)
    for g in GROUP:
        a = rng.integers(-1, 2, size=(7, 7))
        b = g.apply_array(a)
        for x in range(7):
            for y in range(7):
                assert b[g.apply_coord(x, y, 7)] == a[x, y]

    # 6. Directions <-> coordinates consistency: stepping in direction d from
    #    p, then transforming, equals transforming p and stepping in g(d)
    OFFS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    for g in GROUP:
        for d in range(4):
            x, y = 3, 2
            nx, ny = x + OFFS[d][0], y + OFFS[d][1]
            gd = g.apply_dir(d)
            gx, gy = g.apply_coord(x, y, 7)
            assert g.apply_coord(nx, ny, 7) == (gx + OFFS[gd][0], gy + OFFS[gd][1])

    print("All symmetry self-tests passed.")


# ---------------------------------------------------------------------------
# v2 features (week 3): (objective_dir, nb x4, in_danger, safe_dir,
#                        bomb_safe, crates_in_range)
# Directions permute under g; flags and counts are invariant.
# ---------------------------------------------------------------------------

def transform_v2(features: tuple, g: D4Element) -> tuple:
    obj, u, r, d, l, danger, safe, bomb_safe, crates = features
    nb = (u, r, d, l)
    new_nb = [0] * N_DIRS
    for i in range(N_DIRS):
        new_nb[g.apply_dir(i)] = nb[i]
    return (g.apply_dir(obj), *new_nb, danger, g.apply_dir(safe), bomb_safe, crates)


def canonicalize_v2(features: tuple):
    """Orbit representative + the g that maps input to it (see canonicalize_v1)."""
    best, best_g = features, IDENTITY
    for g in GROUP:
        t = transform_v2(features, g)
        if t < best:
            best, best_g = t, g
    return best, best_g


# ---------------------------------------------------------------------------
# v3 features (week 4): v2 + (opp_dir, opp_in_blast).
# opp_dir permutes like every direction; opp_in_blast is invariant.
# ---------------------------------------------------------------------------

def transform_v3(features: tuple, g: D4Element) -> tuple:
    """Apply D4 to the legacy v2/v3 feature prefix and leave the newer scalar
    tactical features untouched. This keeps the Q-agent compatible when the
    feature vector grows without changing the action-index contracts.
    """
    if len(features) <= 11:
        obj, u, r, d, l, urg, safe, bsafe, crates, opp, oblast = features
        nb = (u, r, d, l)
        new_nb = [0] * N_DIRS
        for i in range(N_DIRS):
            new_nb[g.apply_dir(i)] = nb[i]
        return (g.apply_dir(obj), *new_nb, urg, g.apply_dir(safe), bsafe, crates,
                g.apply_dir(opp), oblast)

    obj, u, r, d, l, urg, safe, bsafe, crates, opp, oblast = features[:11]
    nb = (u, r, d, l)
    new_nb = [0] * N_DIRS
    for i in range(N_DIRS):
        new_nb[g.apply_dir(i)] = nb[i]
    transformed_prefix = (g.apply_dir(obj), *new_nb, urg, g.apply_dir(safe), bsafe,
                          crates, g.apply_dir(opp), oblast)
    return transformed_prefix + tuple(features[11:])


def canonicalize_v3(features: tuple):
    best, best_g = features, IDENTITY
    for g in GROUP:
        t = transform_v3(features, g)
        if t < best:
            best, best_g = t, g
    return best, best_g
