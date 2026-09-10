"""D4 symmetry for the Bomberman board.

The board, the wall lattice and the four starting corners are invariant under
the dihedral group D4 (rotations by multiples of 90 degrees and reflections).
Hence Q*(g s, g a) = Q*(s, a) for every group element g, and all eight views
of a situation can share one Q-table row.

Layer 1: the group. Each element is "mirror left-right (m in {0,1}), then
rotate 90 degrees clockwise k times (k in 0..3)", acting on direction
indices, action indices, board coordinates and board arrays.

Layer 2: canonicalisation of a feature tuple. A tuple is described by which
components are direction indices (they permute under g) and which four
components are the per-direction neighbour flags (they are re-ordered); all
other components are invariant scalars. The orbit representative is the
lexicographically smallest transformed tuple.

Conventions (screen coordinates: x right, y down, origin top-left):
    rotation cw on vectors: (dx, dy) -> (-dy, dx)   so UP -> RIGHT
    rotation cw on points:  (x, y)   -> (S-1-y, x)
    mirror LR on vectors:   (dx, dy) -> (-dx, dy)   so RIGHT -> LEFT
    mirror LR on points:    (x, y)   -> (S-1-x, y)
"""

import numpy as np

N_DIRS = 4
NO_DIR = 4
MIRROR_DIR = (0, 3, 2, 1)       # UP fixed, RIGHT <-> LEFT, DOWN fixed


class D4Element:
    def __init__(self, m, k):
        self.m, self.k = m, k

    def apply_dir(self, d):
        if d == NO_DIR:
            return d
        if self.m:
            d = MIRROR_DIR[d]
        return (d + self.k) % N_DIRS

    def apply_action(self, a):
        """Directions permute; WAIT and BOMB (indices >= 4) are fixed."""
        return self.apply_dir(a) if a < N_DIRS else a

    def apply_coord(self, x, y, size):
        if self.m:
            x = size - 1 - x
        for _ in range(self.k):
            x, y = size - 1 - y, x
        return x, y

    def apply_array(self, arr):
        """Board array indexed [x, y]; satisfies new[g(x, y)] == old[x, y]."""
        assert arr.shape[0] == arr.shape[1], "square boards only"
        if self.m:
            arr = arr[::-1, :]
        for _ in range(self.k):
            arr = arr.T[::-1, :]
        return arr

    def inverse(self):
        if self.m:
            return self                     # reflections are involutions
        return D4Element(0, (N_DIRS - self.k) % N_DIRS)

    def __repr__(self):
        return f"D4(m={self.m}, k={self.k})"


GROUP = [D4Element(m, k) for m in (0, 1) for k in range(4)]
IDENTITY = GROUP[0]


# --- feature tuple layouts: (direction component indices, neighbour slice) --
LAYOUTS = {
    "v1": ((0,), slice(1, 5)),
    "v2": ((0, 6), slice(1, 5)),
    "v3": ((0, 6, 9), slice(1, 5)),
    "v4": ((0, 6), slice(1, 5)),
    "v5": ((0, 6), slice(1, 5)),        # engagement (index 9) is a scalar
}


def transform(features, g, layout):
    dir_indices, nb = layout
    out = list(features)
    for i in dir_indices:
        out[i] = g.apply_dir(features[i])
    flags = features[nb]
    new_flags = [0] * N_DIRS
    for d in range(N_DIRS):
        new_flags[g.apply_dir(d)] = flags[d]
    out[nb] = new_flags
    return tuple(out)


def canonicalize(features, layout):
    """Return (orbit representative, g) with transform(features, g) == representative.

    Use: act() reads Q[representative] and executes g.inverse().apply_action(a);
    the update indexes Q[representative] at g.apply_action(action_taken).
    """
    best, best_g = features, IDENTITY
    for g in GROUP:
        t = transform(features, g, layout)
        if t < best:
            best, best_g = t, g
    return best, best_g


def _versioned(name):
    layout = LAYOUTS[name]
    return (lambda f, g: transform(f, g, layout)), (lambda f: canonicalize(f, layout))


transform_v1, canonicalize_v1 = _versioned("v1")
transform_v2, canonicalize_v2 = _versioned("v2")
transform_v3, canonicalize_v3 = _versioned("v3")
transform_v4, canonicalize_v4 = _versioned("v4")
transform_v5, canonicalize_v5 = _versioned("v5")


def transform_game_state(gs, g):
    """Whole-world transform, e.g. for data augmentation of a replay buffer."""
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


if __name__ == "__main__":
    # self-checks: group axioms, bijectivity and invariance on the v5 layout
    from itertools import product
    import random

    for d in range(5):
        x = d
        for _ in range(4):
            x = D4Element(0, 1).apply_dir(x)
        assert x == d
        assert D4Element(1, 0).apply_dir(D4Element(1, 0).apply_dir(d)) == d
    for g in GROUP:
        for d in range(5):
            assert g.inverse().apply_dir(g.apply_dir(d)) == d

    layout = LAYOUTS["v5"]
    space = [(o, *nb, u, s, b, m, e) for o in range(5)
             for nb in product((0, 1), repeat=4) for u in range(3)
             for s in range(5) for b in range(5) for m in range(3) for e in range(3)]
    for g in GROUP:
        assert len({transform(f, g, layout) for f in space}) == len(space)
    orbits = set()
    for f in space:
        canon, g = canonicalize(f, layout)
        assert transform(f, g, layout) == canon
        orbits.add(canon)
    random.seed(0)
    for f in random.sample(space, 500):
        canon = canonicalize(f, layout)[0]
        for h in GROUP:
            assert canonicalize(transform(f, h, layout), layout)[0] == canon

    rng = np.random.default_rng(0)
    for g in GROUP:
        a = rng.integers(-1, 2, size=(7, 7))
        b = g.apply_array(a)
        for x in range(7):
            for y in range(7):
                assert b[g.apply_coord(x, y, 7)] == a[x, y]
    print(f"v5: {len(space)} tuples -> {len(orbits)} orbits; all checks passed")
