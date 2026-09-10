"""Deterministic tests for the three design goals of features v4.

    1. small state space      -> tuple arity and per-component cardinality
    2. opportunistic bombing  -> crate value is visible while heading to a coin
    3. trap avoidance         -> mobility collapses when anything boxes us in

Run from the repository root:  python -m unittest discover -s tests -t .
"""

import unittest

import numpy as np

import importlib

f = importlib.import_module("agent_code.RL-Team.featuresv4")
_sym = importlib.import_module("agent_code.RL-Team.symmetry")
GROUP, canonicalize_v4, transform_v4 = _sym.GROUP, _sym.canonicalize_v4, _sym.transform_v4


def open_field(size=9):
    """Empty board with a stone border."""
    field = np.zeros((size, size), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    return field


def game_state(field, position, *, coins=(), bombs=(), others=(),
               explosions=None, bomb_available=True):
    if explosions is None:
        explosions = np.zeros_like(field)
    return {
        'round': 1, 'step': 1, 'field': field, 'bombs': list(bombs),
        'explosion_map': explosions, 'coins': list(coins),
        'self': ('RL-Team', 0, bomb_available, position),
        'others': [(f'opponent_{i}', 0, True, p) for i, p in enumerate(others)],
        'user_input': None,
    }


class StateSpaceTests(unittest.TestCase):
    def test_tuple_is_nine_small_integers(self):
        state = game_state(open_field(), (4, 4), coins=[(6, 4)])
        features = f.state_to_features(state)
        self.assertEqual(len(features), f.FEATURE_LENGTH)
        self.assertTrue(all(isinstance(v, int) for v in features),
                        "float components would explode a tabular state space")
        self.assertTrue(all(0 <= v <= 4 for v in features))

    def test_symmetry_reduces_the_enumerated_space(self):
        from itertools import product
        space = [(o, *nb, u, s, b, m)
                 for o in range(5) for nb in product((0, 1), repeat=4)
                 for u in range(3) for s in range(5)
                 for b in range(4) for m in range(3)]
        orbits = {canonicalize_v4(x)[0] for x in space}
        self.assertEqual(len(space), 14400)
        self.assertLess(len(orbits), 3000)

    def test_all_eight_views_share_one_canonical_form(self):
        state = game_state(open_field(), (4, 4), coins=[(4, 2)],
                           bombs=[((6, 4), 2)])
        features = f.state_to_features(state)
        canonical = canonicalize_v4(features)[0]
        for g in GROUP:
            self.assertEqual(canonicalize_v4(transform_v4(features, g))[0],
                             canonical)


class OpportunisticBombingTests(unittest.TestCase):
    def test_crate_value_is_reported_while_walking_to_a_coin(self):
        """Goal 2: the objective still points at the coin, and the bomb
        feature simultaneously reports that a crate is in blast range."""
        field = open_field()
        field[4, 3] = 1                       # crate right next to us
        state = game_state(field, (4, 4), coins=[(7, 4)])
        features = f.state_to_features(state)

        self.assertEqual(features[f.F_OBJECTIVE], 1,
                         "objective should still head RIGHT toward the coin")
        self.assertEqual(features[f.F_BOMB_OPPORTUNITY], f.BOMB_CRATES,
                         "a safe bomb hitting a crate must be visible here")

    def test_empty_bomb_is_ranked_below_a_crate_bomb(self):
        state = game_state(open_field(), (4, 4), coins=[(7, 4)])
        features = f.state_to_features(state)
        self.assertEqual(features[f.F_BOMB_OPPORTUNITY], f.BOMB_EMPTY)
        self.assertLess(f.BOMB_EMPTY, f.BOMB_CRATES)

    def test_opponent_outranks_crates(self):
        field = open_field()
        field[4, 3] = 1
        state = game_state(field, (4, 4), others=[(4, 6)])
        self.assertEqual(f.state_to_features(state)[f.F_BOMB_OPPORTUNITY],
                         f.BOMB_OPPONENT)

    def test_unavailable_or_unsurvivable_bomb_is_zero(self):
        state = game_state(open_field(), (4, 4), bomb_available=False)
        self.assertEqual(f.state_to_features(state)[f.F_BOMB_OPPORTUNITY],
                         f.BOMB_NONE)

        # a sealed corridor shorter than the blast: every tile we could reach
        # is inside our own explosion, so the bomb is unsurvivable
        field = open_field()
        for y in range(2, 7):
            field[3, y] = -1
            field[5, y] = -1
        field[4, 2] = -1
        field[4, 6] = -1
        pocket = game_state(field, (4, 3))
        self.assertEqual(f.state_to_features(pocket)[f.F_BOMB_OPPORTUNITY],
                         f.BOMB_NONE)


class TrapAvoidanceTests(unittest.TestCase):
    def test_open_board_is_open(self):
        state = game_state(open_field(), (4, 4))
        self.assertEqual(f.state_to_features(state)[f.F_MOBILITY],
                         f.MOBILITY_OPEN)

    def test_stone_walls_can_trap(self):
        field = open_field()
        # a two-tile pocket: (4,4) and (4,5), sealed by stone on every side
        field[3, 4] = field[5, 4] = field[4, 3] = -1
        field[3, 5] = field[5, 5] = field[4, 6] = -1
        state = game_state(field, (4, 4))
        self.assertEqual(f.state_to_features(state)[f.F_MOBILITY],
                         f.MOBILITY_TRAP)

    def test_crates_can_trap(self):
        field = open_field()
        # same pocket, built from crates instead of stone
        field[3, 4] = field[5, 4] = field[4, 3] = 1
        field[3, 5] = field[5, 5] = field[4, 6] = 1
        state = game_state(field, (4, 4))
        self.assertEqual(f.state_to_features(state)[f.F_MOBILITY],
                         f.MOBILITY_TRAP)

    def test_opponents_can_trap(self):
        """Goal 3: being boxed in by agents reads the same as being boxed in
        by terrain, which is the whole point of one merged mobility number."""
        field = open_field()
        field[4, 3] = -1
        state = game_state(field, (4, 4), others=[(3, 4), (5, 4), (4, 5)])
        self.assertEqual(f.state_to_features(state)[f.F_MOBILITY],
                         f.MOBILITY_TRAP)

    def test_bombs_block_mobility_too(self):
        field = open_field()
        field[4, 3] = -1
        state = game_state(field, (4, 4),
                           bombs=[((3, 4), 3), ((5, 4), 3), ((4, 5), 3)])
        self.assertEqual(f.state_to_features(state)[f.F_MOBILITY],
                         f.MOBILITY_TRAP)



class CrateMaximisationTests(unittest.TestCase):
    def test_richer_bomb_spot_beats_nearer_one(self):
        """One step RIGHT reaches a 1-crate spot; three steps LEFT reach a
        spot whose blast covers 3 crates. The objective must head LEFT."""
        field = open_field(11)
        field[7, 4] = 1                          # lone crate (spot at (6,4))
        field[1, 4] = 1                          # cluster around (2,4)
        field[2, 3] = 1
        field[2, 5] = 1
        state = game_state(field, (5, 4))
        features = f.state_to_features(state)
        self.assertEqual(features[f.F_OBJECTIVE], 3,   # LEFT
                         "objective should prefer the 3-crate spot")

    def test_many_crates_rank_above_one(self):
        field = open_field(11)
        field[4, 3] = 1
        field[4, 5] = 1
        state = game_state(field, (4, 4))
        self.assertEqual(f.state_to_features(state)[f.F_BOMB_OPPORTUNITY],
                         f.BOMB_CRATES_MANY)
        self.assertGreater(f.BOMB_CRATES_MANY, f.BOMB_CRATES)

    def test_revealed_coin_outranks_crates(self):
        """A visible coin always wins the objective cascade, which is what
        makes the agent collect freshly revealed coins before returning to
        crate work."""
        field = open_field(11)
        field[2, 4] = 1
        state = game_state(field, (5, 4), coins=[(7, 4)])
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 1)  # RIGHT


if __name__ == "__main__":
    unittest.main()
