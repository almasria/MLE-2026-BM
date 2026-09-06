"""Deterministic correctness tests for the Q-agent v3 feature semantics."""

import unittest

import numpy as np

from agent_code.q_agent import featuresv3 as features
from agent_code.q_agent.symmetry import GROUP, canonicalize_v3, transform_game_state


UP, RIGHT, DOWN, LEFT, WAIT = range(5)


def bordered_field(size=9):
    field = np.zeros((size, size), dtype=int)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    return field


def game_state(
    field,
    position,
    *,
    coins=(),
    bombs=(),
    others=(),
    explosions=None,
):
    if explosions is None:
        explosions = np.zeros_like(field)
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "bombs": list(bombs),
        "explosion_map": explosions,
        "coins": list(coins),
        "self": ("q_agent", 0, True, position),
        "others": [
            (f"opponent_{i}", 0, True, pos)
            for i, pos in enumerate(others)
        ],
        "user_input": None,
    }


def timed_passable(field, bomb_positions, lethal_from, lethal_until, explosions):
    def passable(tile, minute):
        return (
            field[tile] == 0
            and tile not in bomb_positions
            and not features.is_lethal_at(
                tile,
                minute,
                lethal_from,
                lethal_until,
                explosions,
            )
        )

    return passable


class QAgentFeatureTests(unittest.TestCase):
    def test_opponent_direction_targets_a_reachable_adjacent_tile(self):
        state = game_state(
            bordered_field(),
            (3, 3),
            others=[(6, 3)],
        )

        result = features.state_to_features(state)

        self.assertEqual(result[9], RIGHT)

    def test_reachable_and_unreachable_coins_are_distinguished(self):
        position = (2, 2)
        coin = (4, 2)
        explosions = np.zeros((9, 9), dtype=int)

        open_score = features.compute_coin_score(
            bordered_field(), position, [coin], [], explosions
        )

        blocked = bordered_field()
        for wall in ((3, 2), (4, 1), (5, 2), (4, 3)):
            blocked[wall] = -1
        blocked_score = features.compute_coin_score(
            blocked, position, [coin], [], explosions
        )

        distance_component = 0.65 * (1.0 - 2.0 / 12.0)
        density_component = 0.10 * (1.0 / 3.0)
        self.assertAlmostEqual(
            blocked_score,
            distance_component + density_component,
        )
        self.assertAlmostEqual(open_score - blocked_score, 0.25)

    def test_blast_timing_for_countdowns_zero_through_four(self):
        field = bordered_field()
        explosions = np.zeros_like(field)
        position = (4, 4)

        for timer in range(5):
            with self.subTest(timer=timer):
                lethal_from, lethal_until = features.lethal_windows(
                    field, [(position, timer)], explosions
                )
                for minute in range(1, timer + 1):
                    self.assertFalse(
                        features.is_lethal_at(
                            position,
                            minute,
                            lethal_from,
                            lethal_until,
                            explosions,
                        )
                    )
                for minute in (timer + 1, timer + 2):
                    self.assertTrue(
                        features.is_lethal_at(
                            position,
                            minute,
                            lethal_from,
                            lethal_until,
                            explosions,
                        )
                    )
                self.assertFalse(
                    features.is_lethal_at(
                        position,
                        timer + 3,
                        lethal_from,
                        lethal_until,
                        explosions,
                    )
                )

    def test_escape_from_an_open_corner(self):
        field = bordered_field()
        position = (1, 1)
        bombs = [(position, 3)]
        explosions = np.zeros_like(field)
        lethal_from, lethal_until = features.lethal_windows(
            field, bombs, explosions
        )
        passable = timed_passable(
            field, {position}, lethal_from, lethal_until, explosions
        )

        survivable, first_move, arrival = features.temporal_escape(
            position,
            passable,
            lethal_from,
            explosions,
            lethal_until=lethal_until,
        )

        self.assertTrue(survivable)
        self.assertIn(first_move, (RIGHT, DOWN))
        self.assertEqual(arrival, 2)

    def test_dead_end_inside_blast_has_no_escape(self):
        field = np.full((7, 7), -1, dtype=int)
        corridor = [(1, 1), (2, 1), (3, 1), (4, 1)]
        for tile in corridor:
            field[tile] = 0
        position = corridor[0]
        bombs = [(position, 3)]
        explosions = np.zeros_like(field)
        lethal_from, lethal_until = features.lethal_windows(
            field, bombs, explosions
        )
        passable = timed_passable(
            field, {position}, lethal_from, lethal_until, explosions
        )

        survivable, first_move, arrival = features.temporal_escape(
            position,
            passable,
            lethal_from,
            explosions,
            lethal_until=lethal_until,
        )

        self.assertFalse(survivable)
        self.assertEqual(first_move, WAIT)
        self.assertEqual(arrival, features.INF)

    def test_wait_can_be_the_only_safe_first_action(self):
        field = np.full((6, 6), -1, dtype=int)
        start, transit, goal = (1, 2), (2, 2), (3, 2)
        for tile in (start, transit, goal):
            field[tile] = 0
        explosions = np.zeros_like(field)
        lethal_from = {start: 2, transit: 0}
        lethal_until = {start: 3, transit: 1}
        passable = timed_passable(
            field, set(), lethal_from, lethal_until, explosions
        )

        survivable, first_move, arrival = features.temporal_escape(
            start,
            passable,
            lethal_from,
            explosions,
            lethal_until=lethal_until,
        )

        self.assertTrue(survivable)
        self.assertEqual(first_move, WAIT)
        self.assertEqual(arrival, 3)

    def test_multiple_overlapping_bombs_merge_their_lethal_times(self):
        field = bordered_field()
        explosions = np.zeros_like(field)
        overlap = (4, 4)
        bombs = [((4, 2), 1), ((2, 4), 2)]
        lethal_from, lethal_until = features.lethal_windows(
            field, bombs, explosions
        )

        self.assertEqual(lethal_from[overlap], 1)
        self.assertEqual(lethal_until[overlap], 4)
        self.assertFalse(
            features.is_lethal_at(
                overlap, 1, lethal_from, lethal_until, explosions
            )
        )
        for minute in (2, 3, 4):
            self.assertTrue(
                features.is_lethal_at(
                    overlap,
                    minute,
                    lethal_from,
                    lethal_until,
                    explosions,
                )
            )

    def test_hypothetical_bomb_tile_cannot_be_reentered(self):
        field = np.full((6, 6), -1, dtype=int)
        start, alcove, delayed_exit, goal = (
            (2, 2),
            (2, 1),
            (3, 2),
            (4, 2),
        )
        for tile in (start, alcove, delayed_exit, goal):
            field[tile] = 0
        explosions = np.zeros_like(field)
        lethal_from = {start: 1, alcove: 5, delayed_exit: 0}
        lethal_until = {start: 2, alcove: 7, delayed_exit: 3}
        passable = timed_passable(
            field, set(), lethal_from, lethal_until, explosions
        )

        without_bomb_block, _, _ = features.temporal_escape(
            start,
            passable,
            lethal_from,
            explosions,
            lethal_until=lethal_until,
        )
        with_bomb_block, _, _ = features.temporal_escape(
            start,
            passable,
            lethal_from,
            explosions,
            lethal_until=lethal_until,
            blocked_after_departure={start},
        )

        self.assertTrue(without_bomb_block)
        self.assertFalse(with_bomb_block)

    def test_all_symmetries_share_one_canonical_feature_tuple(self):
        field = bordered_field()
        # Force a unique first escape/movement direction. Directional BFS ties
        # are deliberately outside this invariance test.
        field[2, 3] = -1
        field[3, 2] = -1
        field[3, 4] = -1
        field[5, 3] = 1
        state = game_state(
            field,
            (3, 3),
            coins=[(6, 5)],
            bombs=[((3, 3), 2)],
            others=[(6, 3)],
        )
        canonical = canonicalize_v3(features.state_to_features(state))[0]

        for symmetry in GROUP:
            with self.subTest(symmetry=repr(symmetry)):
                transformed = transform_game_state(state, symmetry)
                transformed_canonical = canonicalize_v3(
                    features.state_to_features(transformed)
                )[0]
                self.assertEqual(transformed_canonical, canonical)


if __name__ == "__main__":
    unittest.main()
