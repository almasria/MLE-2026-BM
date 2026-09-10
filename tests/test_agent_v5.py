"""Tests for features v5: the engagement component.

    - advantage window is detected exactly when I am armed and the nearest
      reachable opponent is not
    - during an advantage window the objective heads for that opponent
    - solo scenarios (no opponents) produce engagement == 0 and otherwise
      identical features to v4, so nothing achieved so far regresses
    - D4 canonicalisation stays invariant with the extra component
"""

import unittest

import numpy as np

import importlib

v4 = importlib.import_module("agent_code.RL-Team.featuresv4")
f = importlib.import_module("agent_code.RL-Team.featuresv5")
_sym = importlib.import_module("agent_code.RL-Team.symmetry")
GROUP, canonicalize_v5, transform_v5 = _sym.GROUP, _sym.canonicalize_v5, _sym.transform_v5
from tests.test_agent_v4 import game_state, open_field


def opponent(pos, armed):
    return ('opp', 0, armed, pos)


class EngagementTests(unittest.TestCase):
    def test_no_opponent_means_none(self):
        state = game_state(open_field(), (4, 4), coins=[(6, 4)])
        self.assertEqual(f.state_to_features(state)[f.F_ENGAGEMENT],
                         f.ENGAGE_NONE)

    def test_far_opponent_means_none(self):
        field = open_field(17)
        state = game_state(field, (1, 1))
        state['others'] = [opponent((15, 15), False)]
        self.assertEqual(f.state_to_features(state)[f.F_ENGAGEMENT],
                         f.ENGAGE_NONE)

    def test_armed_me_unarmed_opponent_is_advantage(self):
        state = game_state(open_field(), (4, 4), bomb_available=True)
        state['others'] = [opponent((4, 7), False)]
        self.assertEqual(f.state_to_features(state)[f.F_ENGAGEMENT],
                         f.ENGAGE_ADVANTAGE)

    def test_both_armed_is_no_advantage(self):
        state = game_state(open_field(), (4, 4), bomb_available=True)
        state['others'] = [opponent((4, 7), True)]
        self.assertEqual(f.state_to_features(state)[f.F_ENGAGEMENT],
                         f.ENGAGE_NO_ADVANTAGE)

    def test_unarmed_me_is_no_advantage_even_if_they_are_unarmed(self):
        state = game_state(open_field(), (4, 4), bomb_available=False)
        state['others'] = [opponent((4, 7), False)]
        self.assertEqual(f.state_to_features(state)[f.F_ENGAGEMENT],
                         f.ENGAGE_NO_ADVANTAGE)

    def test_advantage_puts_opponent_before_a_FAR_coin(self):
        """Coin 5 steps LEFT, vulnerable opponent 2 steps DOWN: hunt first."""
        field = open_field(13)
        state = game_state(field, (6, 6), coins=[(1, 6)], bomb_available=True)
        state['others'] = [opponent((6, 9), False)]
        features = f.state_to_features(state)
        self.assertEqual(features[f.F_OBJECTIVE], 2)          # DOWN toward (6,9)

    def test_a_NEAR_coin_beats_the_advantage_hunt(self):
        """Coin 2 steps LEFT is a real point right there: take it first."""
        state = game_state(open_field(), (4, 4), coins=[(2, 4)],
                           bomb_available=True)
        state['others'] = [opponent((4, 7), False)]
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 3)  # LEFT

    def test_no_advantage_keeps_the_coin_objective(self):
        state = game_state(open_field(), (4, 4), coins=[(2, 4)],
                           bomb_available=True)
        state['others'] = [opponent((4, 7), True)]
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 3)  # LEFT


class NoRegressionTests(unittest.TestCase):
    def test_solo_features_match_v4_plus_zero(self):
        """Without opponents v5 must equal v4 with engagement appended
        (allowing for the EMPTY->NONE fold, which never affects the mask)."""
        field = open_field(11)
        field[4, 3] = 1
        for coins in ([(7, 4)], []):
            state = game_state(field, (4, 4), coins=coins)
            a = list(v4.state_to_features(state))
            b = list(f.state_to_features(state))
            if a[v4.F_BOMB_OPPORTUNITY] == v4.BOMB_EMPTY:
                a[v4.F_BOMB_OPPORTUNITY] = v4.BOMB_NONE
            self.assertEqual(b[:9], a)
            self.assertEqual(b[9], f.ENGAGE_NONE)

    def test_tuple_is_ten_small_ints(self):
        state = game_state(open_field(), (4, 4), coins=[(6, 4)])
        features = f.state_to_features(state)
        self.assertEqual(len(features), f.FEATURE_LENGTH)
        self.assertTrue(all(isinstance(v, int) and 0 <= v <= 4 for v in features))


class SymmetryV5Tests(unittest.TestCase):
    def test_all_views_share_one_canonical_form(self):
        state = game_state(open_field(), (4, 4), coins=[(4, 2)],
                           bombs=[((6, 4), 2)], bomb_available=True)
        state['others'] = [opponent((4, 7), False)]
        features = f.state_to_features(state)
        canonical = canonicalize_v5(features)[0]
        for g in GROUP:
            self.assertEqual(canonicalize_v5(transform_v5(features, g))[0],
                             canonical)



class AtBombSpotTests(unittest.TestCase):
    def test_standing_on_the_best_spot_does_not_fall_through_to_hunting(self):
        """On the best crate spot with no coin in sight, the objective is
        'none' (bomb here) even though opponents exist elsewhere. Falling
        through to the hunt direction made the agent walk off the spot and
        back again."""
        field = open_field(11)
        field[4, 3] = 1
        field[4, 5] = 1
        state = game_state(field, (4, 4), bomb_available=True)
        state['others'] = [opponent((8, 8), True)]
        features = f.state_to_features(state)
        self.assertEqual(features[f.F_OBJECTIVE], f.NO_DIR)
        self.assertEqual(features[f.F_BOMB_OPPORTUNITY], f.BOMB_CRATES_MANY)

    def test_without_crates_the_hunt_still_applies(self):
        state = game_state(open_field(11), (4, 4), bomb_available=True)
        state['others'] = [opponent((8, 4), True)]
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 1)   # RIGHT



class ContestedCoinCascadeTests(unittest.TestCase):
    def _board(self):
        field = open_field(15)
        field[3, 5] = 1                     # crates near the agent at (4,4)
        field[5, 3] = 1
        return field

    def test_far_coin_with_closer_opponent_ranks_below_crate_work(self):
        """Coin 8 steps away with an opponent right next to it: work the
        nearby crates instead of losing the race."""
        state = game_state(self._board(), (4, 4), coins=[(12, 4)],
                           bomb_available=True)
        state['others'] = [opponent((12, 5), True), opponent((11, 5), True)]
        features = f.state_to_features(state)
        self.assertNotEqual(features[f.F_OBJECTIVE], 1,
                            "should not head RIGHT toward the contested coin")
        self.assertNotEqual(features[f.F_OBJECTIVE], f.NO_DIR,
                            "should head for the nearby crates instead")

    def test_far_coin_with_no_competition_is_still_taken(self):
        state = game_state(self._board(), (4, 4), coins=[(12, 4)],
                           bomb_available=True)
        state['others'] = [opponent((2, 12), True)]     # far from the coin
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 1)  # RIGHT

    def test_contested_coin_beats_idling_when_no_crates_remain(self):
        state = game_state(open_field(15), (4, 4), coins=[(12, 4)],
                           bomb_available=True)
        state['others'] = [opponent((12, 5), True), opponent((11, 5), True)]
        self.assertEqual(f.state_to_features(state)[f.F_OBJECTIVE], 1)


class EscapeClearanceTests(unittest.TestCase):
    def test_escape_prefers_the_side_away_from_an_armed_opponent(self):
        """Own bomb under the agent in a corridor open both ways; an armed
        opponent waits on the right. The escape must go LEFT."""
        field = open_field(15)
        for x in range(1, 14):              # horizontal corridor at y=7
            field[x, 6] = -1
            field[x, 8] = -1
        state = game_state(field, (7, 7), bombs=[((7, 7), 3)], bomb_available=False)
        state['others'] = [opponent((11, 7), True)]
        features = f.state_to_features(state)
        self.assertEqual(features[f.F_SAFE_DIR], 3)                     # LEFT

    def test_bomb_is_unsurvivable_when_the_only_exit_leads_to_an_armed_opponent(self):
        field = open_field(15)
        for x in range(1, 14):
            field[x, 6] = -1
            field[x, 8] = -1
        field[3, 7] = -1                    # corridor sealed on the left
        state = game_state(field, (5, 7), bomb_available=True)
        state['others'] = [opponent((10, 7), True)]
        self.assertEqual(f.state_to_features(state)[f.F_BOMB_OPPORTUNITY],
                         f.BOMB_NONE)


if __name__ == "__main__":
    unittest.main()
