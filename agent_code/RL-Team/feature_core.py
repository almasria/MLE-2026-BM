"""Shared feature computations for RL-Team.

Both feature versions (featuresv4, featuresv5) assemble their tuples from the
helpers in this module. Keeping the computations in one place means the two
versions cannot drift apart.

Timing model, with m = number of the agent's future moves (m = 1 is the move
being chosen now):
  * a bomb observed with countdown t makes its blast tiles lethal for
    t < m <= t + EXPLOSION_EXTRA
  * an explosion_map value e makes a tile lethal for m <= e
A tile is "eternally safe" when no current bomb or explosion ever reaches it.
These rules were checked against the engine's step order (moves, coins,
explosion update, bomb update, kill evaluation).
"""

import os
from collections import deque

# --- game constants ----------------------------------------------------------
BOMB_RANGE = 3
BOMB_TIMER = 4                          # steps from drop to detonation
HYPOTHETICAL_BOMB_TIMER = BOMB_TIMER - 1  # a bomb dropped now shows countdown 3 next step
EXPLOSION_EXTRA = 2                     # steps a blast stays lethal after detonation
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # UP, RIGHT, DOWN, LEFT
INF = 10 ** 9
NO_DIR = 4

# --- tunables (environment variables, so sweeps need no code changes) -------
MOBILITY_HORIZON = int(os.environ.get("Q_AGENT_MOBILITY_HORIZON", "4"))
MOBILITY_TRAPPED = int(os.environ.get("Q_AGENT_MOBILITY_TRAPPED", "2"))
MOBILITY_TIGHT = int(os.environ.get("Q_AGENT_MOBILITY_TIGHT", "5"))
# escape planning for our own bomb treats tiles next to opponents as blocked
CONSERVATIVE_BOMB = os.environ.get("Q_AGENT_CONSERVATIVE_BOMB", "1") == "1"
# armed opponents within this Manhattan radius are assumed to bomb right now
THREAT_RADIUS = int(os.environ.get("Q_AGENT_THREAT_RADIUS", "4"))
# movement pruning also plans against those hypothetical opponent bombs
PARANOID_MOVES = os.environ.get("Q_AGENT_PARANOID_MOVES", "1") == "1"
# tournament scenario: 9 coins, 3 opponents (used to detect the endgame)
TOTAL_COINS = int(os.environ.get("Q_AGENT_TOTAL_COINS", "9"))
INITIAL_OPPONENTS = int(os.environ.get("Q_AGENT_INITIAL_OPPONENTS", "3"))
ENGAGE_RADIUS = int(os.environ.get("Q_AGENT_ENGAGE_RADIUS", "5"))
# an escape's safe tile must keep this Manhattan distance from armed
# opponents for a bomb to count as survivable (0 disables the requirement)
ESCAPE_CLEARANCE = int(os.environ.get("Q_AGENT_ESCAPE_CLEARANCE", "2"))
NEAR_COIN = int(os.environ.get("Q_AGENT_NEAR_COIN", "3"))
ALLOW_EMPTY_BOMB = os.environ.get("Q_AGENT_ALLOW_EMPTY_BOMB", "0") == "1"
MANY_CRATES = 2                         # crates in blast for a "many" bomb

# --- shared tuple layout (indices 0-8 are identical in v4 and v5) -----------
F_OBJECTIVE = 0
F_NEIGHBOURS = slice(1, 5)
F_URGENCY = 5
F_SAFE_DIR = 6
F_BOMB_OPPORTUNITY = 7
F_MOBILITY = 8

URGENCY_SAFE, URGENCY_SOON, URGENCY_IMMINENT = 0, 1, 2
BOMB_NONE, BOMB_EMPTY, BOMB_CRATES, BOMB_CRATES_MANY, BOMB_OPPONENT = 0, 1, 2, 3, 4
MOBILITY_TRAP, MOBILITY_TIGHT_BUCKET, MOBILITY_OPEN = 0, 1, 2
ENGAGE_NONE, ENGAGE_ADVANTAGE, ENGAGE_NO_ADVANTAGE = 0, 1, 2


class Perception:
    """Everything derived from one game_state that the feature tuples need.

    Building this once per step and letting each feature version pick its
    components keeps the expensive searches (escape, mobility, engagement)
    from being computed twice.
    """

    def __init__(self, game_state):
        gs = game_state
        self.field = gs['field']
        self.x, self.y = gs['self'][3]
        self.bomb_available = gs['self'][2]
        self.coins = gs['coins']
        self.explosions = gs['explosion_map']
        self.others = gs['others']
        self.opponents = {o[3] for o in gs['others']}
        self.armed_opponents = {o[3] for o in gs['others'] if o[2]}
        self.bomb_positions = {pos for pos, _ in gs['bombs']}
        self.lethal_from, self.lethal_until = lethal_windows(self.field, gs['bombs'])
        self.game_state = gs
        pos = (self.x, self.y)

        self.neighbours = [1 if self.passable((self.x + dx, self.y + dy), 1) else 0
                           for dx, dy in DIRECTIONS]
        self.coin_dir, self.coin_dist, self.coin_target = bfs_nearest_with_distance(
            self.field, pos, set(self.coins), self.walkable_now)
        # crates stop being worth bombing once no coin can be hidden under
        # them AND the opponents are reachable; while crates block the path
        # to the opponents they remain the way to the remaining points
        self.crates_worthless = (bool(self.opponents)
                                 and hidden_coins_remaining(gs) <= 0
                                 and self.hunt_objective() != NO_DIR)
        self.urgency = self._urgency()
        self.safe_dir = NO_DIR
        if self.urgency != URGENCY_SAFE:
            _, self.safe_dir, _ = temporal_escape(
                pos, self.passable, self.lethal_from, self.lethal_until, self.explosions,
                keep_away_from=self.armed_opponents)
        self.bomb_opportunity = compute_bomb_opportunity(
            self.field, pos, self.bomb_available, self.bomb_positions, self.opponents,
            self.lethal_from, self.lethal_until, self.explosions, self.armed_opponents,
            crates_count=not self.crates_worthless)
        self.mobility = compute_mobility(self.field, pos, self.walkable_now)

    # tile predicates -------------------------------------------------------
    def passable(self, tile, m):
        """Can the agent stand on `tile` after its m-th move?"""
        return (self.field[tile] == 0 and tile not in self.bomb_positions
                and tile not in self.opponents
                and not is_lethal_at(tile, m, self.lethal_from, self.lethal_until,
                                     self.explosions))

    def walkable_now(self, tile):
        return (self.field[tile] == 0 and tile not in self.bomb_positions
                and tile not in self.opponents and self.explosions[tile] == 0)

    # objective helpers -----------------------------------------------------
    def crate_objective(self):
        """(direction, at_spot): direction toward the best bomb spot, or
        NO_DIR with at_spot=True when already standing on it."""
        if self.crates_worthless:
            return NO_DIR, False
        return direction_to_best_bomb_spot(self.field, (self.x, self.y), self.walkable_now)

    def hunt_objective(self):
        if not self.opponents:
            return NO_DIR
        return bfs_direction_to_nearest(
            self.field, (self.x, self.y),
            adjacent_tiles(self.field, self.opponents, self.walkable_now), self.walkable_now)

    def coin_contested(self):
        """True when the nearest coin is a race the agent is likely to lose:
        an opponent is at least as close to it, an armed opponent stands next
        to it, or two or more opponents crowd it."""
        if self.coin_target is None or not self.opponents:
            return False
        cx, cy = self.coin_target
        free = lambda t: self.field[t] == 0 and t not in self.bomb_positions
        crowd = 0
        for o in self.opponents:
            d = abs(o[0] - cx) + abs(o[1] - cy)
            if d <= 2:
                crowd += 1
            if d <= 1 and o in self.armed_opponents:
                return True
            if d < self.coin_dist:                 # cheap bound before a BFS
                _, od, _ = bfs_nearest_with_distance(self.field, o, {self.coin_target}, free)
                if od <= self.coin_dist:
                    return True
        return crowd >= 2

    def engagement(self):
        return compute_engagement(self.field, (self.x, self.y), self.bomb_available,
                                  self.others, self.walkable_now)

    def _urgency(self):
        deadline = self.lethal_from.get((self.x, self.y), INF)
        if self.explosions[self.x, self.y] > 0:
            deadline = 0
        if deadline >= INF:
            return URGENCY_SAFE
        return URGENCY_SOON if deadline >= 2 else URGENCY_IMMINENT


# --- action pruning ---------------------------------------------------------

def survivable_actions(game_state):
    """Action indices (0-5) after which at least one survival line exists.

    Only certainly fatal actions are removed; choosing among the survivable
    ones is left to the learned policy. When PARANOID_MOVES is set, nearby
    armed opponents are assumed to bomb immediately.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    bomb_available = game_state['self'][2]
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    opponents = {o[3] for o in game_state['others']}
    armed = {o[3] for o in game_state['others'] if o[2]}
    bomb_positions = {pos for pos, _ in bombs}

    real_from, real_until = lethal_windows(field, bombs)
    lethal_from, lethal_until = real_from, real_until
    if PARANOID_MOVES:
        lethal_from, lethal_until = with_opponent_threats(
            field, (x, y), armed, real_from, real_until)

    def passable(tile, m):
        return (field[tile] == 0 and tile not in bomb_positions and tile not in opponents
                and not is_lethal_at(tile, m, lethal_from, lethal_until, explosions))

    ok = set()
    for i, (dx, dy) in enumerate(DIRECTIONS):
        nxt = (x + dx, y + dy)
        if passable(nxt, 1) and temporal_escape(nxt, passable, lethal_from, lethal_until,
                                                explosions, start_time=1,
                                                keep_away_from=armed)[0]:
            ok.add(i)
    if not is_lethal_at((x, y), 1, lethal_from, lethal_until, explosions) and \
            temporal_escape((x, y), passable, lethal_from, lethal_until, explosions,
                            start_time=1, keep_away_from=armed)[0]:
        ok.add(4)
    if bomb_available and compute_bomb_opportunity(
            field, (x, y), bomb_available, bomb_positions, opponents,
            real_from, real_until, explosions, armed) != BOMB_NONE:
        ok.add(5)
    return ok


# --- feature computations ---------------------------------------------------

def compute_bomb_opportunity(field, pos, bomb_available, bomb_positions, opponents,
                             lethal_from, lethal_until, explosions,
                             armed_opponents=(), crates_count=True):
    """Rank a bomb dropped on `pos`: NONE (unavailable or unsurvivable), EMPTY,
    CRATES, CRATES_MANY or OPPONENT. Survivability is checked against existing
    bombs plus hypothetical bombs from nearby armed opponents."""
    if not bomb_available:
        return BOMB_NONE
    own_blast = blast_coords(field, pos)

    if CONSERVATIVE_BOMB:
        hyp_from, hyp_until = with_opponent_threats(
            field, pos, armed_opponents, lethal_from, lethal_until)
    else:
        hyp_from, hyp_until = dict(lethal_from), dict(lethal_until)
    for tile in own_blast:
        hyp_from[tile] = min(hyp_from.get(tile, INF), HYPOTHETICAL_BOMB_TIMER)
        hyp_until[tile] = max(hyp_until.get(tile, -1),
                              HYPOTHETICAL_BOMB_TIMER + EXPLOSION_EXTRA)

    blocked = {pos}                                   # cannot re-enter own bomb tile
    if CONSERVATIVE_BOMB:
        blocked |= opponents
        blocked |= adjacent_tiles(field, opponents, lambda t: field[t] == 0)

    def passable_hyp(tile, m):
        return (field[tile] == 0 and tile not in bomb_positions
                and tile not in opponents and tile not in blocked
                and not is_lethal_at(tile, m, hyp_from, hyp_until, explosions))

    nearby_armed = {o for o in armed_opponents
                    if abs(o[0] - pos[0]) + abs(o[1] - pos[1]) <= THREAT_RADIUS + 2}
    survivable, _, _ = temporal_escape(
        pos, passable_hyp, hyp_from, hyp_until, explosions,
        blocked_after_departure={pos}, keep_away_from=nearby_armed,
        min_clearance=ESCAPE_CLEARANCE if nearby_armed else 0)
    if not survivable:
        return BOMB_NONE
    if any(o in own_blast for o in opponents):
        return BOMB_OPPONENT
    crates_hit = sum(1 for t in own_blast if field[t] == 1) if crates_count else 0
    if crates_hit >= MANY_CRATES:
        return BOMB_CRATES_MANY
    if crates_hit:
        return BOMB_CRATES
    return BOMB_EMPTY


def compute_engagement(field, pos, bomb_available, others, walkable):
    """Tactical status versus the nearest reachable opponent.

    Returns (engagement, vulnerable_targets). ADVANTAGE means the agent holds
    a bomb while the nearest opponent within ENGAGE_RADIUS has spent theirs;
    vulnerable_targets are the free tiles next to that opponent.
    """
    if not others:
        return ENGAGE_NONE, set()

    dist = {pos: 0}
    queue = deque([pos])
    while queue:
        cur = queue.popleft()
        if dist[cur] >= ENGAGE_RADIUS:
            continue
        for dx, dy in DIRECTIONS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt not in dist and walkable(nxt):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)

    nearest, nearest_d = None, None
    for o in others:
        ox, oy = o[3]
        d = min((dist[(ox + dx, oy + dy)] for dx, dy in DIRECTIONS
                 if (ox + dx, oy + dy) in dist), default=None)
        if d is not None and (nearest_d is None or d < nearest_d):
            nearest, nearest_d = o, d
    if nearest is None:
        return ENGAGE_NONE, set()

    if bomb_available and not nearest[2]:
        ox, oy = nearest[3]
        targets = {(ox + dx, oy + dy) for dx, dy in DIRECTIONS
                   if walkable((ox + dx, oy + dy))}
        return ENGAGE_ADVANTAGE, targets
    return ENGAGE_NO_ADVANTAGE, set()


def hidden_coins_remaining(game_state):
    """Upper bound on coins still hidden under crates.

    While every opponent is alive the sum of all scores equals the coins
    collected so far; after deaths, sum(scores) - 5 * deaths is a lower bound
    on collected coins (a kill is worth 5).
    """
    scores = game_state['self'][1] + sum(o[1] for o in game_state['others'])
    dead = max(0, INITIAL_OPPONENTS - len(game_state['others']))
    collected = max(0, scores - 5 * dead)
    return TOTAL_COINS - collected - len(game_state['coins'])


def with_opponent_threats(field, pos, armed_opponents, lethal_from, lethal_until):
    """Lethal timetable plus a hypothetical bomb at each armed opponent within
    THREAT_RADIUS, as if dropped this step."""
    x, y = pos
    hyp_from, hyp_until = dict(lethal_from), dict(lethal_until)
    for ox, oy in armed_opponents:
        if abs(ox - x) + abs(oy - y) > THREAT_RADIUS:
            continue
        for tile in blast_coords(field, (ox, oy)):
            hyp_from[tile] = min(hyp_from.get(tile, INF), HYPOTHETICAL_BOMB_TIMER)
            hyp_until[tile] = max(hyp_until.get(tile, -1),
                                  HYPOTHETICAL_BOMB_TIMER + EXPLOSION_EXTRA)
    return hyp_from, hyp_until


def compute_mobility(field, pos, walkable):
    """Reachable free area within MOBILITY_HORIZON steps, bucketed.
    Walls, crates, bombs and opponents all block alike."""
    seen = {pos}
    frontier = deque([(pos, 0)])
    reached = 0
    while frontier:
        tile, d = frontier.popleft()
        if d == MOBILITY_HORIZON:
            continue
        for dx, dy in DIRECTIONS:
            nxt = (tile[0] + dx, tile[1] + dy)
            if nxt in seen or not walkable(nxt):
                continue
            seen.add(nxt)
            reached += 1
            if reached > MOBILITY_TIGHT:
                return MOBILITY_OPEN
            frontier.append((nxt, d + 1))
    if reached <= MOBILITY_TRAPPED:
        return MOBILITY_TRAP
    if reached <= MOBILITY_TIGHT:
        return MOBILITY_TIGHT_BUCKET
    return MOBILITY_OPEN


# --- geometry and timing ----------------------------------------------------

def blast_coords(field, pos):
    """Tiles a bomb at `pos` covers; stone walls stop the blast, crates do not."""
    tiles = {pos}
    for dx, dy in DIRECTIONS:
        for r in range(1, BOMB_RANGE + 1):
            t = (pos[0] + dx * r, pos[1] + dy * r)
            if field[t] == -1:
                break
            tiles.add(t)
    return tiles


def lethal_windows(field, bombs):
    """Per tile: earliest countdown reaching it and the last step it stays deadly."""
    lethal_from, lethal_until = {}, {}
    for pos, t in bombs:
        for tile in blast_coords(field, pos):
            lethal_from[tile] = min(lethal_from.get(tile, INF), t)
            lethal_until[tile] = max(lethal_until.get(tile, -1), t + EXPLOSION_EXTRA)
    return lethal_from, lethal_until


def is_lethal_at(tile, m, lethal_from, lethal_until, explosions):
    """Is `tile` deadly after the agent's m-th move?"""
    return (lethal_from.get(tile, INF) < m <= lethal_until.get(tile, -1)
            or (explosions[tile] > 0 and m <= explosions[tile]))


def temporal_escape(start, passable, lethal_from, lethal_until, explosions,
                    max_m=None, blocked_after_departure=(), start_time=0,
                    keep_away_from=(), min_clearance=0):
    """BFS over (tile, time); WAIT is a legal move that costs one step.

    Goal: reach a tile no current bomb or explosion ever touches. Returns
    (survivable, first_move_dir, arrival_step). The horizon covers every
    current threat (BOMB_TIMER + EXPLOSION_EXTRA) plus slack for detours.

    With keep_away_from (armed opponent positions), the search does not stop
    at the first safe tile: among the lines found it prefers the one whose
    safe tile is farthest from those opponents, accepting up to one extra
    step to get there. With min_clearance > 0 only safe tiles at least that
    far from them count at all.
    """
    if max_m is None:
        max_m = BOMB_TIMER + EXPLOSION_EXTRA + 2
    blocked_after_departure = set(blocked_after_departure)
    keep_away_from = list(keep_away_from)

    def eternally_safe(tile):
        return lethal_from.get(tile, INF) >= INF and explosions[tile] == 0

    def clearance(tile):
        if not keep_away_from:
            return INF
        return min(abs(tile[0] - ox) + abs(tile[1] - oy) for ox, oy in keep_away_from)

    if eternally_safe(start) and clearance(start) >= min_clearance:
        return True, NO_DIR, 0

    candidates = []                                  # (arrival, -clearance, first)
    seen = {(start, start_time)}
    queue = deque([(start, start_time, None)])
    while queue:
        tile, m, first = queue.popleft()
        if m > start_time and eternally_safe(tile):
            c = clearance(tile)
            if c >= min_clearance:
                if not keep_away_from:               # plain shortest escape
                    return True, first if first is not None else NO_DIR, m
                candidates.append((m, -c, first if first is not None else NO_DIR))
            continue                                  # no need to search past safety
        if m >= max_m + start_time:
            continue
        moves = [(NO_DIR, tile)] + [(i, (tile[0] + dx, tile[1] + dy))
                                    for i, (dx, dy) in enumerate(DIRECTIONS)]
        for mv, nxt in moves:
            if mv == NO_DIR:
                ok = not is_lethal_at(tile, m + 1, lethal_from, lethal_until, explosions)
            else:
                ok = nxt not in blocked_after_departure and passable(nxt, m + 1)
            if ok and (nxt, m + 1) not in seen:
                seen.add((nxt, m + 1))
                queue.append((nxt, m + 1, mv if first is None else first))
    if not candidates:
        return False, NO_DIR, INF
    fastest = min(c[0] for c in candidates)
    arrival, neg_c, first = min(c for c in candidates if c[0] <= fastest + 1)
    return True, first, arrival


# --- pathfinding ------------------------------------------------------------

def direction_to_best_bomb_spot(field, start, walkable):
    """(first step, at_spot) toward the reachable tile whose bomb destroys the
    most crates; ties broken by distance. at_spot is True when `start` is
    that tile, so callers can stop the objective cascade there."""
    spots = crate_adjacent_tiles(field)
    if not spots:
        return NO_DIR, False
    if start in spots:
        spots = spots | {start}

    dist, first = {start: 0}, {start: NO_DIR}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        cx, cy = cur
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (cx + dx, cy + dy)
            if nxt in dist or not walkable(nxt):
                continue
            dist[nxt] = dist[cur] + 1
            first[nxt] = i if first[cur] == NO_DIR else first[cur]
            queue.append(nxt)

    best, best_key = None, None
    for spot in spots:
        if spot not in dist:
            continue
        crates = sum(1 for t in blast_coords(field, spot) if field[t] == 1)
        key = (-crates, dist[spot])
        if best_key is None or key < best_key:
            best, best_key = spot, key
    if best is None:
        return NO_DIR, False
    return first[best], best == start


def adjacent_tiles(field, positions, walkable):
    """Walkable tiles next to any of `positions`."""
    spots = set()
    for px, py in positions:
        for dx, dy in DIRECTIONS:
            t = (px + dx, py + dy)
            if walkable(t):
                spots.add(t)
    return spots


def crate_adjacent_tiles(field):
    """Free tiles from which a bomb would hit at least one crate."""
    spots = set()
    w, h = field.shape
    for x in range(w):
        for y in range(h):
            if field[x, y] == 1:
                for dx, dy in DIRECTIONS:
                    t = (x + dx, y + dy)
                    if 0 <= t[0] < w and 0 <= t[1] < h and field[t] == 0:
                        spots.add(t)
    return spots


def bfs_direction_to_nearest(field, start, targets, walkable=None):
    """First step of a shortest path to the nearest target; NO_DIR if on one
    or none is reachable."""
    return bfs_nearest_with_distance(field, start, targets, walkable)[0]


def bfs_nearest_with_distance(field, start, targets, walkable=None):
    """(first step, path length, target tile) for the nearest target."""
    if walkable is None:
        walkable = lambda t: field[t] == 0
    if not targets:
        return NO_DIR, INF, None
    if start in targets:
        return NO_DIR, 0, start
    queue = deque([start])
    first_move, dist = {start: None}, {start: 0}
    while queue:
        cur = queue.popleft()
        if cur in targets:
            return first_move[cur], dist[cur], cur
        cx, cy = cur
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (cx + dx, cy + dy)
            if nxt in first_move or not walkable(nxt):
                continue
            first_move[nxt] = i if first_move[cur] is None else first_move[cur]
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)
    return NO_DIR, INF, None
