"""Feature engineering v4 — smaller state space, opportunistic bombing,
trap avoidance.

Design goals (in order):

1. SMALLEST WORKABLE STATE SPACE.  v3 used 11 components whose product was
   5*16*4*5*2*4*5*2 = 128,000 raw combinations.  v4 uses 9 components with a
   product of 5*16*3*5*4*3 = 14,400, about 9x smaller, by MERGING correlated
   components instead of listing them side by side:
     - bomb_safe (2) x crates_in_range (4) x opp_in_blast (2) = 16 values
       collapse into one ranked bomb_opportunity feature with 4 values.
     - opp_dir (5) disappears: opponents already enter through the objective
       fallback (hunt when nothing else is left), through bomb_opportunity
       (bombing that hits someone), and through mobility (being boxed in).
     - urgency shrinks from 4 buckets to 3; the distinction between "3+ steps"
       and "2 steps" of slack never changed the correct action in practice.

2. OPPORTUNISTIC CRATE BOMBING.  crates_in_range is computed at EVERY tile,
   not only when the agent is heading for crates, so bomb_opportunity == 2
   fires while the agent walks toward a coin and happens to pass a crate it
   can safely blast.  The objective direction is unaffected, so the agent
   keeps its route and simply learns "bombing here is free value".

3. TRAP AVOIDANCE.  mobility is a flood fill over tiles the agent can actually
   move through, so stone walls, crates, bombs AND opponents all block it
   alike.  Being boxed in by three opponents looks the same as standing in a
   dead-end corridor, which is the property we want: both are situations to
   leave before dropping a bomb.

Feature tuple (9 components, all small ints):
    0: objective_dir  direction to nearest coin; else to a crate-adjacent
                      tile; else toward an opponent (endgame hunt). 4 = none.
    1-4: up/right/down/left  1 if that neighbour is enterable and not lethal
                      on the next step.  Used by the action mask.
    5: urgency        0 = my tile is eternally safe
                      1 = lethal in 2+ steps, 2 = lethal within 1 step
    6: safe_dir       first move of a time-aware escape path to an eternally
                      safe tile.  4 = already safe / no escape / only WAIT.
    7: bomb_opportunity
                      0 = bombing here is unavailable or unsurvivable
                      1 = safe but hits nothing
                      2 = safe and destroys crates
                      3 = safe and covers an opponent (highest value)
    8: mobility       reachable free area within MOBILITY_HORIZON steps:
                      0 = trapped (<= 2 tiles), 1 = constrained (<= 5), 2 = open

Timing model is unchanged from v3 (m = number of my future moves, m = 1 is
the move I am choosing now):
  - a bomb reported with countdown t makes its blast tiles lethal for
    t < m <= t + EXPLOSION_EXTRA
  - explosion_map value e makes a tile lethal for m <= e
A tile is "eternally safe" when no current bomb or explosion ever reaches it.

Carried over from the v3 fixes:
  - HYPOTHETICAL_BOMB_TIMER = BOMB_TIMER - 1, because a bomb dropped now is
    observed with countdown 3 on the next step, not 4.
  - the tile a bomb is dropped on cannot be re-entered while it ticks.
"""

import os
from collections import deque

BOMB_RANGE = 3
BOMB_TIMER = 4                          # steps from drop to detonation
HYPOTHETICAL_BOMB_TIMER = BOMB_TIMER - 1
EXPLOSION_EXTRA = 2                     # extra deadly steps after detonation

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # UP, RIGHT, DOWN, LEFT
INF = 10 ** 9

# --- tunables (sweepable without touching code) -----------------------------
MOBILITY_HORIZON = int(os.environ.get("Q_AGENT_MOBILITY_HORIZON", "4"))
MOBILITY_TRAPPED = int(os.environ.get("Q_AGENT_MOBILITY_TRAPPED", "2"))
MOBILITY_TIGHT = int(os.environ.get("Q_AGENT_MOBILITY_TIGHT", "5"))
# treat tiles next to an opponent as blocked when checking our own bomb's
# escape route: opponents move into corridors and cancel a "guaranteed" escape
CONSERVATIVE_BOMB = os.environ.get("Q_AGENT_CONSERVATIVE_BOMB", "1") == "1"
# WORST-CASE OPPONENT BOMBS. Death-log analysis vs rule_based (24/24 self-
# kills): we dropped a survivable bomb, an adjacent opponent dropped theirs,
# the exit sealed, no escape existed. rule_based bombs whenever it touches
# you. So: any armed opponent within THREAT_RADIUS is assumed to bomb NOW,
# and our escapes (bomb drops AND moves) must survive that as well.
THREAT_RADIUS = int(os.environ.get("Q_AGENT_THREAT_RADIUS", "3"))
# ENDGAME: in the tournament scenario only 9 coins exist. Once they are all
# collected, crates are worthless (nothing left to reveal) and the only
# remaining points are kills -- so the objective switches to hunting and
# crate bombs stop counting as valuable. Coins collected so far are inferred
# from visible scores: with all opponents alive, sum(scores) == coins exactly;
# after deaths, sum(scores) - 5*deaths is a lower bound (kills are 5 each).
TOTAL_COINS = int(os.environ.get("Q_AGENT_TOTAL_COINS", "9"))
INITIAL_OPPONENTS = int(os.environ.get("Q_AGENT_INITIAL_OPPONENTS", "3"))
PARANOID_MOVES = os.environ.get("Q_AGENT_PARANOID_MOVES", "1") == "1"

# --- symbolic tuple layout (import these instead of hard-coding indices) ----
F_OBJECTIVE = 0
F_NEIGHBOURS = slice(1, 5)
F_URGENCY = 5
F_SAFE_DIR = 6
F_BOMB_OPPORTUNITY = 7
F_MOBILITY = 8
FEATURE_LENGTH = 9

NO_DIR = 4

URGENCY_SAFE, URGENCY_SOON, URGENCY_IMMINENT = 0, 1, 2
BOMB_NONE, BOMB_EMPTY, BOMB_CRATES, BOMB_CRATES_MANY, BOMB_OPPONENT = 0, 1, 2, 3, 4
MANY_CRATES = 2      # >= this many crates in blast counts as a MANY bomb
MOBILITY_TRAP, MOBILITY_TIGHT_BUCKET, MOBILITY_OPEN = 0, 1, 2


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]
    bomb_available = game_state['self'][2]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    opponents = {o[3] for o in game_state['others']}
    armed_opponents = {o[3] for o in game_state['others'] if o[2]}
    bomb_positions = {pos for pos, _ in bombs}

    lethal_from, lethal_until = lethal_windows(field, bombs)

    def passable(tile, m):
        """Can I stand on `tile` after my m-th move?"""
        return (field[tile] == 0
                and tile not in bomb_positions
                and tile not in opponents
                and not is_lethal_at(tile, m, lethal_from, lethal_until,
                                     explosions))

    def walkable_now(tile):
        return (field[tile] == 0
                and tile not in bomb_positions
                and tile not in opponents
                and explosions[tile] == 0)

    # --- 1-4: neighbours that are safe to enter on the next move ------------
    neighbours = [1 if passable((x + dx, y + dy), 1) else 0
                  for dx, dy in DIRECTIONS]

    # --- 0: objective, with the coin -> crate -> hunt cascade ---------------
    objective_dir = bfs_direction_to_nearest(field, (x, y), set(coins),
                                             walkable_now)
    crates_worthless = bool(opponents) and hidden_coins_remaining(game_state) <= 0
    if objective_dir == NO_DIR and not crates_worthless:
        objective_dir = direction_to_best_bomb_spot(field, (x, y), walkable_now)
    if objective_dir == NO_DIR and opponents:
        objective_dir = bfs_direction_to_nearest(
            field, (x, y), adjacent_tiles(field, opponents, walkable_now),
            walkable_now)

    # --- 5: urgency of my own tile ------------------------------------------
    my_deadline = lethal_from.get((x, y), INF)
    if explosions[x, y] > 0:
        my_deadline = 0
    if my_deadline >= INF:
        urgency = URGENCY_SAFE
    elif my_deadline >= 2:
        urgency = URGENCY_SOON
    else:
        urgency = URGENCY_IMMINENT

    # --- 6: time-aware escape direction -------------------------------------
    if urgency == URGENCY_SAFE:
        safe_dir = NO_DIR
    else:
        _, safe_dir, _ = temporal_escape((x, y), passable, lethal_from,
                                         lethal_until, explosions)

    # --- 7: merged bomb opportunity ------------------------------------------
    bomb_opportunity = compute_bomb_opportunity(
        field, (x, y), bomb_available, bomb_positions, opponents,
        lethal_from, lethal_until, explosions, armed_opponents,
        crates_count=not crates_worthless)

    # --- 8: mobility / trap detection ----------------------------------------
    mobility = compute_mobility(field, (x, y), walkable_now)

    return (int(objective_dir), *neighbours, int(urgency), int(safe_dir),
            int(bomb_opportunity), int(mobility))


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------

def survivable_actions(game_state):
    """Action indices (0-5) after which at least one survival line exists.

    This is the full-depth version of the neighbour mask: a tile can be safe
    to ENTER (m = 1) yet be a dead end at m >= 2.  For every candidate action
    we ask temporal_escape whether the agent can still reach an eternally
    safe tile AFTER committing to it.  Only certainly-fatal actions are
    removed; choosing among the survivable ones stays with the learned policy,
    so this is death-avoidance, not decision-making.
    """
    field = game_state['field']
    x, y = game_state['self'][3]
    bomb_available = game_state['self'][2]
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    opponents = {o[3] for o in game_state['others']}
    armed_opponents = {o[3] for o in game_state['others'] if o[2]}
    bomb_positions = {pos for pos, _ in bombs}

    lethal_from, lethal_until = lethal_windows(field, bombs)
    real_from, real_until = lethal_from, lethal_until
    if PARANOID_MOVES:
        lethal_from, lethal_until = with_opponent_threats(
            field, (x, y), armed_opponents, lethal_from, lethal_until)

    def passable(tile, m):
        return (field[tile] == 0 and tile not in bomb_positions
                and tile not in opponents
                and not is_lethal_at(tile, m, lethal_from, lethal_until,
                                     explosions))

    ok = set()
    # movement: commit to the neighbour, then require an escape from there
    for i, (dx, dy) in enumerate(DIRECTIONS):
        nxt = (x + dx, y + dy)
        if not passable(nxt, 1):
            continue
        alive, _, _ = temporal_escape(nxt, passable, lethal_from, lethal_until,
                                      explosions, start_time=1)
        if alive:
            ok.add(i)
    # WAIT: survive standing still this step, then escape from here
    if not is_lethal_at((x, y), 1, lethal_from, lethal_until, explosions):
        alive, _, _ = temporal_escape((x, y), passable, lethal_from,
                                      lethal_until, explosions, start_time=1)
        if alive:
            ok.add(4)
    # BOMB: bomb_opportunity already runs this exact check with the
    # hypothetical bomb added, so reuse it
    if bomb_available and compute_bomb_opportunity(
            field, (x, y), bomb_available, bomb_positions, opponents,
            real_from, real_until, explosions, armed_opponents) != BOMB_NONE:
        ok.add(5)
    return ok


def compute_bomb_opportunity(field, pos, bomb_available, bomb_positions,
                             opponents, lethal_from, lethal_until, explosions,
                             armed_opponents=(), crates_count=True):
    """Rank the value of dropping a bomb on `pos` (see module docstring).

    Merging survivability and payoff into one ordered feature is what keeps
    the state space small: the agent never needs to see "unsafe but valuable"
    as a separate case, because such a bomb should never be dropped.
    """
    if not bomb_available:
        return BOMB_NONE

    own_blast = blast_coords(field, pos)

    # add the hypothetical bomb to the lethal timetable -- and, if armed
    # opponents are close, THEIR hypothetical bombs too
    if CONSERVATIVE_BOMB:
        hyp_from, hyp_until = with_opponent_threats(
            field, pos, armed_opponents, lethal_from, lethal_until)
    else:
        hyp_from, hyp_until = dict(lethal_from), dict(lethal_until)
    for tile in own_blast:
        hyp_from[tile] = min(hyp_from.get(tile, INF), HYPOTHETICAL_BOMB_TIMER)
        hyp_until[tile] = max(hyp_until.get(tile, -1),
                              HYPOTHETICAL_BOMB_TIMER + EXPLOSION_EXTRA)

    blocked = {pos}                      # cannot walk back onto our own bomb
    if CONSERVATIVE_BOMB:
        # opponents step into corridors and invalidate a "guaranteed" escape,
        # so treat their neighbourhood as unavailable while planning it
        blocked |= opponents
        blocked |= adjacent_tiles(field, opponents, lambda t: field[t] == 0)

    def passable_hyp(tile, m):
        return (field[tile] == 0
                and tile not in bomb_positions
                and tile not in opponents
                and tile not in blocked
                and not is_lethal_at(tile, m, hyp_from, hyp_until, explosions))

    survivable, _, _ = temporal_escape(pos, passable_hyp, hyp_from, hyp_until,
                                       explosions, blocked_after_departure={pos})
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


def hidden_coins_remaining(game_state):
    """Upper bound on coins still hidden under crates (see TOTAL_COINS note).
    Only meaningful with opponents; solo scenarios have 50 coins and no
    opponents, and the callers fall back to crate work in that case."""
    scores = game_state['self'][1] + sum(o[1] for o in game_state['others'])
    dead = max(0, INITIAL_OPPONENTS - len(game_state['others']))
    collected_lower_bound = max(0, scores - 5 * dead)
    return TOTAL_COINS - collected_lower_bound - len(game_state['coins'])


def with_opponent_threats(field, pos, armed_opponents, lethal_from, lethal_until):
    """Copy of the lethal timetable plus a hypothetical bomb at every ARMED
    opponent within THREAT_RADIUS (Manhattan), as if dropped this step."""
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
    """Flood fill the free area reachable within MOBILITY_HORIZON steps.

    Walls, crates, bombs and opponents all block alike, so this single number
    answers "am I being boxed in?" regardless of what is doing the boxing.
    """
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
            if reached > MOBILITY_TIGHT:        # already "open", stop early
                return MOBILITY_OPEN
            frontier.append((nxt, d + 1))
    if reached <= MOBILITY_TRAPPED:
        return MOBILITY_TRAP
    if reached <= MOBILITY_TIGHT:
        return MOBILITY_TIGHT_BUCKET
    return MOBILITY_OPEN


# ---------------------------------------------------------------------------
# Shared geometry / timing helpers
# ---------------------------------------------------------------------------

def blast_coords(field, pos):
    """Tiles a bomb at `pos` covers. Stone walls stop it; crates do not."""
    tiles = {pos}
    for dx, dy in DIRECTIONS:
        for r in range(1, BOMB_RANGE + 1):
            t = (pos[0] + dx * r, pos[1] + dy * r)
            if field[t] == -1:
                break
            tiles.add(t)
    return tiles


def lethal_windows(field, bombs):
    """Per tile: earliest countdown reaching it, and last step it stays deadly."""
    lethal_from, lethal_until = {}, {}
    for pos, t in bombs:
        for tile in blast_coords(field, pos):
            lethal_from[tile] = min(lethal_from.get(tile, INF), t)
            lethal_until[tile] = max(lethal_until.get(tile, -1),
                                     t + EXPLOSION_EXTRA)
    return lethal_from, lethal_until


def is_lethal_at(tile, m, lethal_from, lethal_until, explosions):
    """Is `tile` deadly after my m-th move?"""
    return (lethal_from.get(tile, INF) < m <= lethal_until.get(tile, -1)
            or (explosions[tile] > 0 and m <= explosions[tile]))


def temporal_escape(start, passable, lethal_from, lethal_until, explosions,
                    max_m=None, blocked_after_departure=(), start_time=0):
    """BFS over (tile, time) pairs; WAIT is a legal move that costs one step.

    Goal: reach a tile no current bomb or explosion ever touches.
    Returns (survivable, first_move_dir 0..4, arrival_step).

    The horizon only needs to outlast every current threat
    (BOMB_TIMER + EXPLOSION_EXTRA); the +2 is slack for detours that step
    sideways or wait before the corridor to safety opens.
    """
    if max_m is None:
        max_m = BOMB_TIMER + EXPLOSION_EXTRA + 2
    blocked_after_departure = set(blocked_after_departure)

    def eternally_safe(tile):
        return lethal_from.get(tile, INF) >= INF and explosions[tile] == 0

    if eternally_safe(start):
        return True, NO_DIR, 0

    seen = {(start, start_time)}
    queue = deque([(start, start_time, None)])
    while queue:
        tile, m, first = queue.popleft()
        if m > start_time and eternally_safe(tile):
            return True, first if first is not None else NO_DIR, m
        if m >= max_m + start_time:
            continue
        moves = [(NO_DIR, tile)] + [(i, (tile[0] + dx, tile[1] + dy))
                                    for i, (dx, dy) in enumerate(DIRECTIONS)]
        for mv, nxt in moves:
            if mv == NO_DIR:                     # WAIT: survive standing still
                ok = not is_lethal_at(tile, m + 1, lethal_from, lethal_until,
                                      explosions)
            else:
                ok = (nxt not in blocked_after_departure
                      and passable(nxt, m + 1))
            if ok and (nxt, m + 1) not in seen:
                seen.add((nxt, m + 1))
                queue.append((nxt, m + 1, mv if first is None else first))
    return False, NO_DIR, INF


def direction_to_best_bomb_spot(field, start, walkable):
    """First step toward the reachable tile whose bomb would destroy the MOST
    crates; ties broken by distance.  Implements "prefer the placement that
    clears as many crates as possible with one bomb" -- the value side is
    already proportional (CRATE_DESTROYED fires once per crate), this supplies
    the matching decision-time signal without adding a feature dimension.
    """
    spots = crate_adjacent_tiles(field)
    if not spots:
        return NO_DIR
    if start in spots:
        spots = spots | {start}

    # one BFS gives distance and first step to every reachable tile
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
        key = (-crates, dist[spot])          # most crates, then closest
        if best_key is None or key < best_key:
            best, best_key = spot, key
    if best is None:
        return NO_DIR
    return first[best]


def adjacent_tiles(field, positions, walkable):
    """Free tiles next to any of `positions` (targets for approach/avoidance)."""
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


def bfs_direction_to_nearest(field, start, targets, walkable=None) -> int:
    """First step of a shortest path to the nearest target; 4 if none/on one."""
    if walkable is None:
        walkable = lambda t: field[t] == 0
    if not targets or start in targets:
        return NO_DIR

    queue = deque([start])
    first_move = {start: None}
    while queue:
        current = queue.popleft()
        if current in targets:
            return first_move[current]
        cx, cy = current
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nxt = (cx + dx, cy + dy)
            if nxt in first_move or not walkable(nxt):
                continue
            first_move[nxt] = i if first_move[current] is None else first_move[current]
            queue.append(nxt)
    return NO_DIR
