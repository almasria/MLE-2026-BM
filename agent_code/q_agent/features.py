"""Feature engineering — v2 (final): TIME-AWARE danger.

The first draft failed the survival milestone because danger was a timeless boolean:
the agent could not distinguish "bomb just dropped, 4 steps of slack" from
"detonating now", and its escape BFS routed through tiles without knowing
whether it would clear them in time. This version fixes both using information the
game_state provided all along: bomb COUNTDOWNS and the EXPLOSION_MAP.

Timing model (m = number of my future moves, m = 1 is the move I pick now):
  - bomb with countdown t covers its blast tiles lethally for  t < m <= t+2
    (detonation after t steps, explosion stays deadly one extra step)
  - explosion_map value e makes a tile lethal for  m <= e
A tile is "eternally safe" if no current bomb/explosion ever makes it lethal.

Feature tuple (same arity as v2; index 5 is now an urgency bucket):
    0: objective_dir   toward nearest reachable coin, else crate-adjacent tile
    1-4: up/right/down/left safe to step on NOW (walkable & not lethal at m=1)
    5: urgency         0 = my tile eternally safe; else by time-to-lethal:
                       1 (>=3 steps), 2 (2 steps), 3 (<=1 step)
    6: safe_dir        first move of a TIME-AWARE escape path to an eternally
                       safe tile (4 = none needed / only WAIT survives / none)
    7: bomb_safe       1 if dropping a bomb HERE is survivable (time-aware,
                       accounting for existing bombs too)
    8: crates_in_range crates my bomb here would destroy (capped 3)
"""

from collections import deque

BOMB_RANGE = 3
BOMB_TIMER = 4          # steps from drop to detonation
EXPLOSION_EXTRA = 2     # blast tiles stay lethal this many steps after t

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]     # UP, RIGHT, DOWN, LEFT
INF = 10 ** 9


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = {o[3] for o in game_state['others']}
    bomb_positions = {pos for pos, _ in bombs}

    lethal_from, lethal_until = lethal_windows(field, bombs, explosions)

    def lethal_at(tile, m):
        return lethal_from.get(tile, INF) < m <= lethal_until.get(tile, -1) \
            or m <= explosions[tile]

    def passable(tile, m):
        """Can I stand on `tile` after my m-th move?"""
        return (field[tile] == 0 and tile not in bomb_positions
                and tile not in others and not lethal_at(tile, m))

    # 1-4: neighbors safe to enter right now (m = 1)
    neighbors = [1 if passable((x + dx, y + dy), 1) else 0
                 for dx, dy in DIRECTIONS]

    # 0: objective (timing-agnostic path planning is fine here)
    plain_walkable = lambda t: field[t] == 0 and t not in bomb_positions \
        and t not in others and explosions[t] == 0
    objective_dir = bfs_direction_to_nearest(field, (x, y), set(coins), plain_walkable)
    if objective_dir == 4 and (x, y) not in set(coins):
        objective_dir = bfs_direction_to_nearest(
            field, (x, y), crate_adjacent_tiles(field), plain_walkable)

    # 5: urgency of MY tile
    my_deadline = lethal_from.get((x, y), INF)
    if explosions[x, y] > 0:
        my_deadline = 0
    if my_deadline >= INF:
        urgency = 0
    elif my_deadline >= 3:
        urgency = 1
    elif my_deadline == 2:
        urgency = 2
    else:
        urgency = 3

    # 6: time-aware escape (only meaningful when not eternally safe)
    if urgency > 0:
        _, safe_dir = temporal_escape((x, y), passable, lethal_from, explosions)
    else:
        safe_dir = 4

    # 7: would dropping a bomb here be survivable? add hypothetical own bomb
    own_blast = blast_coords(field, (x, y))
    own_from = {t: min(lethal_from.get(t, INF), BOMB_TIMER) for t in own_blast}
    own_until = {t: max(lethal_until.get(t, -1), BOMB_TIMER + EXPLOSION_EXTRA)
                 for t in own_blast}
    hyp_from = dict(lethal_from); hyp_from.update(own_from)
    hyp_until = dict(lethal_until); hyp_until.update(own_until)

    def passable_hyp(tile, m):
        lethal = hyp_from.get(tile, INF) < m <= hyp_until.get(tile, -1) \
            or m <= explosions[tile]
        return (field[tile] == 0 and tile not in bomb_positions
                and tile not in others and not lethal)

    survivable, _ = temporal_escape((x, y), passable_hyp, hyp_from, explosions)
    bomb_safe = 1 if survivable else 0

    # 8: crates my bomb would hit
    crates = min(3, sum(1 for t in own_blast if field[t] == 1))

    return (objective_dir, *neighbors, urgency, safe_dir, bomb_safe, crates)


# ---------------------------------------------------------------------------

def blast_coords(field, pos):
    tiles = {pos}
    for dx, dy in DIRECTIONS:
        for r in range(1, BOMB_RANGE + 1):
            t = (pos[0] + dx * r, pos[1] + dy * r)
            if field[t] == -1:
                break
            tiles.add(t)
    return tiles


def lethal_windows(field, bombs, explosions):
    """Per tile: earliest bomb deadline (countdown) and last lethal step."""
    lethal_from, lethal_until = {}, {}
    for pos, t in bombs:
        for tile in blast_coords(field, pos):
            lethal_from[tile] = min(lethal_from.get(tile, INF), t)
            lethal_until[tile] = max(lethal_until.get(tile, -1),
                                     t + EXPLOSION_EXTRA)
    return lethal_from, lethal_until


def temporal_escape(start, passable, lethal_from, explosions, max_m=None):
    """BFS over (tile, time). WAIT is a legal move (staying costs a step).
    Goal: reach a tile that is ETERNALLY safe (never lethal from current
    bombs/explosions). Returns (survivable, first_move_dir 0..4)."""
    if max_m is None:
        max_m = BOMB_TIMER + EXPLOSION_EXTRA + 2

    def eternally_safe(tile):
        return lethal_from.get(tile, INF) >= INF and explosions[tile] == 0

    if eternally_safe(start):
        return True, 4

    seen = {(start, 0)}
    queue = deque([(start, 0, None)])          # tile, time, first move
    while queue:
        tile, m, first = queue.popleft()
        if m > 0 and eternally_safe(tile):
            return True, first if first is not None else 4
        if m == max_m:
            continue
        # WAIT edge: stay put one step (only if staying is survivable then)
        moves = [(4, tile)] + [(i, (tile[0] + dx, tile[1] + dy))
                               for i, (dx, dy) in enumerate(DIRECTIONS)]
        for mv, nxt in moves:
            ok = passable(nxt, m + 1) if mv != 4 else not (
                lethal_from.get(tile, INF) < m + 1 or m + 1 <= explosions[tile])
            if ok and (nxt, m + 1) not in seen:
                seen.add((nxt, m + 1))
                queue.append((nxt, m + 1, mv if first is None else first))
    return False, 4


def free_tiles(field):
    w, h = field.shape
    return {(x, y) for x in range(w) for y in range(h) if field[x, y] == 0}


def crate_adjacent_tiles(field):
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
    if walkable is None:
        walkable = lambda t: field[t] == 0
    if not targets or start in targets:
        return 4
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
    return 4
