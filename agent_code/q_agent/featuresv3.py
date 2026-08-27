"""Strategic feature version for q_agent.

This is the real new feature implementation. The compatibility import in
features.py simply points here so the rest of the project can keep importing
from .features without changing call sites.
"""

from collections import deque

BOMB_RANGE = 3
BOMB_TIMER = 4
EXPLOSION_EXTRA = 2

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
INF = 10 ** 9


def state_to_features(game_state):
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosions = game_state['explosion_map']
    others = game_state['others']
    others_pos = {o[3] for o in others}
    bomb_positions = {pos for pos, _ in bombs}

    lethal_from, lethal_until = lethal_windows(field, bombs, explosions)

    def lethal_at(tile, m):
        return lethal_from.get(tile, INF) < m <= lethal_until.get(tile, -1) or m <= explosions[tile]

    def passable(tile, m):
        return field[tile] == 0 and tile not in bomb_positions and tile not in others_pos and not lethal_at(tile, m)

    plain_walkable = lambda t: field[t] == 0 and t not in bomb_positions and t not in others_pos and explosions[t] == 0
    objective_dir = bfs_direction_to_nearest(field, (x, y), set(coins), plain_walkable)
    if objective_dir == 4:
        objective_dir = bfs_direction_to_nearest(field, (x, y), crate_adjacent_tiles(field), plain_walkable)
    if objective_dir == 4 and others:
        hunt = set()
        for o in others:
            for dx, dy in DIRECTIONS:
                t = (o[3][0] + dx, o[3][1] + dy)
                if field[t] == 0:
                    hunt.add(t)
        objective_dir = bfs_direction_to_nearest(field, (x, y), hunt, plain_walkable)

    neighbors = [1 if passable((x + dx, y + dy), 1) else 0 for dx, dy in DIRECTIONS]

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

    if urgency > 0:
        _, safe_dir, _ = temporal_escape((x, y), passable, lethal_from, explosions)
    else:
        safe_dir = 4

    own_blast = blast_coords(field, (x, y))
    own_from = {t: min(lethal_from.get(t, INF), BOMB_TIMER) for t in own_blast}
    own_until = {t: max(lethal_until.get(t, -1), BOMB_TIMER + EXPLOSION_EXTRA) for t in own_blast}
    hyp_from = dict(lethal_from)
    hyp_from.update(own_from)
    hyp_until = dict(lethal_until)
    hyp_until.update(own_until)

    def passable_hyp(tile, m):
        lethal = hyp_from.get(tile, INF) < m <= hyp_until.get(tile, -1) or m <= explosions[tile]
        return field[tile] == 0 and tile not in bomb_positions and tile not in others_pos and not lethal

    survivable, _, arrival = temporal_escape((x, y), passable_hyp, hyp_from, explosions)
    bomb_safe = 1 if survivable else 0

    # post-drop safe exits: neighbors safe at time=1 under hypothetical windows
    post_drop_safe_exits = sum(1 if passable_hyp((x + dx, y + dy), 1) else 0 for dx, dy in DIRECTIONS)
    post_drop_safe_exits_bucket = min(3, post_drop_safe_exits)  # 3 means 3+

    # escape margin bucket: bucketize arrival time to eternally-safe tile
    if not survivable or arrival >= INF:
        escape_margin_bucket = 0
    elif arrival == 1:
        escape_margin_bucket = 1
    elif arrival <= 3:
        escape_margin_bucket = 2
    else:
        escape_margin_bucket = 3
    crates = min(3, sum(1 for t in own_blast if field[t] == 1))

    opp_positions = [o[3] for o in others]
    opp_dir = bfs_direction_to_nearest(field, (x, y), set(opp_positions), plain_walkable) if opp_positions else 4
    opp_in_blast = 1 if any(pos in own_blast for pos in opp_positions) else 0

    danger_score = compute_danger_score(field, (x, y), bombs, explosions)
    coin_score = compute_coin_score(field, (x, y), coins, bombs, explosions)
    attack_score = compute_attack_score(field, (x, y), others, bombs, explosions)
    strategic_score = compute_strategic_score(field, (x, y), coins)
    safe_exit_count = sum(neighbors)
    enemy_dist_bucket = compute_enemy_distance_bucket((x, y), opp_positions)
    bomb_value = compute_bomb_value(field, (x, y), others, coins)
    coin_density = local_density((x, y), coins, radius=3)
    crate_density = local_crate_density(field, (x, y), radius=3)

    return (
        objective_dir,
        *neighbors,
        urgency,
        safe_dir,
        bomb_safe,
        crates,
        opp_dir,
        opp_in_blast,
        danger_score,
        coin_score,
        attack_score,
        strategic_score,
        safe_exit_count,
        enemy_dist_bucket,
        bomb_value,
        coin_density,
        crate_density,
        escape_margin_bucket,
        post_drop_safe_exits_bucket,
    )


def compute_danger_score(field, pos, bombs, explosions):
    x, y = pos
    danger = 0.0
    if explosions[x, y] > 0:
        danger += 0.7

    for (bx, by), timer in bombs:
        if abs(bx - x) + abs(by - y) <= 2:
            danger += 0.25 / (1 + timer)

    safe_exits = 0
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and explosions[nx, ny] == 0:
                safe_exits += 1
    if safe_exits <= 1:
        danger += 0.25
    elif safe_exits == 2:
        danger += 0.1
    return min(1.0, danger)


def compute_coin_score(field, pos, coins, bombs, explosions):
    if not coins:
        return 0.0
    coin_targets = {(cx, cy) for cx, cy in coins}
    nearest = min(abs(cx - pos[0]) + abs(cy - pos[1]) for cx, cy in coin_targets)
    reachable = bfs_direction_to_nearest(field, pos, coin_targets) is not None
    density = local_density(pos, coins, radius=3)
    score = 0.65 * (1.0 - min(1.0, nearest / 12.0)) + 0.25 * (1.0 if reachable else 0.0) + 0.10 * density
    return min(1.0, max(0.0, score))


def compute_attack_score(field, pos, others, bombs, explosions):
    if not others:
        return 0.0
    opp_positions = [o[3] for o in others]
    nearest = min(abs(ox - pos[0]) + abs(oy - pos[1]) for ox, oy in opp_positions)
    in_blast = any((ox, oy) in blast_coords(field, pos) for ox, oy in opp_positions)
    safe_escape = 1 if has_safe_exit(field, pos, explosions) else 0
    survivable = 1 if can_survive_bomb(field, pos, bombs, explosions) else 0

    score = 0.0
    if in_blast:
        score += 0.5
    if survivable:
        score += 0.25
    if safe_escape:
        score += 0.15
    score += max(0.0, 1.0 - nearest / 8.0) * 0.1
    return min(1.0, score)


def compute_strategic_score(field, pos, coins):
    x, y = pos
    near_crates = 0
    safe_exits = 0
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0:
                safe_exits += 1
            if field[nx, ny] == 1:
                near_crates += 1
    center = 1.0 - min(1.0, (abs(x - field.shape[0] / 2) + abs(y - field.shape[1] / 2)) / 12.0)
    score = 0.25 * center + 0.30 * min(1.0, near_crates / 3.0) + 0.25 * min(1.0, safe_exits / 3.0) + 0.20 * min(1.0, len(coins) / 10.0)
    return min(1.0, score)


def compute_enemy_distance_bucket(pos, opp_positions):
    if not opp_positions:
        return 0
    nearest = min(abs(ox - pos[0]) + abs(oy - pos[1]) for ox, oy in opp_positions)
    if nearest <= 1:
        return 3
    if nearest <= 3:
        return 2
    if nearest <= 6:
        return 1
    return 0


def compute_bomb_value(field, pos, others, coins):
    blast = blast_coords(field, pos)
    hit_crates = sum(1 for x, y in blast if field[x, y] == 1)
    hit_opponents = sum(1 for _, _, _, (ox, oy) in others if (ox, oy) in blast)
    hit_coins = sum(1 for cx, cy in coins if (cx, cy) in blast)
    score = hit_crates * 0.5 + hit_opponents * 0.8 + hit_coins * 0.4
    return min(3, int(score))


def can_survive_bomb(field, pos, bombs, explosions):
    x, y = pos
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and explosions[nx, ny] == 0:
                return True
    return False


def has_safe_exit(field, pos, explosions):
    x, y = pos
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            if field[nx, ny] == 0 and explosions[nx, ny] == 0:
                return True
    return False


def local_density(pos, targets, radius=3):
    x, y = pos
    count = 0
    for tx, ty in targets:
        if abs(tx - x) + abs(ty - y) <= radius:
            count += 1
    return 0.0 if count == 0 else min(1.0, count / 3.0)


def local_crate_density(field, pos, radius=3):
    x, y = pos
    count = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 1:
                count += 1
    return 0.0 if count == 0 else min(1.0, count / 6.0)


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
    lethal_from, lethal_until = {}, {}
    for pos, t in bombs:
        for tile in blast_coords(field, pos):
            lethal_from[tile] = min(lethal_from.get(tile, INF), t)
            lethal_until[tile] = max(lethal_until.get(tile, -1), t + EXPLOSION_EXTRA)
    return lethal_from, lethal_until


def temporal_escape(start, passable, lethal_from, explosions, max_m=None):
    if max_m is None:
        max_m = BOMB_TIMER + EXPLOSION_EXTRA + 2

    def eternally_safe(tile):
        return lethal_from.get(tile, INF) >= INF and explosions[tile] == 0

    # If the start tile is already eternally safe, arrival time is 0
    if eternally_safe(start):
        return True, 4, 0

    seen = {(start, 0)}
    queue = deque([(start, 0, None)])
    while queue:
        tile, m, first = queue.popleft()
        if m > 0 and eternally_safe(tile):
            arrival = m
            return True, first if first is not None else 4, arrival
        if m == max_m:
            continue
        moves = [(4, tile)] + [(i, (tile[0] + dx, tile[1] + dy)) for i, (dx, dy) in enumerate(DIRECTIONS)]
        for mv, nxt in moves:
            ok = passable(nxt, m + 1) if mv != 4 else not (lethal_from.get(tile, INF) < m + 1 or m + 1 <= explosions[tile])
            if ok and (nxt, m + 1) not in seen:
                seen.add((nxt, m + 1))
                queue.append((nxt, m + 1, mv if first is None else first))
    return False, 4, INF


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
