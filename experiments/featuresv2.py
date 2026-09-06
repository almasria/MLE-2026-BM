"""Legacy feature set kept for comparison.

This is the earlier Q-agent feature representation, kept as a reference while the
new strategic feature set is introduced in features.py.
"""

from collections import deque

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def state_to_features(game_state):
    """Original coin-focused feature set.

    Features:
        0: direction to nearest and reachable coin
        1: is UP free?
        2: is RIGHT free?
        3: is DOWN free?
        4: is LEFT free?
    """
    if game_state is None:
        return None

    field = game_state['field']
    x, y = game_state['self'][3]
    coins = game_state['coins']

    coin_direction = bfs_direction_to_nearest(field, (x, y), set(coins))
    neighbors = []
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        neighbors.append(1 if field[nx, ny] == 0 else 0)
    return (coin_direction, *neighbors)


def bfs_direction_to_nearest(field, start, targets) -> int:
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
            if nxt in first_move:
                continue
            if field[nxt[0], nxt[1]] != 0:
                continue
            first_move[nxt] = i if first_move[current] is None else first_move[current]
            queue.append(nxt)
    return 4
