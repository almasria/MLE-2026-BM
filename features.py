"""Feature engineering — the heart of the agent"""

from collections import deque
import numpy as np


# Direction encoding for the "where is the nearest coin" feature
# 0 = up, 1 = right, 2 = down, 3 = left, 4 = no coin reachable / on top of coin
DIRECTIONS = [
    (0, -1),    # matches UP
    (1, 0),     # matches RIGHT
    (0, 1),     # matches DOWN
    (-1, 0)     # matches LEFT
]

def state_to_features(game_state):
    """
    Covert game state into W2 feature representation v1.

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

    coin_direction = bfs_direction_to_nearest(
        field,
        (x,y),
        set(coins)
    )

    neighbors = []

    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        neighbors.append(1 if field[nx, ny] == 0 else 0)

    return (coin_direction, *neighbors)

def bfs_direction_to_nearest(field, start, targets) -> int:
    """
    Breadth-first search from `start` over free tiles (field == 0).
    Returns the index of the FIRST STEP direction (0..3) on a shortest path
    to the nearest target, or 4 if no target is reachable (or we're on one).

    This one helper generalizes to later stages: pass crates, safe tiles,
    or opponents as `targets` to get "direction to nearest X" features.
    """
    if not targets or start in targets:
        return 4
    
    # Standard BFS, remembering the first move that led to each visited tile
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
            if field[nxt[0], nxt[1]] != 0:   # wall or crate blocks the path
                continue
            # Propagate the first move: if we're one step from start, this
            # direction IS the first move; otherwise inherit it.
            first_move[nxt] = i if first_move[current] is None else first_move[current]
            queue.append(nxt)
    
    return 4  # no coin reachable
    