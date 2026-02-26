# ASCII-only. Basic 5x5 GridWorld treasure hunt environment.
# API: reset() -> state_id ; step(action) -> (state_id, reward, done, info)
# Action mapping: 0=Up, 1=Right, 2=Down, 3=Left, 4=Stay

from typing import Tuple, Dict, Any, Optional

Action = int  # 0..4

class GridWorldEnv:
    def __init__(self,
                 width: int = 5,
                 height: int = 5,
                 start: Tuple[int, int] = (0, 4),
                 goal: Tuple[int, int] = (4, 0),
                 treasure: Tuple[int, int] = (2, 2),
                 walls: Optional[set] = None,
                 traps: Optional[set] = None,
                 step_cost: float = -0.1,
                 hit_cost: float = -1.0,
                 trap_cost: float = -10.0,
                 pickup_reward: float = +10.0,
                 success_reward: float = +20.0,
                 shaping: bool = True,
                 max_steps: int = 200):
        self.W = width
        self.H = height
        self.start = start
        self.goal = goal
        self.treasure = treasure
        self.walls = walls or {(1, 1), (1, 2), (3, 3)}
        self.traps = traps or {(2, 1), (3, 1)}
        self.step_cost = step_cost
        self.hit_cost = hit_cost
        self.trap_cost = trap_cost
        self.pickup_reward = pickup_reward
        self.success_reward = success_reward
        self.shaping = shaping
        self.max_steps = max_steps

        # runtime state
        self.pos = start
        self.has_treasure = False
        self.steps = 0

    # --- Public API ---

    def reset(self, seed: Optional[int] = None) -> int:
        # seed kept for API compatibility; env is deterministic here
        self.pos = self.start
        self.has_treasure = False
        self.steps = 0
        return self._encode_state()

    def step(self, action: Action) -> Tuple[int, float, bool, Dict[str, Any]]:
        assert 0 <= action <= 4, "invalid action"
        self.steps += 1

        old_pos = self.pos
        old_potential = self._potential(old_pos)

        nx, ny = self._move(self.pos, action)

        reward = self.step_cost
        done = False
        info: Dict[str, Any] = {}

        # block by boundary or wall
        if not self._in_bounds((nx, ny)) or (nx, ny) in self.walls:
            nx, ny = old_pos
            reward += self.hit_cost

        self.pos = (nx, ny)

        # pickup once
        if (not self.has_treasure) and self.pos == self.treasure:
            self.has_treasure = True
            reward += self.pickup_reward
            info["picked"] = True

        # trap penalty (non-terminal in base mode)
        if self.pos in self.traps:
            reward += self.trap_cost
            info["trap"] = True

        # success
        if self.has_treasure and self.pos == self.goal:
            reward += self.success_reward
            done = True
            info["success"] = True

        # shaping (potential-based; difference guarantees policy invariance)
        if self.shaping:
            reward += self._potential(self.pos) - old_potential

        # timeout
        if self.steps >= self.max_steps and not done:
            done = True
            info["timeout"] = True

        return self._encode_state(), reward, done, info

    # --- Helpers ---

    def _encode_state(self) -> int:
        # state_id in [0, 49] for 5x5 with carry bit
        x, y = self.pos
        return (1 if self.has_treasure else 0) * (self.W * self.H) + (y * self.W + x)

    def _move(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        x, y = pos
        if action == 0:   # Up
            return x, y - 1
        if action == 1:   # Right
            return x + 1, y
        if action == 2:   # Down
            return x, y + 1
        if action == 3:   # Left
            return x - 1, y
        return x, y       # Stay

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.W and 0 <= y < self.H

    def _target(self) -> Tuple[int, int]:
        return self.goal if self.has_treasure else self.treasure

    def _potential(self, pos: Tuple[int, int]) -> float:
        # linear negative Manhattan distance scaled to ~0.05 per step improvement
        if not self.shaping:
            return 0.0
        tx, ty = self._target()
        x, y = pos
        dist = abs(tx - x) + abs(ty - y)
        return -0.05 * dist

    # --- Debug text render ---

      # --- Debug text render ---
    # --- Debug text render ---
    def render_text(self) -> str:
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]
        for (wx, wy) in self.walls:
            grid[wy][wx] = "#"
        for (tx, ty) in self.traps:
            grid[ty][tx] = "X"
        gx, gy = self.goal
        grid[gy][gx] = "G"
        tx, ty = self.treasure
        if not self.has_treasure:
            grid[ty][tx] = "T"
        sx, sy = self.start
        if grid[sy][sx] == ".":
            grid[sy][sx] = "S"
        x, y = self.pos
        grid[y][x] = "*" if self.has_treasure else "P"

        lines = []
        for row in range(self.H):
            lines.append(" ".join(grid[row]))
        legend = "Legend: P=player, *=player+treasure, T=treasure, G=goal, #=wall, X=trap, S=start"
        return "\n".join(lines + [legend])
