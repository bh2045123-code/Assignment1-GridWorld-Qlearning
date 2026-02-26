# Tabular Q-learning agent for GridWorld.
# API:
#   agent = QAgent(n_states=50, n_actions=5, ...)
#   a = agent.select_action(s)
#   agent.update(s, a, r, s_next, done)
#   agent.decay_epsilon()

import random
from typing import List, Optional

class QAgent:
    def __init__(self,
                 n_states: int,
                 n_actions: int = 5,
                 alpha: float = 0.2,
                 gamma: float = 0.99,
                 eps_start: float = 1.0,
                 eps_end: float = 0.05,
                 eps_decay: float = 0.995,
                 seed: Optional[int] = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = random.Random(seed)

        # Q-table initialized to zeros
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def select_action(self, state: int) -> int:
        # epsilon-greedy exploration
        if self.rng.random() < self.eps:
            return self.rng.randrange(self.n_actions)
        return self._argmax_index(self.Q[state])

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        qsa = self.Q[s][a]
        best_next = 0.0 if done else max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        self.Q[s][a] = qsa + self.alpha * (td_target - qsa)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    # Helper: break argmax ties randomly
    def _argmax_index(self, arr: List[float]) -> int:
        max_val = max(arr)
        cands = [i for i, v in enumerate(arr) if v == max_val]
        return self.rng.choice(cands)
