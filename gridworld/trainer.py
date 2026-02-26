# Training loop with logging for GridWorld + QAgent.

from typing import Dict, Any, List
from gridworld.env import GridWorldEnv
from gridworld.agent import QAgent

class Trainer:
    def __init__(self,
                 env: GridWorldEnv,
                 agent: QAgent,
                 episodes: int = 1000,
                 render_every: int = 0):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.render_every = render_every

        self.ep_returns: List[float] = []
        self.ep_lengths: List[int] = []
        self.ep_success: List[int] = []

    def run(self):
        area = self.env.W * self.env.H
        assert self.agent.n_states == 2 * area, "Agent n_states must equal 2*W*H (carry bit)."

        for ep in range(1, self.episodes + 1):
            s = self.env.reset()
            done = False
            total_r = 0.0
            steps = 0
            success = 0
            info: Dict[str, Any] = {}

            while not done:
                a = self.agent.select_action(s)
                s_next, r, done, info = self.env.step(a)
                self.agent.update(s, a, r, s_next, done)
                s = s_next
                total_r += r
                steps += 1

            if "success" in info:
                success = 1

            self.agent.decay_epsilon()
            self.ep_returns.append(total_r)
            self.ep_lengths.append(steps)
            self.ep_success.append(success)

            if self.render_every and (ep % self.render_every == 0):
                print(f"\nEpisode {ep} | R={total_r:.2f} | steps={steps} | success={success} | eps={self.agent.eps:.3f}")
                print(self.env.render_text())

    def summary(self) -> Dict[str, Any]:
        import statistics as stats
        n = len(self.ep_returns)
        succ_rate = sum(self.ep_success) / max(1, n)
        return {
            "episodes": n,
            "avg_return": stats.mean(self.ep_returns) if n else 0.0,
            "avg_length": stats.mean(self.ep_lengths) if n else 0.0,
            "success_rate": succ_rate,
            "epsilon_final": self.agent.eps,
        }
