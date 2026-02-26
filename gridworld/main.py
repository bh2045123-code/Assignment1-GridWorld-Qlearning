# Entry point to train Q-learning on GridWorld and run a greedy rollout.

from gridworld.env import GridWorldEnv
from gridworld.agent import QAgent
from gridworld.trainer import Trainer

def main():
    env = GridWorldEnv(
        width=5, height=5,
        start=(0, 4), goal=(4, 0), treasure=(2, 2),
        walls={(1, 1), (1, 2), (3, 3)},
        traps={(2, 1), (3, 1)},
        shaping=True,
        max_steps=200
    )
    n_states = 2 * env.W * env.H  # 50 for 5x5
    agent = QAgent(
        n_states=n_states, n_actions=5,
        alpha=0.2, gamma=0.99,
        eps_start=1.0, eps_end=0.05, eps_decay=0.995,
        seed=42
    )
    trainer = Trainer(env, agent, episodes=1000, render_every=0)
    trainer.run()
    print(trainer.summary())

    # Greedy rollout after training
    s = env.reset()
    done = False
    steps = 0
    total_r = 0.0
    info = {}
    while not done and steps < 100:
        qrow = agent.Q[s]
        a = max(range(len(qrow)), key=lambda i: qrow[i])
        s, r, done, info = env.step(a)
        total_r += r
        steps += 1
    print(f"Greedy result: steps={steps}, return={total_r:.2f}, success={'success' in info}")

if __name__ == "__main__":
    main()
