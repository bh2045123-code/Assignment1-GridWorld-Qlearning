Here’s the same README content in English. You can paste it directly into README.md.

---
# Assignment1-GridWorld-Qlearning

Project: 5x5 GridWorld Treasure Hunt (Q-learning)

Overview
- A 5x5 grid-based treasure hunt using Q-learning. The agent starts at a fixed position, must first pick up the treasure, then return to the goal. We train a tabular Q-learning agent and visualize the results.
- This repo includes: Environment (gridworld/env.py), Agent (gridworld/agent.py), Trainer (gridworld/trainer.py), a Colab notebook (colab_notebook.ipynb), and a local Pygame demo (pygame_demo.py).

Directory Layout
- README.md
- requirements.txt
- colab_notebook.ipynb
- gridworld/
  - __init__.py
  - env.py
  - agent.py
  - trainer.py
  - utils_plot.py
- pygame_demo.py
- report.docx or report.pdf
- saved_models/
- experiments/

Quick Start
1) Run on Colab (recommended for TAs/graders)
- Open colab_notebook.ipynb (or use GitHub → Open in Colab).
- Run all cells (installs dependencies, trains the agent, plots learning curves).
- Pros: no local setup, easy to reproduce and grade.

2) Run locally (for Pygame UI and recording the demo video)
- Clone the repo:
  git clone <your-repo-url>
  cd Assignment1-GridWorld-Qlearning
- Create a virtual env and install deps:
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate     # Windows
  pip install -r requirements.txt
- Train (text/Matplotlib):
  python main.py
- Pygame demo (for screen-recorded video):
  python pygame_demo.py

Task Rules (brief)
- 5x5 grid; start (0,4), goal (4,0), treasure (2,2); walls {(1,1),(1,2),(3,3)}; traps {(2,1),(3,1)}.
- Must pick up the treasure first, then reach the goal to succeed; boundary/wall hit keeps position with small penalty; stepping on a trap gives a penalty.
- Episode limit: 200 steps.

Q-learning Summary
- State: s = (x, y, has_treasure), 50 discrete states; Actions: up/right/down/left/stay (5 actions).
- Rewards: treasure +10, treasure-to-goal +20; trap -10; wall/boundary -1; step cost -0.1; optional potential-based shaping.
- Policy: epsilon-greedy; Update: Q(s,a) ← Q + α[r + γ max_a' Q(s', a') − Q].

Default Hyperparameters
- alpha=0.2, gamma=0.99
- eps_start=1.0, eps_end=0.05, eps_decay=0.995
- episodes=1000, max_steps=200, shaping=True

Repro and Submission
- TAs can run the Colab notebook to reproduce learning curves and a sample rollout.
- Locally, run pygame_demo.py and record a short demo video; add the YouTube (unlisted) link to the README or README_VIDEO.txt.

Contents and License
- Ensure the repo includes source code, the notebook, experiment figures, the report, and the video link. License can be MIT or omitted.

Support
- If you encounter errors, please open an Issue with logs and steps to reproduce.
---

Let me know once you’ve pasted and committed it. I’ll proceed to Step 2 (add gridworld/env.py) and tell you exactly where to place it.
