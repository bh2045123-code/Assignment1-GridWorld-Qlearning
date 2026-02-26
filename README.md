# GridWorld Q-learning (Assignment 1)

A minimal 5x5 GridWorld where the agent must pick up Treasure (T) before reaching the Goal (G). Trained with tabular Q-learning. Includes a Colab notebook and a Pygame visual demo.

## Run (Colab)
- Open `colab_notebook.ipynb`
- Run cells in order: Install → Train → Plot → Greedy Demo → Shaping Comparison
- The notebook shows learning curves, a greedy episode printout, and shaping on/off comparison plots

## Run (Local)
- Create and activate a virtual environment
  - Windows: `py -m venv venv` → `venv\Scripts\Activate.ps1`
  - macOS/Linux: `python3 -m venv venv` → `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Train summary: `python main.py`

## Visual Demo (Pygame)
- Run: `python pygame_demo.py`
- Keys: SPACE (step), A (autoplay), R (reset), Q/ESC (quit)
- Demo video (10–20s): https://youtube.com/shorts/R3DrcZ04oGY?feature=share

## Notes
- Core code: `gridworld/{env,agent,trainer}.py`
- Visualizer: `pygame_demo.py`
- If Colab “Save to GitHub” fails, download the `.ipynb` and upload via GitHub web (Add file → Upload files)
