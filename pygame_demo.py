# Minimal Pygame visualizer for GridWorld.
# Controls:
#   SPACE: step one action using greedy policy after a short warmup training
#   A: autoplay toggle
#   R: reset episode
#   ESC or Q: quit

import pygame
import sys
import time
from gridworld.env import GridWorldEnv
from gridworld.agent import QAgent
from gridworld.trainer import Trainer

W, H = 640, 640
GRID = 5
CELL = W // GRID
MARGIN = 2
FPS = 30

COL_BG = (30, 30, 30)
COL_GRID = (200, 200, 200)
COL_WALL = (80, 80, 80)
COL_TRAP = (180, 60, 60)
COL_TREASURE = (230, 200, 40)
COL_GOAL = (70, 160, 90)
COL_START = (90, 140, 220)
COL_AGENT = (240, 240, 240)
COL_AGENT_WITH_T = (255, 170, 60)
COL_TEXT = (230, 230, 230)

def draw_cell(screen, x, y, color, text=None, font=None):
    rect = pygame.Rect(x*CELL+MARGIN, y*CELL+MARGIN, CELL-2*MARGIN, CELL-2*MARGIN)
    pygame.draw.rect(screen, color, rect, border_radius=6)
    if text and font:
        surf = font.render(text, True, (0,0,0))
        screen.blit(surf, (rect.x + 6, rect.y + 4))

def greedy_action(Q, s):
    row = Q[s]
    best = max(range(len(row)), key=lambda i: row[i])
    return best

def warmup_train(env, episodes=400):
    # quick training to get a reasonable policy for demo
    agent = QAgent(n_states=2*env.W*env.H, n_actions=5,
                   alpha=0.2, gamma=0.99,
                   eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                   seed=123)
    Trainer(env, agent, episodes=episodes, render_every=0).run()
    return agent

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GridWorld Q-learning Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 20)

    env = GridWorldEnv(shaping=True, max_steps=200)
    agent = warmup_train(env, episodes=600)

    s = env.reset()
    done = False
    autoplay = False
    total_r = 0.0
    steps = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); sys.exit(0)
                if event.key == pygame.K_r:
                    s = env.reset(); done = False; total_r = 0.0; steps = 0
                if event.key == pygame.K_a:
                    autoplay = not autoplay
                if event.key == pygame.K_SPACE and not done:
                    a = greedy_action(agent.Q, s)
                    s, r, done, info = env.step(a)
                    total_r += r; steps += 1

        if autoplay and not done:
            a = greedy_action(agent.Q, s)
            s, r, done, info = env.step(a)
            total_r += r; steps += 1
            time.sleep(0.05)

        screen.fill(COL_BG)

        # draw grid content
        for y in range(env.H):
            for x in range(env.W):
                draw_cell(screen, x, y, COL_GRID)

        for (wx, wy) in env.walls:
            draw_cell(screen, wx, wy, COL_WALL, "#", font)
        for (tx, ty) in env.traps:
            draw_cell(screen, tx, ty, COL_TRAP, "X", font)

        gx, gy = env.goal
        draw_cell(screen, gx, gy, COL_GOAL, "G", font)
        sx, sy = env.start
        draw_cell(screen, sx, sy, COL_START, "S", font)
        tx, ty = env.treasure
        if not env.has_treasure:
            draw_cell(screen, tx, ty, COL_TREASURE, "T", font)

        ax, ay = env.pos
        draw_cell(screen, ax, ay, COL_AGENT_WITH_T if env.has_treasure else COL_AGENT, "P", font)

        # HUD
        hud_lines = [
            f"steps={steps}  total_r={total_r:.2f}  autoplay={'ON' if autoplay else 'OFF'}",
            "keys: SPACE=step  A=autoplay  R=reset  Q/ESC=quit",
        ]
        y0 = 8
        for line in hud_lines:
            surf = font.render(line, True, COL_TEXT)
            screen.blit(surf, (8, y0))
            y0 += 22

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
