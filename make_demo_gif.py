#!/usr/bin/env python3
"""
make_demo_gif.py — Demo GIF: high task demand → low task demand.

Two strategies side by side in time:
  Part 1: task_demand=0.8 — 3 short episodes (reactive, tight correction)
  Part 2: task_demand=0.0 — 1 episode (prospective, smooth long approach)

Output: assets/demo.gif  (256×256 upscaled, 15 fps, auto-loop)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import imageio
from pathlib import Path
from goal_tracking_env import GoalTrackingEnv

SCALE   = 4          # 64 → 256 px
FPS     = 12
OUT     = Path(__file__).parent / "assets" / "demo.gif"
OUT.parent.mkdir(exist_ok=True)

def add_label(frame, text, color=(255, 255, 255)):
    """Burn a small label into the top-left corner of an upscaled frame."""
    img = frame.copy()
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)          # shadow
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color,    1, cv2.LINE_AA)
    return img

def upscale(rgb64):
    """Nearest-neighbour upscale to keep pixel art look."""
    return cv2.resize(rgb64, (64 * SCALE, 64 * SCALE),
                      interpolation=cv2.INTER_NEAREST)

def force_opposite_goal(env):
    """Reset goal to opposite side from cursor (rejection sample, max 20 tries)."""
    cursor = env._cursor_px
    for _ in range(20):
        env.reset_goal()
        goal = env._goal_px
        # opposite side = goal and cursor on different sides of center (32)
        if (goal - 32) * (cursor - 32) < 0:
            return
    # fallback: accept whatever reset_goal gave us


def run_goals(task_demand, n_goals, label, color, seed=0):
    env = GoalTrackingEnv(task_demand=task_demand, seed=seed, anti_alias=False)
    env.reset(seed=seed)
    frames        = []
    goals_reached = 0

    while goals_reached < n_goals:
        raw = env.render()
        up  = upscale(raw)
        frames.append(add_label(up, label, color))
        action = env.oracle_action()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            # hold 8 frames on goal reached
            for _ in range(8):
                frames.append(add_label(up, label, color))
            goals_reached += 1
            if goals_reached < n_goals:
                force_opposite_goal(env)   # cursor stays, goal flips side
    env.close()
    return frames

# ── collect frames ─────────────────────────────────────────────────────────
frames  = run_goals(0.8, n_goals=3,
                    label="td = 0.8  (reactive)", color=(200, 80, 80))
frames += run_goals(0.0, n_goals=3,
                    label="td = 0.0  (prospective)", color=(80, 140, 220))

# ── save ───────────────────────────────────────────────────────────────────
imageio.mimsave(str(OUT), frames, fps=FPS, loop=0)
print(f"saved {OUT}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
