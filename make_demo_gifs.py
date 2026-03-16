#!/usr/bin/env python3
"""
make_demo_gifs.py — Two strategy GIFs using the full interactive renderer.

  assets/demo_prospective.gif  — td=0.0, 1 episode (long smooth approach)
  assets/demo_reactive.gif     — td=0.8, 3 episodes (tight reactive correction)

Reuses _draw() from interactive.py: full HUD, yellow band, arrow, info bar.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
import imageio
from pathlib import Path

from goal_tracking_env import GoalTrackingEnv
from interactive import _draw, _update_td, TOL_MIN, _sv_for_td

FPS  = 12
OUT  = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)


def make_gif(task_demand, n_goals, outpath, seed=0):
    tol0 = max(TOL_MIN, 5.0 * (1.0 - task_demand))
    env  = GoalTrackingEnv(
        task_demand=task_demand,
        step_velocity=_sv_for_td(tol0),
        anti_alias=False,
        seed=seed,
    )
    env.reset(seed=seed)
    env.tolerance_radius = tol0

    frames        = []
    goals_reached = 0
    flash_frames  = 0

    while goals_reached < n_goals:
        # render + collect frame
        bgr = _draw(env, goals_reached, flash_frames)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

        if flash_frames > 0:
            flash_frames -= 1
            continue                         # hold on flash, don't step

        action = env.oracle_action()
        _, _, terminated, _, _ = env.step(action)

        if terminated:
            goals_reached += 1
            flash_frames   = 10             # green GOAL! flash
            # hold flash
            for fi in range(flash_frames, 0, -1):
                bgr = _draw(env, goals_reached, fi)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            flash_frames = 0
            if goals_reached < n_goals:
                env.reset_goal()            # cursor stays, only goal moves

    env.close()
    imageio.mimsave(str(outpath), frames, fps=FPS, loop=0)
    print(f"saved {outpath}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")


# ── Prospective: td=0.0, 1 long episode ───────────────────────────────────
make_gif(
    task_demand=0.0,
    n_goals=2,
    outpath=OUT / "demo_prospective.gif",
    seed=7,
)

# ── Reactive: td=0.8, 3 goals, cursor never resets ────────────────────────
make_gif(
    task_demand=0.8,
    n_goals=3,
    outpath=OUT / "demo_reactive.gif",
    seed=7,
)
