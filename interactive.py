"""
interactive.py — play GoalTrackingEnv with the keyboard.

controls:
  A / D     move cursor left / right
  + / =     raise task_demand  (tighter arrival, shorter reach)
  -         lower task_demand  (looser arrival, longer reach)
  R         reset episode
  Q / ESC   quit

the two parameters you can feel:
  arrival zone  — how big the yellow landing band is  (r_max * (1-td))
  to goal       — how far the cursor needs to travel  (updates live as you move)
usage:
    python env/interactive.py
    python env/interactive.py --td 0.5
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from env.goal_tracking_env import GoalTrackingEnv

SCALE   = 6
HUD_H   = 42
INFO_H  = 52     # two rows: arrival+goals / to-goal+step
BAR_H   = 28
TD_STEP = 0.05


B_HI  = 24
B_MID = 11

TOL_MIN = 0.5   # minimum tolerance in interactive so td=1 is still reachable


def _sv_for_td(tol):
    """Step velocity matched to tolerance so precise landing is always possible."""
    return max(0.5, min(3.0, tol))


def _update_td(env, new_td):
    """apply a new task_demand and update all derived parameters in-place."""
    new_td = float(np.clip(round(new_td, 10), 0.0, 1.0))
    env.task_demand      = new_td
    env.tolerance_radius = max(TOL_MIN, env.r_max * (1.0 - new_td))
    env.goal_distance    = env.d_max * (1.0 - new_td)
    env.step_velocity    = _sv_for_td(env.tolerance_radius)


def _constrained_reset_goal(env):
    """reset goal within the td band; cursor stays."""
    return env.reset_goal()


def _draw(env, goals_reached=0, flash_frames=0):
    raw   = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)  # render() returns RGB; imshow expects BGR
    W     = 64 * SCALE
    H     = 64 * SCALE
    frame = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)

    c_px = env._cursor_px
    g_px = env._goal_px
    tol  = env.tolerance_radius
    dist = abs(c_px - g_px)
    y_mid = H // 2

    # yellow tolerance band
    if tol > 0:
        x_lo = int(round((g_px - tol) * SCALE + SCALE // 2))
        x_hi = int(round((g_px + tol) * SCALE + SCALE // 2))
        x_lo, x_hi = max(0, x_lo), min(W, x_hi)
        if x_hi > x_lo:
            ov = frame.copy()
            cv2.rectangle(ov, (x_lo, 0), (x_hi, H), (0, 220, 255), -1)  # BGR yellow
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)

    # dashed boundary lines
    for side in [-1, +1]:
        x_t = int(round((g_px + side * tol) * SCALE + SCALE // 2))
        x_t = max(0, min(W - 1, x_t))
        y = 0
        while y < H:
            cv2.line(frame, (x_t, y), (x_t, min(y + 8, H)),
                     (0, 220, 255), 1, cv2.LINE_AA)  # BGR yellow
            y += 13

    # cyan arrow cursor -> goal
    x_c = int(round(c_px * SCALE + SCALE // 2))
    x_g = int(round(g_px * SCALE + SCALE // 2))
    if abs(x_c - x_g) > 2:
        tip = max(0.05, 8.0 / max(abs(x_c - x_g), 1))
        cv2.arrowedLine(frame, (x_c, y_mid), (x_g, y_mid),
                        (255, 220, 0), 2, cv2.LINE_AA, tipLength=tip)  # BGR cyan

    # GOAL! flash
    if flash_frames > 0:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (W - 1, H - 1), (0, 200, 70), -1)
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 220, 80), 7)
        cv2.putText(frame, "GOAL!", (W // 2 - 55, H // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 80), 3, cv2.LINE_AA)

    # ── HUD top: task demand ─────────────────────────────────────────────────
    hud = np.zeros((HUD_H, W, 3), dtype=np.uint8)
    cv2.putText(hud, "task demand: {:.2f}".format(env.task_demand),
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                (200, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, "+  /  -  to adjust",
                (W - 175, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (130, 130, 130), 1, cv2.LINE_AA)

    # ── info bar (two rows) ──────────────────────────────────────────────────
    info = np.zeros((INFO_H, W, 3), dtype=np.uint8)
    # row 1: arrival zone (left)  |  goals reached (right)
    cv2.putText(info, "arrival: +/-{:.1f} px".format(tol),
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (80, 220, 255), 1, cv2.LINE_AA)  # BGR yellow
    cv2.putText(info, "goals: {}".format(goals_reached),
                (W - 88, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (160, 255, 160), 1, cv2.LINE_AA)
    # row 2: to goal (left)  |  step size (right)
    cv2.putText(info, "to goal: {:.0f} px".format(dist),
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (255, 220, 80), 1, cv2.LINE_AA)  # BGR cyan
    cv2.putText(info, "step: {:.1f} px".format(env.step_velocity),
                (W - 105, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (160, 160, 160), 1, cv2.LINE_AA)

    # ── key guide ────────────────────────────────────────────────────────────
    bar = np.zeros((BAR_H, W, 3), dtype=np.uint8)
    cv2.putText(bar, "A / D : move     R : reset     Q : quit",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (100, 100, 100), 1, cv2.LINE_AA)

    return np.vstack([hud, frame, info, bar])


def run_interactive(task_demand=0.2, seed=None, anti_alias=False):
    tol0 = max(TOL_MIN, 5.0 * (1.0 - task_demand))
    env  = GoalTrackingEnv(task_demand=task_demand,
                           step_velocity=_sv_for_td(tol0), anti_alias=anti_alias, seed=seed)
    env.reset(seed=seed)
    env.tolerance_radius = tol0   # apply floor immediately

    goals_reached = 0
    flash_frames  = 0

    cv2.namedWindow("GoalTrackingEnv", cv2.WINDOW_AUTOSIZE)

    while True:
        cv2.imshow("GoalTrackingEnv", _draw(env, goals_reached, flash_frames))
        key = cv2.waitKey(40) & 0xFF   # ~25 fps

        terminated = False

        if key in (ord('q'), 27):                  # Q or ESC
            break
        elif key == ord('a'):                       # left
            _, _, terminated, _, _ = env.step(
                np.array([-env.step_velocity], dtype=np.float32))
        elif key == ord('d'):                       # right
            _, _, terminated, _, _ = env.step(
                np.array([+env.step_velocity], dtype=np.float32))
        elif key in (ord('+'), ord('=')):           # harder
            _update_td(env, env.task_demand + TD_STEP)
        elif key == ord('-'):                       # easier
            _update_td(env, env.task_demand - TD_STEP)
        elif key == ord('r'):                       # full reset
            td_keep = env.task_demand
            env = GoalTrackingEnv(
                task_demand=td_keep,
                d_max=env.d_max, r_max=env.r_max,
                step_velocity=_sv_for_td(env.r_max * (1.0 - td_keep)),
                anti_alias=anti_alias, seed=seed)
            env.reset()
            goals_reached = 0
            flash_frames  = 0

        if terminated:
            goals_reached += 1
            flash_frames   = 15
            _constrained_reset_goal(env)

        if flash_frames > 0:
            flash_frames -= 1

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="interactive GoalTrackingEnv")
    p.add_argument("--td",         type=float, default=0.2,  help="initial task_demand [0,1]")
    p.add_argument("--seed",       type=int,   default=None, help="RNG seed")
    p.add_argument("--anti-alias", action="store_true",      help="smooth circle rendering (default: sharp pixels)")
    args = p.parse_args()
    run_interactive(task_demand=args.td, seed=args.seed, anti_alias=args.anti_alias)
