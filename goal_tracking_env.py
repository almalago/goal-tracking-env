"""
GoalTrackingEnv — linear 1D cursor-to-goal tracking environment.

state space  : (cursor_px, goal_px) — integer pixel x-positions; y fixed at 32.
action space : Box(1,)  — continuous pixel delta, clipped to ±step_velocity per step.
task_demand  : scalar ∈ [0, 1]. Higher → shorter distance + tighter tolerance.
    goal_distance   = d_max × (1 - task_demand)   [pixels]
    tolerance_radius= r_max × (1 - task_demand)   [pixels]
termination  : distance_to_goal < tolerance_radius (strict). No time limit.
Reward       : 0.0 

observations: composite RGB image (3, 64, 64) float32 CHW in [0, 1].
  black background, white goal dot (radius 5 px), red cursor dot (radius 3 px).


rendering strategy
------------------
the goal dot is stable between resets, so it is pre-rendered into a cached
BGR frame (_goal_frame) whenever the goal changes.  each step clones that
frame (memcpy) and draws the cursor dot on top — identical efficiency pattern
to the dot_bank variant without the file dependency.
"""

import numpy as np
import cv2

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    import gym
    from gym import spaces

# ── rendering constants ────────────────────────────────────────────────────────
_IMG_H         = 64
_IMG_W         = 64
_CURSOR_RADIUS = 3
_GOAL_RADIUS   = 5

_CURSOR_MIN = 0
_CURSOR_MAX = _IMG_W - 1   # 63
_GOAL_MIN   = 0
_GOAL_MAX   = _IMG_W - 1   # 63

# Safe bounds: dot fully within frame (center ≥ radius+1, ≤ W-radius-2)
CURSOR_SAFE_MIN = _CURSOR_RADIUS + 1          # 4
CURSOR_SAFE_MAX = _IMG_W - _CURSOR_RADIUS - 1 # 60
GOAL_SAFE_MIN   = _GOAL_RADIUS + 1            # 6
GOAL_SAFE_MAX   = _IMG_W - _GOAL_RADIUS - 2   # 57  (conservative, matches band geometry)

_B_HI  = 24   # band boundary: join of td=0.0 and td=0.5 bands
_B_MID = 11   # band boundary: join of td=0.5 and td=1.0 bands


class GoalTrackingEnv(gym.Env):
    """
    Linear cursor-to-goal tracking with task-demand curriculum.

    Parameters
    ----------
    task_demand : float in [0, 1]
        Higher → shorter initial distance AND tighter tolerance simultaneously.
        goal_distance   = d_max × (1 - task_demand)
        tolerance_radius= r_max × (1 - task_demand)
    d_max : float
        Maximum cursor-to-goal distance in pixels (default: 28 = D_MAX_GUARANTEED).
    r_max : float
        Maximum tolerance radius in pixels (default: 5 = goal dot radius).
    step_velocity : float or None
        Pixels the cursor moves per step. None → auto from tolerance via _sv_for_td.
    velocity_mode : {"bounded", "fixed", "free"}
        "bounded" : action clipped to ±step_velocity.
        "fixed"   : always step_velocity × sign(action); recomputes from tol each step.
        "free"    : raw action passed through unclipped.
    render_mode : str or None
        "rgb_array" or "human".
    seed : int or None
        RNG seed.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    CURSOR_RADIUS: int = _CURSOR_RADIUS
    GOAL_RADIUS:   int = _GOAL_RADIUS

    CURSOR_MIN: int = _CURSOR_MIN
    CURSOR_MAX: int = _CURSOR_MAX
    GOAL_MIN:   int = _GOAL_MIN
    GOAL_MAX:   int = _GOAL_MAX

    CURSOR_SAFE_MIN: int = CURSOR_SAFE_MIN
    CURSOR_SAFE_MAX: int = CURSOR_SAFE_MAX
    GOAL_SAFE_MIN:   int = GOAL_SAFE_MIN
    GOAL_SAFE_MAX:   int = GOAL_SAFE_MAX

    # min over g∈[GOAL_SAFE_MIN, GOAL_SAFE_MAX] of max(g-CURSOR_SAFE_MIN, CURSOR_SAFE_MAX-g) = 28
    D_MAX_GUARANTEED: float = 28.0

    B_HI:  int = _B_HI
    B_MID: int = _B_MID

    def __init__(
        self,
        task_demand:   float = 0.5,
        d_max:         float = 28.0,
        r_max:         float = 5.0,
        step_velocity: float = None,       # None → auto from tolerance via _sv_for_td
        velocity_mode: str   = "bounded",  # "free" | "bounded" | "fixed"
        anti_alias:    bool  = False,      # True → cv2.LINE_AA circles (smooth); False → sharp pixels
        render_mode:   str   = None,
        seed:          int   = None,
    ):
        super().__init__()

        if not (0.0 <= task_demand <= 1.0):
            raise ValueError(f"task_demand must be in [0, 1], got {task_demand}")
        if d_max <= 0:
            raise ValueError(f"d_max must be positive, got {d_max}")
        if r_max <= 0:
            raise ValueError(f"r_max must be positive, got {r_max}")
        if velocity_mode not in ("free", "bounded", "fixed"):
            raise ValueError(f"velocity_mode must be 'free', 'bounded', or 'fixed'")

        self.task_demand   = task_demand
        self.d_max         = d_max
        self.r_max         = r_max
        self.velocity_mode = velocity_mode
        self.render_mode   = render_mode
        self._line_type    = cv2.LINE_AA if anti_alias else cv2.LINE_8

        self.goal_distance    = d_max * (1.0 - task_demand)
        self.tolerance_radius = r_max * (1.0 - task_demand)

        if step_velocity is None:
            step_velocity = self._sv_for_td(self.tolerance_radius)
        if step_velocity <= 0:
            raise ValueError(f"step_velocity must be positive, got {step_velocity}")
        self.step_velocity = step_velocity

        # ── spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, _IMG_H, _IMG_W), dtype=np.float32
        )
        _sv = float(self.CURSOR_MAX - self.CURSOR_MIN) if velocity_mode == "free" else step_velocity
        self.action_space = spaces.Box(
            low=np.array([-_sv], dtype=np.float32),
            high=np.array([+_sv], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng           = np.random.default_rng(seed)
        self._cursor_px:    float          = float(self.CURSOR_SAFE_MIN)
        self._goal_px:      float          = float(self.GOAL_SAFE_MIN)
        self._goal_frame:   np.ndarray     = self._make_goal_frame(self.GOAL_SAFE_MIN, self._line_type)

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        c_px = int(self._rng.integers(self.CURSOR_SAFE_MIN, self.CURSOR_SAFE_MAX + 1))
        g_px = self._sample_goal(c_px)
        self._cursor_px  = float(c_px)
        self._set_goal(g_px)

        return self._get_obs(), self._make_info()

    def step(self, action: np.ndarray):
        raw = float(np.asarray(action).ravel()[0])
        if self.velocity_mode == "free":
            delta = raw
        elif self.velocity_mode == "bounded":
            delta = float(np.clip(raw, -self.step_velocity, self.step_velocity))
        else:  # "fixed" — recompute each step so live td changes take effect
            self.step_velocity = self._sv_for_td(self.tolerance_radius)
            delta = self.step_velocity * float(np.sign(raw))

        self._cursor_px = float(np.clip(
            self._cursor_px + delta, self.CURSOR_MIN, self.CURSOR_MAX,
        ))

        distance   = abs(self._cursor_px - self._goal_px)
        terminated = distance < self.tolerance_radius
        return self._get_obs(), 0.0, terminated, False, self._make_info()

    def reset_goal(self, *, seed: int = None) -> tuple:
        """Place a new goal without moving the cursor. Returns (obs, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._set_goal(self._sample_goal(int(round(self._cursor_px))))
        return self._get_obs(), self._make_info()

    def oracle_action(self) -> np.ndarray:
        """Move directly toward goal at step_velocity. Returns ndarray shape (1,)."""
        delta = self._goal_px - self._cursor_px
        return np.array(
            [np.clip(delta, -self.step_velocity, self.step_velocity)],
            dtype=np.float32,
        )

    def render(self):
        c_px  = int(np.clip(round(self._cursor_px), self.CURSOR_MIN, self.CURSOR_MAX))
        frame = self._draw_cursor(c_px)  # HWC uint8 RGB

        if self.render_mode == "human":
            cv2.imshow("GoalTrackingEnv", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return frame

    def close(self):
        cv2.destroyAllWindows()

    # ── internal ──────────────────────────────────────────────────────────────

    def _set_goal(self, g_px: int) -> None:
        """update goal position and invalidate the cached goal frame."""
        self._goal_px    = float(g_px)
        self._goal_frame = self._make_goal_frame(g_px, self._line_type)

    @staticmethod
    def _make_goal_frame(g_px: int, line_type: int = cv2.LINE_8) -> np.ndarray:
        """pre-render the goal dot onto a black RGB frame. cached until goal moves."""
        frame = np.zeros((_IMG_H, _IMG_W, 3), dtype=np.uint8)
        cv2.circle(frame, (g_px, _IMG_H // 2), _GOAL_RADIUS,   (255, 255, 255), -1, line_type)
        return frame

    def _draw_cursor(self, c_px: int) -> np.ndarray:
        """clone cached goal frame and draw cursor on top. Returns HWC uint8 RGB."""
        frame = self._goal_frame.copy()  # O(H×W) memcpy; goal dot already in place
        cv2.circle(frame, (c_px, _IMG_H // 2), _CURSOR_RADIUS, (255, 0,   0),   -1, self._line_type)
        return frame  # RGB: no conversion needed

    def _get_obs(self) -> np.ndarray:
        c_px  = int(np.clip(round(self._cursor_px), self.CURSOR_MIN, self.CURSOR_MAX))
        frame = self._draw_cursor(c_px)  # HWC uint8 RGB
        return (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW float32

    @staticmethod
    def _sv_for_td(tol: float) -> float:
        """Step velocity matched to tolerance: clamped to [0.5, 3.0]."""
        return max(0.5, min(3.0, tol))

    def _sample_goal(self, c_px: int) -> int:
        """
        sample goal within the distance band for current task_demand.
        three non-overlapping bands, linearly interpolated between anchors:
          td=0.0: d ∈ [B_HI=24, d_avail]
          td=0.5: d ∈ [B_MID=11, B_HI=24]
          td=1.0: d ∈ [1,        B_MID=11]
        """
        td      = self.task_demand
        d_avail = max(c_px - self.GOAL_SAFE_MIN, self.GOAL_SAFE_MAX - c_px)

        if td <= 0.5:
            t     = td * 2.0
            d_min = int(round(self.B_HI  + (self.B_MID - self.B_HI)  * t))  # 24 → 11
            d_max = int(round(d_avail    + (self.B_HI  - d_avail)    * t))  # avail → 24
        else:
            t     = (td - 0.5) * 2.0
            d_min = int(round(self.B_MID + (1          - self.B_MID) * t))  # 11 → 1
            d_max = int(round(self.B_HI  + (self.B_MID - self.B_HI)  * t))  # 24 → 11

        d_min = max(1, d_min)
        d_max = max(d_min, min(d_max, d_avail))

        valid = [g for g in range(self.GOAL_SAFE_MIN, self.GOAL_SAFE_MAX + 1)
                 if d_min <= abs(g - c_px) <= d_max]
        if valid:
            return int(valid[self._rng.integers(len(valid))])
        options = [int(np.clip(c_px + s * d_min, self.GOAL_SAFE_MIN, self.GOAL_SAFE_MAX))
                   for s in (-1, +1)]
        options = [g for g in options if g != c_px]
        return options[0] if options else self.GOAL_SAFE_MIN

    def _make_info(self) -> dict:
        c = int(round(self._cursor_px))
        g = int(round(self._goal_px))
        return {
            "cursor_px":        c,
            "goal_px":          g,
            "distance":         abs(self._cursor_px - self._goal_px),
            "tolerance_radius": self.tolerance_radius,
            "task_demand":      self.task_demand,
            "step_velocity":    self.step_velocity,
            "steps_to_goal":    (
                int(np.ceil(abs(self._cursor_px - self._goal_px) / self.step_velocity))
                if self.step_velocity > 0 else float("inf")
            ),
        }
