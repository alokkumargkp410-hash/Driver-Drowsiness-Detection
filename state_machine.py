"""
state_machine.py - Temporal sliding-window fatigue decision engine.

Consumes per-frame binary observations (eye closed? yawning? head off-axis? face
present?) - which are themselves produced from *personalised* thresholds - and
evaluates a 60-second sliding window:

    N_b(t) : blinks            closure episodes of  blink_min .. micro_sleep_min seconds
    N_s(t) : micro-sleeps      closures lasting >= micro_sleep_min seconds
    N_y(t) : yawns             mouth-open episodes lasting >= yawn_min seconds
    PERCLOS(t) = closed time / window time (Eq. 2 of the paper, time-weighted)

Alert levels
    0  NORMAL
    1  WARNING                 any metric breaches its warning limit
    2  CRITICAL DROWSINESS     a critical limit, or >= critical_breach_count
                               simultaneous warning-level breaches

The level rises immediately but only falls after `alert_clear_s` of continuous
improvement (hysteresis), which prevents flickering banners.

All timestamps are supplied by the caller, so the engine behaves identically for live
cameras (wall-clock) and video files (frame_index / fps).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from config import StateConfig

LEVEL_LABELS = {0: "NORMAL", 1: "WARNING", 2: "CRITICAL DROWSINESS ALERT"}


# --------------------------------------------------------------------------- #
# Smoothing helper
# --------------------------------------------------------------------------- #
class RollingSmoother:
    """Fixed-length buffer with median or mean output (NaNs are ignored)."""

    def __init__(self, length: int, use_median: bool) -> None:
        self._buf: Deque[float] = deque(maxlen=max(1, length))
        self._median = use_median

    def push(self, value: float) -> float:
        if np.isfinite(value):
            self._buf.append(float(value))
        return self.value

    @property
    def value(self) -> float:
        if not self._buf:
            return float("nan")
        arr = np.fromiter(self._buf, dtype=np.float64)
        return float(np.median(arr) if self._median else arr.mean())

    def clear(self) -> None:
        self._buf.clear()


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameObservation:
    t: float                    # seconds
    face_found: bool
    eye_closed: bool
    yawning: bool
    head_off_axis: bool = False


@dataclass(frozen=True)
class WindowMetrics:
    window_span_s: float = 0.0
    blink_count: int = 0                # N_b(t)
    microsleep_count: int = 0           # N_s(t)
    yawn_count: int = 0                 # N_y(t)
    perclos: float = 0.0
    closure_run_s: float = 0.0          # current continuous closure
    yawn_run_s: float = 0.0
    head_off_run_s: float = 0.0
    face_lost_s: float = 0.0


@dataclass(frozen=True)
class AlertStatus:
    level: int
    label: str
    reasons: Tuple[str, ...]
    metrics: WindowMetrics


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class DrowsinessStateMachine:
    def __init__(self, cfg: StateConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._events: Deque[Tuple[float, str]] = deque()           # (t, kind)
        self._samples: Deque[Tuple[float, float, bool]] = deque()  # (t, dt, closed)
        self._closed_time = 0.0
        self._total_time = 0.0
        self._first_t: Optional[float] = None
        self._last_t: Optional[float] = None

        self._closure_start: Optional[float] = None
        self._closure_is_microsleep = False
        self._yawn_start: Optional[float] = None
        self._yawn_counted = False
        self._last_yawn_t = -1e9
        self._head_start: Optional[float] = None
        self._lost_since: Optional[float] = None

        self._level = 0
        self._last_active_t = 0.0
        self._last_reasons: Tuple[str, ...] = ()

    # ----------------------------------------------------------------- main -- #
    def update(self, obs: FrameObservation) -> AlertStatus:
        t = obs.t
        cfg = self.cfg
        if self._first_t is None:
            self._first_t = t
        dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, cfg.max_frame_gap_s))
        self._last_t = t

        if obs.face_found:
            self._lost_since = None
            self._add_sample(t, dt, obs.eye_closed)
            self._track_closure(t, obs.eye_closed)
            self._track_yawn(t, obs.yawning)
            self._track_head(t, obs.head_off_axis)
        else:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since > cfg.face_loss_grace_s:
                # Drop open episodes without counting them: the evidence is unknown.
                self._closure_start = None
                self._closure_is_microsleep = False
                self._yawn_start = None
                self._yawn_counted = False
                self._head_start = None

        self._prune(t)
        metrics = self._metrics(t)
        return self._evaluate(t, metrics)

    # ------------------------------------------------------------- episodes -- #
    def _add_sample(self, t: float, dt: float, closed: bool) -> None:
        self._samples.append((t, dt, closed))
        self._total_time += dt
        if closed:
            self._closed_time += dt

    def _track_closure(self, t: float, closed: bool) -> None:
        cfg = self.cfg
        if closed:
            if self._closure_start is None:
                self._closure_start = t
                self._closure_is_microsleep = False
            if (not self._closure_is_microsleep
                    and t - self._closure_start >= cfg.micro_sleep_min_duration_s):
                self._closure_is_microsleep = True
                self._events.append((t, "microsleep"))
        else:
            if self._closure_start is not None:
                duration = t - self._closure_start
                if not self._closure_is_microsleep and duration >= cfg.blink_min_duration_s:
                    self._events.append((t, "blink"))
            self._closure_start = None
            self._closure_is_microsleep = False

    def _track_yawn(self, t: float, yawning: bool) -> None:
        cfg = self.cfg
        if yawning:
            if self._yawn_start is None:
                self._yawn_start = t
                self._yawn_counted = False
            if (not self._yawn_counted and t - self._yawn_start >= cfg.yawn_min_duration_s
                    and t - self._last_yawn_t >= cfg.yawn_refractory_s):
                self._yawn_counted = True
                self._last_yawn_t = t
                self._events.append((t, "yawn"))
        else:
            self._yawn_start = None
            self._yawn_counted = False

    def _track_head(self, t: float, off_axis: bool) -> None:
        if off_axis:
            if self._head_start is None:
                self._head_start = t
        else:
            self._head_start = None

    # -------------------------------------------------------------- windows -- #
    def _prune(self, t: float) -> None:
        horizon = t - self.cfg.window_seconds
        while self._events and self._events[0][0] < horizon:
            self._events.popleft()
        while self._samples and self._samples[0][0] < horizon:
            _, dt, closed = self._samples.popleft()
            self._total_time -= dt
            if closed:
                self._closed_time -= dt
        if not self._samples:
            self._total_time = 0.0
            self._closed_time = 0.0

    def _count(self, kind: str) -> int:
        return sum(1 for _, k in self._events if k == kind)

    def _metrics(self, t: float) -> WindowMetrics:
        span = 0.0 if self._first_t is None else min(t - self._first_t, self.cfg.window_seconds)
        perclos = (self._closed_time / self._total_time) if self._total_time > 1e-9 else 0.0
        return WindowMetrics(
            window_span_s=span,
            blink_count=self._count("blink"),
            microsleep_count=self._count("microsleep"),
            yawn_count=self._count("yawn"),
            perclos=float(min(max(perclos, 0.0), 1.0)),
            closure_run_s=0.0 if self._closure_start is None else t - self._closure_start,
            yawn_run_s=0.0 if self._yawn_start is None else t - self._yawn_start,
            head_off_run_s=0.0 if self._head_start is None else t - self._head_start,
            face_lost_s=0.0 if self._lost_since is None else t - self._lost_since,
        )

    # ------------------------------------------------------------- alerting -- #
    def _evaluate(self, t: float, m: WindowMetrics) -> AlertStatus:
        c = self.cfg
        warn: List[str] = []
        crit: List[str] = []
        groups: set = set()      # modalities in breach: eye / mouth / head

        perclos_ok = m.window_span_s >= c.perclos_min_span_s
        if perclos_ok and m.perclos >= c.warn_perclos:
            warn.append(f"PERCLOS {m.perclos:.0%}")
            groups.add("eye")
        if perclos_ok and m.perclos >= c.critical_perclos:
            crit.append(f"PERCLOS {m.perclos:.0%}")
        if m.microsleep_count >= c.warn_microsleeps:
            warn.append(f"MICRO-SLEEP x{m.microsleep_count}")
            groups.add("eye")
        if m.microsleep_count >= c.critical_microsleeps:
            crit.append(f"MICRO-SLEEPS x{m.microsleep_count}")
        if m.yawn_count >= c.warn_yawns:
            warn.append(f"YAWNING x{m.yawn_count}")
            groups.add("mouth")
        if m.yawn_count >= c.critical_yawns:
            crit.append(f"YAWNING x{m.yawn_count}")
        if m.blink_count >= c.warn_blinks_per_window:
            warn.append(f"HIGH BLINK RATE ({m.blink_count}/min)")
            groups.add("eye")
        if m.head_off_run_s >= c.head_off_axis_min_duration_s:
            warn.append("HEAD NOD / OFF-AXIS")
            groups.add("head")
        if m.face_lost_s >= c.face_lost_alert_s:
            warn.append("DRIVER NOT VISIBLE")
        if m.closure_run_s >= c.critical_closure_s:
            warn.append(f"EYES CLOSED {m.closure_run_s:.1f}s")
            crit.append(f"EYES CLOSED {m.closure_run_s:.1f}s")
            groups.add("eye")
        elif m.closure_run_s >= c.micro_sleep_min_duration_s:
            warn.append(f"EYES CLOSED {m.closure_run_s:.1f}s")
            groups.add("eye")

        if crit or len(groups) >= c.critical_breach_count:
            raw = 2
            reasons = tuple(crit) if crit else tuple(warn)
        elif warn:
            raw, reasons = 1, tuple(warn)
        else:
            raw, reasons = 0, ()

        if raw >= self._level:
            self._level = raw
            self._last_active_t = t
            self._last_reasons = reasons
        elif t - self._last_active_t >= c.alert_clear_s:
            self._level = raw
            self._last_reasons = reasons
            self._last_active_t = t
        return AlertStatus(self._level, LEVEL_LABELS[self._level], self._last_reasons, m)
