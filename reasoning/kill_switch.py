"""Phase 7: Kill-switch supervisor — stateful threshold monitor with async wrapper.

Architecture:

    KillSignal — immutable dataclass carrying condition name, observed value,
        threshold, and timestamp.

    KillSwitch — synchronous stateful supervisor.
        check_block(**signals) → list[KillSignal]
            Evaluates up to 5 conditions per call; returns newly fired signals.
            Sets is_halted = True on first violation and keeps it until reset().
        is_halted          — True once any condition has fired
        halt_reason()      — reason string from the first fired signal, or None
        get_active_signals() — all signals fired since last reset
        reset()            — clear halted flag and all tracked state

    KillSwitchMonitor — non-blocking async wrapper (background thread).
        report(**kwargs)       — submit a signal dict; returns immediately
        wait_processed(t)      — block until all queued reports are processed
        flush()                — drain the queue synchronously (join)
        is_halted              — delegates to inner KillSwitch
        reset()                — delegates; thread-safe
        start() / stop()       — lifecycle

Five kill conditions (all strict >):
    ood          ood_score        > cfg.ood_threshold
    epistemic    epistemic_var    > cfg.epistemic_threshold
    drawdown     (peak−value)/peak > cfg.max_drawdown_frac  (tracks rolling peak)
    rug_rate     rug events/window > cfg.rug_rate_threshold  (rolling window)
    lp_depth_drop (prev−curr)/prev > cfg.lp_depth_drop_threshold (consecutive blocks)
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .config import KillSwitchConfig


# ── Signal ────────────────────────────────────────────────────────────────────

@dataclass
class KillSignal:
    reason: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


# ── Synchronous supervisor ────────────────────────────────────────────────────

class KillSwitch:
    """Stateful, single-threaded kill-switch supervisor.

    Thread safety: NOT thread-safe.  Use KillSwitchMonitor for multi-threaded use.
    """

    def __init__(self, cfg: KillSwitchConfig, event_window: int = 100) -> None:
        self.cfg = cfg
        self.event_window = event_window
        self._halted: bool = False
        self._active_signals: list[KillSignal] = []
        self._peak_value: Optional[float] = None
        self._prev_lp_depth: Optional[float] = None
        self._rug_window: deque[bool] = deque(maxlen=event_window)

    # ── main interface ────────────────────────────────────────────────────────

    def check_block(
        self,
        *,
        ood_score: Optional[float] = None,
        epistemic_var: Optional[float] = None,
        portfolio_value: Optional[float] = None,
        event_type: Optional[str] = None,
        lp_depth: Optional[float] = None,
    ) -> list[KillSignal]:
        """Evaluate all supplied signals; return any newly fired KillSignals."""
        now = time.time()
        new_signals: list[KillSignal] = []

        if ood_score is not None and ood_score > self.cfg.ood_threshold:
            new_signals.append(
                KillSignal("ood", ood_score, self.cfg.ood_threshold, now)
            )

        if epistemic_var is not None and epistemic_var > self.cfg.epistemic_threshold:
            new_signals.append(
                KillSignal("epistemic", epistemic_var, self.cfg.epistemic_threshold, now)
            )

        if portfolio_value is not None:
            if self._peak_value is None or portfolio_value > self._peak_value:
                self._peak_value = portfolio_value
            elif self._peak_value > 0:
                drawdown = (self._peak_value - portfolio_value) / self._peak_value
                if drawdown > self.cfg.max_drawdown_frac:
                    new_signals.append(
                        KillSignal("drawdown", drawdown, self.cfg.max_drawdown_frac, now)
                    )

        if event_type is not None:
            self._rug_window.append(event_type == "rug_pull")
            rug_rate = sum(self._rug_window) / len(self._rug_window)
            if rug_rate > self.cfg.rug_rate_threshold:
                new_signals.append(
                    KillSignal("rug_rate", rug_rate, self.cfg.rug_rate_threshold, now)
                )

        if lp_depth is not None:
            if self._prev_lp_depth is not None and self._prev_lp_depth > 0:
                drop_frac = (self._prev_lp_depth - lp_depth) / self._prev_lp_depth
                if drop_frac > self.cfg.lp_depth_drop_threshold:
                    new_signals.append(
                        KillSignal(
                            "lp_depth_drop", drop_frac, self.cfg.lp_depth_drop_threshold, now
                        )
                    )
            self._prev_lp_depth = lp_depth

        if new_signals:
            self._halted = True
            self._active_signals.extend(new_signals)

        return new_signals

    # ── state accessors ───────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    def halt_reason(self) -> Optional[str]:
        """Reason string of the first fired signal, or None if not halted."""
        if self._active_signals:
            return self._active_signals[0].reason
        return None

    def get_active_signals(self) -> list[KillSignal]:
        """Return a snapshot of all signals fired since last reset."""
        return list(self._active_signals)

    def reset(self) -> None:
        """Clear halted state and all tracked history."""
        self._halted = False
        self._active_signals.clear()
        self._peak_value = None
        self._prev_lp_depth = None
        self._rug_window.clear()


# ── Async monitor wrapper ─────────────────────────────────────────────────────

class KillSwitchMonitor:
    """Non-blocking async wrapper around KillSwitch.

    Thread model: one background worker processes reports from a FIFO queue.
    report() returns immediately; wait_processed() blocks until the queue drains.
    """

    def __init__(self, cfg: KillSwitchConfig, event_window: int = 100) -> None:
        self._ks = KillSwitch(cfg, event_window)
        self._queue: queue.Queue[Optional[dict]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._pending_count = 0
        self._idle_event = threading.Event()
        self._idle_event.set()  # no pending items initially

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ks-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)  # sentinel wakes the worker
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._ks.check_block(**item)
            finally:
                self._queue.task_done()
                with self._lock:
                    self._pending_count -= 1
                    if self._pending_count == 0:
                        self._idle_event.set()

    # ── public interface ──────────────────────────────────────────────────────

    def report(self, **kwargs: float) -> None:
        """Submit a signal dict for background processing; returns immediately."""
        with self._lock:
            self._pending_count += 1
            self._idle_event.clear()
        self._queue.put(kwargs)

    def wait_processed(self, timeout: float = 5.0) -> bool:
        """Block until all submitted reports have been processed.

        Returns True if idle before timeout, False otherwise.
        """
        return self._idle_event.wait(timeout=timeout)

    def flush(self) -> None:
        """Synchronously drain the queue (blocks until all items processed)."""
        self._queue.join()
        with self._lock:
            if self._pending_count == 0:
                self._idle_event.set()

    @property
    def is_halted(self) -> bool:
        return self._ks.is_halted

    def halt_reason(self) -> Optional[str]:
        return self._ks.halt_reason()

    def get_active_signals(self) -> list[KillSignal]:
        return self._ks.get_active_signals()

    def reset(self) -> None:
        with self._lock:
            self._ks.reset()


# ── Factories ─────────────────────────────────────────────────────────────────

def build_kill_switch(cfg: KillSwitchConfig, event_window: int = 100) -> KillSwitch:
    return KillSwitch(cfg, event_window)


def build_monitor(
    cfg: KillSwitchConfig, event_window: int = 100
) -> KillSwitchMonitor:
    mon = KillSwitchMonitor(cfg, event_window)
    mon.start()
    return mon
