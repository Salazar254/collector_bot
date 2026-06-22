"""Phase 7 tests — Kill-switch supervisor.

Coverage:
    TestKillSignal              — dataclass fields and defaults
    TestKillSwitchConditions    — each of the 5 conditions, below/above, boundary
    TestKillSwitchStateful      — halted flag, halt_reason, reset, peak tracking
    TestKillSwitchMonitor       — async wrapper lifecycle, report/wait, delegates
    TestLatency                 — check_block p99 <1ms, report non-blocking p99 <1ms
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import pytest

from reasoning.config import KillSwitchConfig
from reasoning.kill_switch import (
    KillSignal,
    KillSwitch,
    KillSwitchMonitor,
    build_kill_switch,
    build_monitor,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _cfg(**overrides) -> KillSwitchConfig:
    return KillSwitchConfig(**overrides)


def _ks(**cfg_overrides) -> KillSwitch:
    return KillSwitch(_cfg(**cfg_overrides))


# ── TestKillSignal ─────────────────────────────────────────────────────────────

class TestKillSignal:
    def test_fields_set(self):
        sig = KillSignal(reason="ood", value=4.0, threshold=3.0, timestamp=1.0)
        assert sig.reason == "ood"
        assert sig.value == 4.0
        assert sig.threshold == 3.0
        assert sig.timestamp == 1.0

    def test_default_timestamp_is_recent(self):
        before = time.time()
        sig = KillSignal(reason="test", value=1.0, threshold=0.5)
        after = time.time()
        assert before <= sig.timestamp <= after

    def test_multiple_signals_independent(self):
        s1 = KillSignal("ood", 5.0, 3.0)
        s2 = KillSignal("epistemic", 0.9, 0.5)
        assert s1.reason != s2.reason
        assert s1.value != s2.value


# ── TestKillSwitchConditions ───────────────────────────────────────────────────

class TestKillSwitchConditions:
    # ── OOD ───────────────────────────────────────────────────────────────────

    def test_ood_below_threshold_no_signal(self):
        ks = _ks(ood_threshold=3.0)
        sigs = ks.check_block(ood_score=2.9)
        assert sigs == []
        assert not ks.is_halted

    def test_ood_at_threshold_no_signal(self):
        ks = _ks(ood_threshold=3.0)
        sigs = ks.check_block(ood_score=3.0)
        assert sigs == []

    def test_ood_above_threshold_fires(self):
        ks = _ks(ood_threshold=3.0)
        sigs = ks.check_block(ood_score=3.001)
        assert len(sigs) == 1
        assert sigs[0].reason == "ood"
        assert sigs[0].value == pytest.approx(3.001)
        assert sigs[0].threshold == 3.0

    # ── Epistemic ──────────────────────────────────────────────────────────────

    def test_epistemic_below_no_signal(self):
        ks = _ks(epistemic_threshold=0.5)
        assert ks.check_block(epistemic_var=0.4) == []

    def test_epistemic_at_threshold_no_signal(self):
        ks = _ks(epistemic_threshold=0.5)
        assert ks.check_block(epistemic_var=0.5) == []

    def test_epistemic_above_fires(self):
        ks = _ks(epistemic_threshold=0.5)
        sigs = ks.check_block(epistemic_var=0.51)
        assert len(sigs) == 1
        assert sigs[0].reason == "epistemic"

    # ── Drawdown ───────────────────────────────────────────────────────────────

    def test_drawdown_no_trigger_while_rising(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        ks.check_block(portfolio_value=110.0)
        sigs = ks.check_block(portfolio_value=120.0)
        assert sigs == []

    def test_drawdown_below_threshold_no_signal(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        sigs = ks.check_block(portfolio_value=86.0)  # 14% drawdown < 15%
        assert sigs == []

    def test_drawdown_at_threshold_no_signal(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        sigs = ks.check_block(portfolio_value=85.0)  # exactly 15%
        assert sigs == []

    def test_drawdown_above_fires(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        sigs = ks.check_block(portfolio_value=84.9)  # >15% drawdown
        assert len(sigs) == 1
        assert sigs[0].reason == "drawdown"
        assert sigs[0].value == pytest.approx(0.151, rel=1e-2)

    def test_drawdown_uses_running_peak(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        ks.check_block(portfolio_value=120.0)  # new peak
        sigs = ks.check_block(portfolio_value=101.0)  # only 15.8% from 120 peak
        assert len(sigs) == 1
        assert sigs[0].reason == "drawdown"

    # ── Rug rate ───────────────────────────────────────────────────────────────

    def test_rug_rate_all_normal_no_signal(self):
        ks = KillSwitch(_cfg(rug_rate_threshold=0.5), event_window=4)
        for _ in range(4):
            ks.check_block(event_type="whale_buy")
        assert not ks.is_halted

    def test_rug_rate_at_threshold_no_signal(self):
        ks = KillSwitch(_cfg(rug_rate_threshold=0.5), event_window=4)
        ks.check_block(event_type="rug_pull")
        ks.check_block(event_type="rug_pull")
        ks.check_block(event_type="whale_buy")
        sigs = ks.check_block(event_type="whale_buy")  # 2/4 = 0.5, not >
        assert sigs == []

    def test_rug_rate_above_fires(self):
        ks = KillSwitch(_cfg(rug_rate_threshold=0.5), event_window=4)
        ks.check_block(event_type="rug_pull")
        ks.check_block(event_type="rug_pull")
        ks.check_block(event_type="rug_pull")
        sigs = ks.check_block(event_type="whale_buy")  # 3/4 = 0.75 > 0.5
        assert len(sigs) == 1
        assert sigs[0].reason == "rug_rate"

    def test_rug_rate_window_slides(self):
        ks = KillSwitch(_cfg(rug_rate_threshold=0.5), event_window=4)
        for _ in range(3):
            ks.check_block(event_type="rug_pull")
        ks.check_block(event_type="whale_buy")  # 3/4 > 0.5, halted
        ks.reset()
        # New window: push out rugs with normal events
        for _ in range(4):
            ks.check_block(event_type="whale_buy")
        assert not ks.is_halted

    # ── LP depth drop ──────────────────────────────────────────────────────────

    def test_lp_depth_first_call_no_signal(self):
        ks = _ks(lp_depth_drop_threshold=0.30)
        sigs = ks.check_block(lp_depth=1000.0)
        assert sigs == []

    def test_lp_depth_drop_below_no_signal(self):
        ks = _ks(lp_depth_drop_threshold=0.30)
        ks.check_block(lp_depth=1000.0)
        sigs = ks.check_block(lp_depth=710.0)  # 29% drop
        assert sigs == []

    def test_lp_depth_drop_at_threshold_no_signal(self):
        ks = _ks(lp_depth_drop_threshold=0.30)
        ks.check_block(lp_depth=1000.0)
        sigs = ks.check_block(lp_depth=700.0)  # exactly 30%
        assert sigs == []

    def test_lp_depth_drop_above_fires(self):
        ks = _ks(lp_depth_drop_threshold=0.30)
        ks.check_block(lp_depth=1000.0)
        sigs = ks.check_block(lp_depth=699.0)  # >30% drop
        assert len(sigs) == 1
        assert sigs[0].reason == "lp_depth_drop"

    def test_lp_depth_increase_no_signal(self):
        ks = _ks(lp_depth_drop_threshold=0.30)
        ks.check_block(lp_depth=1000.0)
        sigs = ks.check_block(lp_depth=1500.0)
        assert sigs == []


# ── TestKillSwitchStateful ─────────────────────────────────────────────────────

class TestKillSwitchStateful:
    def test_not_halted_initially(self):
        ks = _ks()
        assert not ks.is_halted

    def test_halted_after_ood_trigger(self):
        ks = _ks(ood_threshold=3.0)
        ks.check_block(ood_score=5.0)
        assert ks.is_halted

    def test_halt_reason_none_before_trigger(self):
        ks = _ks()
        assert ks.halt_reason() is None

    def test_halt_reason_returns_first_signal(self):
        ks = _ks(ood_threshold=3.0, epistemic_threshold=0.5)
        ks.check_block(ood_score=5.0, epistemic_var=0.9)
        assert ks.halt_reason() == "ood"

    def test_get_active_signals_empty_initially(self):
        ks = _ks()
        assert ks.get_active_signals() == []

    def test_get_active_signals_accumulates(self):
        ks = _ks(ood_threshold=3.0)
        ks.check_block(ood_score=4.0)
        ks.check_block(ood_score=5.0)
        sigs = ks.get_active_signals()
        assert len(sigs) == 2
        assert all(s.reason == "ood" for s in sigs)

    def test_get_active_signals_is_snapshot(self):
        ks = _ks(ood_threshold=3.0)
        ks.check_block(ood_score=4.0)
        snapshot = ks.get_active_signals()
        ks.check_block(ood_score=5.0)
        assert len(snapshot) == 1  # snapshot is not mutated

    def test_reset_clears_halted(self):
        ks = _ks(ood_threshold=3.0)
        ks.check_block(ood_score=5.0)
        assert ks.is_halted
        ks.reset()
        assert not ks.is_halted

    def test_reset_clears_active_signals(self):
        ks = _ks(ood_threshold=3.0)
        ks.check_block(ood_score=5.0)
        ks.reset()
        assert ks.get_active_signals() == []
        assert ks.halt_reason() is None

    def test_reset_clears_drawdown_peak(self):
        ks = _ks(max_drawdown_frac=0.15)
        ks.check_block(portfolio_value=100.0)
        ks.reset()
        # After reset, peak is gone — should not fire for a small portfolio value
        sigs = ks.check_block(portfolio_value=10.0)  # first call sets new peak
        assert sigs == []

    def test_multiple_condition_signals_in_one_block(self):
        ks = _ks(ood_threshold=3.0, epistemic_threshold=0.5)
        sigs = ks.check_block(ood_score=5.0, epistemic_var=0.9)
        assert len(sigs) == 2
        reasons = {s.reason for s in sigs}
        assert reasons == {"ood", "epistemic"}


# ── TestKillSwitchMonitor ──────────────────────────────────────────────────────

class TestKillSwitchMonitor:
    def _mon(self, **cfg_overrides) -> KillSwitchMonitor:
        mon = KillSwitchMonitor(_cfg(**cfg_overrides))
        mon.start()
        return mon

    def test_not_halted_initially(self):
        mon = self._mon()
        try:
            assert not mon.is_halted
        finally:
            mon.stop()

    def test_start_stop_clean(self):
        mon = self._mon()
        mon.stop()  # should not raise

    def test_report_triggers_halt(self):
        mon = self._mon(ood_threshold=3.0)
        try:
            mon.report(ood_score=5.0)
            assert mon.wait_processed(timeout=2.0)
            assert mon.is_halted
        finally:
            mon.stop()

    def test_wait_processed_true_when_idle(self):
        mon = self._mon()
        try:
            assert mon.wait_processed(timeout=0.1)
        finally:
            mon.stop()

    def test_wait_processed_after_report(self):
        mon = self._mon(ood_threshold=3.0)
        try:
            mon.report(ood_score=1.0)
            result = mon.wait_processed(timeout=2.0)
            assert result
        finally:
            mon.stop()

    def test_is_halted_delegates(self):
        mon = self._mon(ood_threshold=3.0)
        try:
            assert not mon.is_halted
            mon.report(ood_score=9.0)
            mon.wait_processed(timeout=2.0)
            assert mon.is_halted
        finally:
            mon.stop()

    def test_reset_delegates(self):
        mon = self._mon(ood_threshold=3.0)
        try:
            mon.report(ood_score=9.0)
            mon.wait_processed(timeout=2.0)
            assert mon.is_halted
            mon.reset()
            assert not mon.is_halted
        finally:
            mon.stop()

    def test_multiple_reports_all_processed(self):
        mon = self._mon(ood_threshold=3.0)
        try:
            for i in range(5):
                mon.report(ood_score=float(i))
            assert mon.wait_processed(timeout=3.0)
        finally:
            mon.stop()

    def test_build_monitor_factory_starts(self):
        cfg = _cfg(ood_threshold=3.0)
        mon = build_monitor(cfg)
        try:
            assert not mon.is_halted
        finally:
            mon.stop()


# ── TestLatency ────────────────────────────────────────────────────────────────

class TestLatency:
    N = 200

    def test_check_block_p99_under_1ms(self):
        ks = _ks()
        latencies = []
        for i in range(self.N):
            t0 = time.perf_counter()
            ks.check_block(
                ood_score=float(i % 3),
                epistemic_var=0.1,
                portfolio_value=100.0 + i,
                event_type="whale_buy",
                lp_depth=1000.0,
            )
            latencies.append(time.perf_counter() - t0)
        latencies.sort()
        p99 = latencies[int(0.99 * self.N)]
        assert p99 < 1e-3, f"check_block p99={p99*1000:.2f}ms > 1ms"

    def test_report_nonblocking_p99_under_1ms(self):
        cfg = _cfg()
        mon = KillSwitchMonitor(cfg)
        mon.start()
        try:
            latencies = []
            for i in range(self.N):
                t0 = time.perf_counter()
                mon.report(ood_score=float(i % 3))
                latencies.append(time.perf_counter() - t0)
            mon.wait_processed(timeout=5.0)
            latencies.sort()
            p99 = latencies[int(0.99 * self.N)]
            assert p99 < 1e-3, f"report p99={p99*1000:.2f}ms > 1ms"
        finally:
            mon.stop()

    def test_build_factories(self):
        cfg = _cfg()
        ks = build_kill_switch(cfg)
        assert isinstance(ks, KillSwitch)
        mon = build_monitor(cfg)
        assert isinstance(mon, KillSwitchMonitor)
        mon.stop()
