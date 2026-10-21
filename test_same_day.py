"""Tests für den Same-Day-Modus (Live-Kurs, Gate, Signal, Trader-Prüfungen). Ausführen:
    python -m unittest test_same_day -v
"""
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from wf_backtest import core_intraday as ci
from wf_backtest import core_strategy as cs
from trading import trade_window as tw

NY = ZoneInfo("America/New_York")
NOW_NY = datetime(2026, 10, 21, 15, 40, tzinfo=NY)          # Mittwoch, 20 min vor Schluss
NOW_UTC = NOW_NY.astimezone(timezone.utc)


def _import_trader():
    """alpaca_trader lädt beim Import eine lokale .env — im Test nie lesen."""
    orig = Path.exists
    with mock.patch.object(Path, "exists", lambda self: False if self.name == ".env" else orig(self)):
        from trading import alpaca_trader
    return alpaca_trader


at = _import_trader()


def _snapshot(px=101.0, prev=100.0, age_s=30, prev_date="2026-10-20"):
    return {"latestTrade": {"p": px, "t": (NOW_UTC - timedelta(seconds=age_s)).isoformat().replace("+00:00", "Z")},
            "prevDailyBar": {"c": prev, "t": f"{prev_date}T04:00:00Z"}}


class LiveQuote(unittest.TestCase):
    def test_session_window(self):
        self.assertTrue(ci.session_is_running(NOW_NY))
        self.assertFalse(ci.session_is_running(NOW_NY.replace(hour=16, minute=0)))
        self.assertFalse(ci.session_is_running(NOW_NY.replace(hour=9, minute=29)))
        self.assertFalse(ci.session_is_running(datetime(2026, 10, 24, 12, 0, tzinfo=NY)))     # Samstag

    def test_proxy_scales_last_index_close(self):
        est = ci.proxy_estimate(_snapshot(px=101.0, prev=100.0), 5000.0, NOW_NY.date() - timedelta(days=1), NOW_UTC)
        self.assertAlmostEqual(est, 5050.0)

    def test_proxy_rejects_stale_quote_and_date_mismatch(self):
        with self.assertRaises(ci.LiveQuoteError):
            ci.proxy_estimate(_snapshot(age_s=ci.MAX_QUOTE_AGE_S + 60), 5000.0, NOW_NY.date() - timedelta(days=1), NOW_UTC)
        with self.assertRaises(ci.LiveQuoteError):
            ci.proxy_estimate(_snapshot(prev_date="2026-10-19"), 5000.0, NOW_NY.date() - timedelta(days=1), NOW_UTC)
        with self.assertRaises(ci.LiveQuoteError):
            ci.proxy_estimate({"latestTrade": {}}, 5000.0, NOW_NY.date(), NOW_UTC)

    def test_combine(self):
        self.assertEqual(ci.combine_estimates(100.0, 100.3)[0], 100.0)
        self.assertEqual(ci.combine_estimates(None, 100.3)[0], 100.3)
        self.assertEqual(ci.combine_estimates(100.0, None)[0], 100.0)
        with self.assertRaises(ci.LiveQuoteError):
            ci.combine_estimates(100.0, 102.0)             # 2 % Abweichung
        with self.assertRaises(ci.LiveQuoteError):
            ci.combine_estimates(None, None)

    def test_splice(self):
        s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-19", "2026-10-20"]))
        out = ci.splice_live_bar(s, 3.0, NOW_NY)
        self.assertEqual(out.index[-1], pd.Timestamp("2026-10-21"))
        self.assertEqual(out.iloc[-1], 3.0)
        self.assertEqual(len(s), 2)                         # Original unverändert
        with self.assertRaises(ValueError):
            ci.splice_live_bar(out, 4.0, NOW_NY)


def _flat_close(last=100.5):
    idx = pd.bdate_range(end="2026-10-20", periods=300)
    px = np.full(len(idx), 100.0)
    px[-1] = last
    return pd.Series(px, index=idx)


class SignalTiming(unittest.TestCase):
    def _signal(self, timing, now_ny=NOW_NY, live_price=102.0, live_error=None):
        est = mock.Mock(side_effect=live_error) if live_error else \
            mock.Mock(return_value={"price": live_price, "source": "mock", "quote_utc": "x"})
        with mock.patch.object(cs, "load_signal_close", lambda sl, cfg, now: (_flat_close(), "mock")), \
                mock.patch.object(ci, "estimate_index_close", est):
            return cs.compute_core_signal(now_ny=now_ny, with_backtest=False, timing=timing)

    def test_nextday_ignores_live_price(self):
        sig = self._signal("nextday")
        self.assertEqual(sig["status"], "ok")
        self.assertEqual((sig["signal"], sig["as_of"], sig["intraday"]), ("RISK_OFF", "2026-10-20", False))

    def test_sameday_uses_live_price_as_todays_close(self):
        sig = self._signal("sameday")
        self.assertEqual(sig["status"], "ok")
        self.assertEqual((sig["signal"], sig["as_of"], sig["intraday"], sig["timing"]),
                         ("RISK_ON", "2026-10-21", True, "sameday"))
        self.assertEqual(set(sig["weights"]), {"SSO", "QLD"})
        self.assertEqual(sig["sleeves"][0]["index"]["close"], 102.0)

    def test_sameday_live_price_below_band_stays_off(self):
        self.assertEqual(self._signal("sameday", live_price=100.8)["signal"], "RISK_OFF")

    def test_sameday_without_session_has_no_live_bar(self):
        sig = self._signal("sameday", now_ny=NOW_NY.replace(hour=17))
        self.assertFalse(sig["intraday"])
        self.assertEqual(sig["as_of"], "2026-10-20")

    def test_sameday_quote_failure_blocks_signal(self):
        sig = self._signal("sameday", live_error=ci.LiveQuoteError("down"))
        self.assertEqual(sig["status"], "error")
        self.assertIn("live quote failed", sig["error"])

    def test_resolve_timing(self):
        with mock.patch.dict("os.environ", {"CORE_TIMING": "sameday"}):
            self.assertEqual(cs.resolve_timing(), "sameday")
        with mock.patch.dict("os.environ", {"CORE_TIMING": "weekly"}):
            with self.assertRaises(ValueError):
                cs.resolve_timing()


class BacktestTiming(unittest.TestCase):
    def test_sameday_holds_decision_from_the_close(self):
        cfg = dict(cs.CONFIG)
        syms = list(dict.fromkeys([cfg["etf_cash_hist"], cfg["benchmark"]] +
                                  [sl[k] for sl in cfg["sleeves"] for k in ("etf_1x", "etf_2x")]))
        idx = pd.bdate_range("2020-01-01", periods=20)
        rng = np.random.default_rng(1)
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (20, len(syms))), axis=0), index=idx, columns=syms)
        openp = close.shift(1).fillna(close.iloc[0]) * 1.001
        df = pd.concat({"Open": openp, "Close": close}, axis=1)
        cfg["backtest_start"] = "2020-01-01"
        states = {sl["name"]: pd.Series(1.0, index=idx) for sl in cfg["sleeves"]}
        with mock.patch.object(cs, "_yf_ohlc", lambda *a, **k: df):
            same = cs.backtest_real(states, cfg, "sameday")
            nxt = cs.backtest_real(states, cfg, "nextday")
        self.assertEqual((same["timing"], nxt["timing"]), ("sameday", "nextday"))
        w = cs.target_weights({sl["name"]: True for sl in cfg["sleeves"]}, cfg, cfg["etf_cash_hist"])
        w[cfg["etf_cash_hist"]] = w.get(cfg["etf_cash_hist"], 0.0) + max(0.0, 1.0 - sum(w.values()))
        ret = close.pct_change()
        eq = same["equity"]
        self.assertEqual(eq[1]["date"], str(idx[2].date()))
        # konstante Gewichte: Tagesrendite = Close-zu-Close mit den Zielgewichten
        self.assertAlmostEqual(eq[1]["switch"] / eq[0]["switch"] - 1, sum(wt * ret[s].iloc[2] for s, wt in w.items()), delta=3e-4)
        # Der erste Handelstag unterscheidet sich: Same-Day hält ab dem Close, Folgetag erst ab der Eröffnung
        self.assertNotAlmostEqual(same["equity"][0]["switch"], nxt["equity"][0]["switch"], places=4)


class Gate(unittest.TestCase):
    def d(self, event, schedule="", repo="", inp="", secs=1200.0):
        return tw.decide(event, schedule, repo, inp, lambda: secs)

    def test_default_uses_sameday(self):
        self.assertFalse(self.d("schedule", tw.LEGACY_CRON).go)
        d = self.d("schedule", "28 19 * * 1-5", secs=20 * 60)
        self.assertTrue(d.go)
        self.assertEqual(d.timing, "sameday")

    def test_nextday_variable_swaps_to_legacy_cron(self):
        self.assertEqual(self.d("schedule", tw.LEGACY_CRON, repo="nextday"), tw.Decision(True, "nextday", "Folgetag-Lauf"))
        self.assertFalse(self.d("schedule", "28 19 * * 1-5", repo="nextday").go)

    def test_window_edges(self):
        for secs, go in ((36 * 60, False), (35 * 60, True), (7 * 60, True), (6 * 60, False), (None, False)):
            self.assertEqual(self.d("schedule", "41 20 * * 1-5", repo="sameday", secs=secs).go, go, secs)

    def test_dispatch(self):
        self.assertEqual(self.d("workflow_dispatch", inp="auto"), tw.Decision(True, "sameday", "manueller Lauf (sameday)"))
        self.assertEqual(self.d("workflow_dispatch", repo="nextday", inp="auto").timing, "nextday")
        self.assertEqual(self.d("workflow_dispatch", repo="sameday", inp="nextday").timing, "nextday")
        with self.assertRaises(ValueError):
            self.d("workflow_dispatch", inp="hourly")

    def test_clock_only_needed_for_sameday_cron(self):
        boom = mock.Mock(side_effect=AssertionError("Uhr darf hier nicht abgefragt werden"))
        tw.decide("schedule", tw.LEGACY_CRON, "", "", boom)
        tw.decide("schedule", tw.LEGACY_CRON, "nextday", "", boom)
        tw.decide("schedule", "28 19 * * 1-5", "nextday", "", boom)

    def test_seconds_to_close(self):
        clock = {"is_open": True, "timestamp": "2026-10-21T15:40:00-04:00", "next_close": "2026-10-21T16:00:00-04:00"}
        self.assertEqual(tw.seconds_to_close(clock), 1200.0)
        self.assertEqual(at.seconds_to_close(clock), 1200.0)
        self.assertIsNone(at.seconds_to_close({"is_open": False}))


class TraderValidation(unittest.TestCase):
    def _sig(self, **kw):
        sig = {"status": "ok", "signal": "RISK_ON", "as_of": "2026-10-21", "timing": "sameday", "intraday": True,
               "generated_utc": NOW_UTC.strftime("%Y-%m-%d %H:%M UTC"), "weights": {"SSO": 0.495, "QLD": 0.495}}
        sig.update(kw)
        return sig

    def test_valid_live_signal(self):
        self.assertIsNone(at.validate_core_signal(self._sig(), NOW_NY))

    def test_rejects_unsafe_live_signals(self):
        for bad, text in ((dict(intraday=False), "Live-Kurs"),
                          (dict(as_of="2026-10-20"), "heute ist"),
                          (dict(generated_utc=(NOW_UTC - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M UTC")), "min alt"),
                          (dict(generated_utc="kaputt"), "generated_utc")):
            err = at.validate_core_signal(self._sig(**bad), NOW_NY)
            self.assertIsNotNone(err, bad)
            self.assertIn(text, err)

    def test_nextday_signal_unchanged(self):
        sig = self._sig(timing="nextday", intraday=False, as_of="2026-10-20")
        self.assertIsNone(at.validate_core_signal(sig, NOW_NY))
        sig.pop("timing")
        self.assertIsNone(at.validate_core_signal(sig, NOW_NY))

    def test_trade_log_deduplicates_backup_cron_on_same_day(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trade_log.json"
            with mock.patch.object(at, "TRADE_LOG_PAPER", p):
                extra = {"core_signal": "RISK_ON", "signal_as_of": "2026-10-21", "timing": "sameday"}
                orders = [{"symbol": "SSO", "side": "buy", "qty": 10.0, "value": 900.0, "status": "filled"}]
                at._save_trade_log(1000.0, 100.0, {"SSO": 10.0}, {"SSO": 90.0}, {}, orders, True, False, 0.0, extra)
                at._save_trade_log(1005.0, 100.0, {"SSO": 10.0}, {"SSO": 90.5}, {}, [], True, False, 0.0, extra)
                data = json.loads(p.read_text(encoding="utf-8"))
                self.assertEqual(len(data), 1)
                self.assertEqual(data[0]["equity"], 1005.0)
                self.assertEqual(len(data[0]["orders"]), 1)


if __name__ == "__main__":
    unittest.main()
