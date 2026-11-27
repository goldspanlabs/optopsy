"""Regression tests for core backtesting, strategy, simulator, and signal correctness fixes."""

from __future__ import annotations

import datetime
import functools

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import optopsy as op
from optopsy.checks import _run_checks
from optopsy.datafeeds import load_cached_options, load_cached_stocks, options_data
from optopsy.filters import _cut_options_by_delta, _cut_options_by_dte
from optopsy.metrics import omega_ratio
from optopsy.output import _group_by_intervals
from optopsy.pricing import _calculate_fill_price
from optopsy.rules import _rule_butterfly_strikes
from optopsy.signals import (
    and_signals,
    atr_above,
    atr_below,
    daily_return_above,
    daily_return_below,
    day_of_week,
    donchian_above_upper,
    donchian_below_lower,
    drawdown_from_high,
    fisher_cross_above,
    gap_up,
    iv_rank_above,
    psar_buy,
    psar_sell,
    rally_from_low,
    signal,
    signal_dates,
    sma_above,
    smi_cross_above,
    squeeze_off,
    squeeze_on,
    stoch_below,
    supertrend_buy,
    supertrend_sell,
    sustained,
    tsi_cross_above,
)
from optopsy.simulator import (
    _select_highest_premium,
    _select_lowest_premium,
    _select_nearest,
    simulate,
)
from optopsy.types import CalendarStrategyParams, StrategyParams, TargetRange

# =============================================================================
# Helpers
# =============================================================================


def _make_option_row(
    symbol: str,
    opt_type: str,
    expiration: str | datetime.datetime,
    quote_date: str | datetime.datetime,
    strike: float,
    bid: float,
    ask: float,
    delta: float,
    *,
    volume: int = 100,
    implied_volatility: float | None = None,
) -> dict:
    row: dict = {
        "underlying_symbol": symbol,
        "option_type": opt_type,
        "expiration": pd.Timestamp(expiration),
        "quote_date": pd.Timestamp(quote_date),
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "delta": delta,
        "volume": volume,
    }
    if implied_volatility is not None:
        row["implied_volatility"] = implied_volatility
    return row


def _make_stock_series(
    closes: list[float],
    *,
    symbol: str = "SPX",
    start: str = "2020-01-02",
    spread: float = 0.01,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes))
    close_arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "underlying_symbol": symbol,
            "quote_date": dates,
            "open": close_arr,
            "high": close_arr * (1.0 + spread),
            "low": close_arr * (1.0 - spread),
            "close": close_arr,
            "volume": 1_000_000,
        }
    )


def _tr(target: float, tol: float = 0.05) -> TargetRange:
    return TargetRange(
        target=target,
        min=max(0.0, target - tol),
        max=min(1.0, target + tol),
    )


# =============================================================================
# 1. Evaluation, Filters, Output, Pricing, Rules, Checks, Metrics, Types
# =============================================================================


class TestEvaluationAndFilteringRegressions:
    def test_exit_dte_tolerance_matches_lower_half(self):
        """Exits below exit_dte within tolerance [exit_dte - tol, exit_dte) must match."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Entry at DTE=30
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 4.9, 5.1, 0.30
                ),
                # Exit only available at DTE=8 (below exit_dte=10, within tol=3)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-24", 100.0, 6.9, 7.1, 0.45
                ),
            ]
        )
        res = op.long_calls(
            df,
            max_entry_dte=45,
            exit_dte=10,
            exit_dte_tolerance=3,
            slippage="mid",
            raw=True,
        )
        assert len(res) == 1
        # Only DTE=30 should enter; DTE=8 must NOT also be treated as an entry
        assert res["dte_entry"].iloc[0] == 30
        assert res["exit"].iloc[0] == pytest.approx(7.0)

    def test_exit_dates_applied_before_tolerance_snapping(self):
        """A valid exit_dates match within tolerance must not be shadowed by closer non-signal rows."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Entry at DTE=30
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 4.9, 5.1, 0.30
                ),
                # Signal exit at DTE=12 (2020-01-20)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-20", 100.0, 7.9, 8.1, 0.50
                ),
                # Closer non-signal date at DTE=10 (2020-01-22)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-22", 100.0, 2.9, 3.1, 0.20
                ),
            ]
        )
        exit_dates = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-20")]}
        )
        res = op.long_calls(
            df,
            max_entry_dte=45,
            exit_dte=10,
            exit_dte_tolerance=3,
            exit_dates=exit_dates,
            slippage="mid",
            raw=True,
        )
        assert len(res) == 1
        assert res["exit"].iloc[0] == pytest.approx(8.0)

    def test_exit_dte_tolerance_deterministic_tie_breaking(self):
        """Equidistant DTE ties must break identically regardless of input row order."""
        exp = "2020-02-01"
        rows = [
            _make_option_row("SPX", "call", exp, "2020-01-02", 100.0, 4.9, 5.1, 0.30),
            # DTE=12 (diff=2)
            _make_option_row("SPX", "call", exp, "2020-01-20", 100.0, 6.9, 7.1, 0.40),
            # DTE=8 (diff=2)
            _make_option_row("SPX", "call", exp, "2020-01-24", 100.0, 8.9, 9.1, 0.55),
        ]
        entry_dates = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-02")]}
        )
        res_fwd = op.long_calls(
            pd.DataFrame(rows),
            max_entry_dte=45,
            exit_dte=10,
            exit_dte_tolerance=2,
            entry_dates=entry_dates,
            raw=True,
        )
        res_rev = op.long_calls(
            pd.DataFrame(list(reversed(rows))),
            max_entry_dte=45,
            exit_dte=10,
            exit_dte_tolerance=2,
            entry_dates=entry_dates,
            raw=True,
        )
        assert len(res_fwd) == 1
        assert res_fwd["exit"].iloc[0] == pytest.approx(res_rev["exit"].iloc[0])

    def test_cut_options_by_dte_keeps_top_bin(self):
        """Rows with dte_entry == max_entry_dte must not be dropped as NaN."""
        df = pd.DataFrame({"dte_entry": [7, 14, 28, 30]})
        cut = _cut_options_by_dte(df, dte_interval=7, max_entry_dte=30)
        assert cut["dte_range"].notna().all()

    def test_cut_options_by_delta_keeps_minus_one_and_fine_interval(self):
        """delta == -1.0 must be binned (include_lowest=True) and fine intervals must not error."""
        df = pd.DataFrame({"delta_entry": [-1.0, -0.5, 0.0, 0.5, 1.0]})
        for interval in (0.05, 0.025, 0.005):
            cut = _cut_options_by_delta(df, delta_interval=interval)
            assert cut["delta_range"].notna().all()

    def test_drop_nan_drops_all_nan_groups_and_profit_factor_nan(self):
        """drop_nan=True must drop groups with 0 valid pct_change rows and keep profit_factor NaN."""
        df = pd.DataFrame(
            {
                "dte_range": ["(0, 7]", "(0, 7]", "(7, 14]"],
                "pct_change": [np.nan, np.nan, 0.25],
            }
        )
        grouped_keep = _group_by_intervals(df, ["dte_range"], drop_na=False)
        assert pd.isna(grouped_keep.loc["(0, 7]", "profit_factor"])

        grouped_drop = _group_by_intervals(df, ["dte_range"], drop_na=True)
        assert list(grouped_drop.index) == ["(7, 14]"]

    def test_pricing_clamps_negative_volume_and_rejects_unknown_slippage(self):
        """Negative volume must clamp within [bid, ask]; unknown slippage must raise ValueError."""
        bid = pd.Series([1.0, 2.0])
        ask = pd.Series([2.0, 3.0])
        vol = pd.Series([-50.0, 5000.0])

        buy_fill = _calculate_fill_price(bid, ask, "buy", "liquidity", volume=vol)
        sell_fill = _calculate_fill_price(bid, ask, "sell", "liquidity", volume=vol)
        assert (buy_fill >= bid).all() and (buy_fill <= ask).all()
        assert (sell_fill >= bid).all() and (sell_fill <= ask).all()

        with pytest.raises(ValueError, match="Unknown slippage"):
            _calculate_fill_price(bid, ask, "buy", "invalid_mode")

    def test_butterfly_floating_point_wing_widths(self):
        """Equal wing widths like 100.1 / 100.2 / 100.3 must pass despite float64 rounding."""
        df = pd.DataFrame(
            {
                "strike_leg1": [100.1, 100.1],
                "strike_leg2": [100.2, 100.2],
                "strike_leg3": [100.3, 100.5],
            }
        )
        filtered = _rule_butterfly_strikes(df, [(1, None), (-1, None), (1, None)])
        assert len(filtered) == 1
        assert filtered["strike_leg3"].iloc[0] == pytest.approx(100.3)

    def test_checks_accept_tz_aware_datetimes_and_reject_bool_numeric(self):
        """_run_checks must accept tz-aware datetimes and reject boolean columns as numeric."""
        df = pd.DataFrame(
            {
                "underlying_symbol": ["SPX"],
                "option_type": ["call"],
                "expiration": pd.to_datetime(["2020-02-01"], utc=True),
                "quote_date": pd.to_datetime(["2020-01-02"], utc=True),
                "strike": [100.0],
                "bid": [5.0],
                "ask": [5.2],
                "delta": [0.30],
            }
        )
        _run_checks({}, df)

        bad_df = df.copy()
        bad_df["strike"] = [True]
        with pytest.raises(ValueError, match="does not match expected types"):
            _run_checks({}, bad_df)

    def test_types_reject_nan_inf_and_invalid_calendar_exit_dte(self):
        """NaN/inf floats and calendar exit_dte >= front_dte_max must be rejected."""
        with pytest.raises(ValidationError):
            StrategyParams(stop_loss=float("nan"))
        with pytest.raises(ValidationError):
            StrategyParams(take_profit=float("inf"))
        with pytest.raises(ValidationError):
            CalendarStrategyParams(
                front_dte_min=7,
                front_dte_max=14,
                back_dte_min=30,
                back_dte_max=60,
                exit_dte=14,
            )

    def test_omega_ratio_scales_risk_free_rate_daily_and_handles_nullable_series(self):
        """omega_ratio must scale annual risk_free_rate by trading_days and handle Float64 Series."""
        returns = pd.Series([0.002, 0.003, -0.001, 0.004, -0.0005], dtype="Float64")
        val = omega_ratio(returns, risk_free_rate=0.05, trading_days=252)
        assert np.isfinite(val)
        assert val > 1.0

    def test_options_data_and_single_leg_do_not_mutate_caller_df(self):
        """options_data() and single-leg strategies must not mutate caller DataFrames."""
        raw = pd.DataFrame(
            {
                "underlying_symbol": ["SPX", "SPX"],
                "option_type": ["call", "call"],
                "expiration": ["2020-02-01", "2020-02-01"],
                "quote_date": ["2020-01-02", "2020-02-01"],
                "strike": [100.0, 100.0],
                "bid": [4.9, 7.9],
                "ask": [5.1, 8.1],
                "delta": [0.30, 0.50],
            }
        )
        orig_quote_dtype = raw["quote_date"].dtype
        parsed = options_data(raw)
        assert raw["quote_date"].dtype == orig_quote_dtype
        assert pd.api.types.is_datetime64_any_dtype(parsed["quote_date"])

        parsed_cols = list(parsed.columns)
        _ = op.long_calls(parsed, raw=True)
        assert list(parsed.columns) == parsed_cols

    def test_load_cached_options_preserves_underlying_price_and_stocks_normalizes_tz(
        self, monkeypatch
    ):
        """load_cached_options keeps underlying_price; load_cached_stocks strips tz info."""
        from optopsy.data.providers import cache as cache_mod

        opt_fixture = pd.DataFrame(
            {
                "underlying_symbol": ["SPX"],
                "option_type": ["call"],
                "expiration": pd.to_datetime(["2020-02-01"]),
                "quote_date": pd.to_datetime(["2020-01-02"]),
                "strike": [100.0],
                "bid": [5.0],
                "ask": [5.2],
                "delta": [0.30],
                "underlying_price": [101.5],
            }
        )
        stk_fixture = pd.DataFrame(
            {
                "underlying_symbol": ["SPX"],
                "quote_date": pd.to_datetime(["2020-01-02"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        class _FakeStore:
            def read(self, dataset: str, _symbol: str) -> pd.DataFrame:
                return (
                    opt_fixture.copy() if dataset == "options" else stk_fixture.copy()
                )

        monkeypatch.setattr(cache_mod, "get_store", lambda: _FakeStore())
        loaded_opts = load_cached_options("SPX")
        assert "underlying_price" in loaded_opts.columns
        assert loaded_opts["underlying_price"].iloc[0] == pytest.approx(101.5)

        loaded_stk = load_cached_stocks("SPX")
        assert loaded_stk["quote_date"].dt.tz is None


# =============================================================================
# 2. Core Multi-Leg, Calendar Spreads, Early Exits
# =============================================================================


class TestMultiLegCalendarAndExitsRegressions:
    def test_multi_leg_rejects_mismatched_exit_dates_under_tolerance(self):
        """Multi-leg spreads must not pair legs that closed on different dates."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Entries at DTE=30
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 6.0, 6.2, 0.50
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 105.0, 3.0, 3.2, 0.30
                ),
                # Leg 1 only has exit at DTE=10 (2020-01-22)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-22", 100.0, 7.0, 7.2, 0.55
                ),
                # Leg 2 only has exit at DTE=12 (2020-01-20)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-20", 105.0, 2.0, 2.2, 0.25
                ),
            ]
        )
        res = op.long_call_spread(
            df,
            max_entry_dte=45,
            exit_dte=10,
            exit_dte_tolerance=3,
            leg1_delta=_tr(0.50),
            leg2_delta=_tr(0.30),
            raw=True,
        )
        assert len(res) == 0

    def test_calendar_no_look_backward_exit_dates_and_liquidity_slippage(self):
        """Calendar spreads must enforce exit_date > quote_date, apply exit_dates before tolerance, and use volume for liquidity slippage."""
        front_exp = "2020-01-25"
        back_exp = "2020-02-25"
        base_rows = [
            # Entry 2020-01-10 (front DTE=15)
            _make_option_row(
                "SPX", "call", front_exp, "2020-01-10", 100.0, 2.0, 3.0, 0.30, volume=10
            ),
            _make_option_row(
                "SPX", "call", back_exp, "2020-01-10", 100.0, 4.0, 5.0, 0.30, volume=10
            ),
            # Signal exit 2020-01-16 (front DTE=9): exit_leg2 mid = 4.0
            _make_option_row(
                "SPX", "call", front_exp, "2020-01-16", 100.0, 1.0, 2.0, 0.30, volume=10
            ),
            _make_option_row(
                "SPX", "call", back_exp, "2020-01-16", 100.0, 3.5, 4.5, 0.30, volume=10
            ),
            # Closer non-signal exit 2020-01-18 (front DTE=7): exit_leg2 mid = 7.0
            _make_option_row(
                "SPX", "call", front_exp, "2020-01-18", 100.0, 0.5, 1.5, 0.30, volume=10
            ),
            _make_option_row(
                "SPX", "call", back_exp, "2020-01-18", 100.0, 6.5, 7.5, 0.30, volume=10
            ),
            # Late entry 2020-01-20 (front DTE=5) — earlier exit rows at DTE=7/9 precede entry!
            _make_option_row(
                "SPX", "call", front_exp, "2020-01-20", 100.0, 0.4, 0.6, 0.30, volume=10
            ),
            _make_option_row(
                "SPX", "call", back_exp, "2020-01-20", 100.0, 2.5, 3.5, 0.30, volume=10
            ),
        ]
        df_low_vol = pd.DataFrame(base_rows)

        # 1. No look-backward: 2020-01-20 entry (front DTE=5) must never exit backwards on 2020-01-18
        res_all = op.long_call_calendar(
            df_low_vol,
            front_dte_min=1,
            front_dte_max=20,
            back_dte_min=25,
            back_dte_max=60,
            exit_dte=7,
            exit_dte_tolerance=3,
            raw=True,
        )
        assert len(res_all) >= 1
        assert 5 not in set(res_all["dte_entry_leg1"])

        # 2. exit_dates applied before tolerance snapping: picks 2020-01-16 (exit_leg2=4.0 mid)
        #    over closer 2020-01-18 (exit_leg2=7.0 mid)
        exit_dates = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-16")]}
        )
        res_sig = op.long_call_calendar(
            df_low_vol,
            front_dte_min=1,
            front_dte_max=20,
            back_dte_min=25,
            back_dte_max=60,
            exit_dte=7,
            exit_dte_tolerance=3,
            exit_dates=exit_dates,
            slippage="mid",
            raw=True,
        )
        assert len(res_sig) == 1
        assert res_sig["exit_leg2"].iloc[0] == pytest.approx(4.0)

        # 3. Liquidity slippage must respect volume (high volume has tighter fill than low volume)
        df_high_vol = df_low_vol.copy()
        df_high_vol["volume"] = 5000
        res_low = op.long_call_calendar(
            df_low_vol,
            front_dte_min=10,
            front_dte_max=20,
            back_dte_min=25,
            back_dte_max=60,
            exit_dte=7,
            slippage="liquidity",
            raw=True,
        )
        res_high = op.long_call_calendar(
            df_high_vol,
            front_dte_min=10,
            front_dte_max=20,
            back_dte_min=25,
            back_dte_max=60,
            exit_dte=7,
            slippage="liquidity",
            raw=True,
        )
        assert (
            res_low["total_entry_cost"].iloc[0] > res_high["total_entry_cost"].iloc[0]
        )

    def test_early_exit_does_not_scan_past_planned_exit_dte(self):
        """Early exit monitor must not trigger on quotes occurring AFTER planned exit_dte."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Entry at DTE=30: mid = 10.0
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 9.9, 10.1, 0.30
                ),
                # Intermediate at DTE=20: mid = 10.5 (no stop triggered)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-12", 100.0, 10.4, 10.6, 0.35
                ),
                # Planned exit at DTE=10 (2020-01-22): mid = 11.0 (+10%)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-22", 100.0, 10.9, 11.1, 0.40
                ),
                # Post-exit crash at DTE=5 (2020-01-27): mid = 1.0 (-90%) — AFTER planned exit!
                _make_option_row(
                    "SPX", "call", exp, "2020-01-27", 100.0, 0.9, 1.1, 0.05
                ),
            ]
        )
        entry_dates = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-02")]}
        )
        res = op.long_calls(
            df,
            max_entry_dte=45,
            exit_dte=10,
            entry_dates=entry_dates,
            stop_loss=-0.50,
            slippage="mid",
            raw=True,
        )
        assert len(res) == 1
        assert res["exit_type"].iloc[0] == "expiration"
        assert res["pct_change"].iloc[0] == pytest.approx(0.10)

    def test_early_exit_works_on_straddles_and_calendar_spreads(self):
        """Early exit monitoring must work cleanly on straddles (shared strike) and calendar spreads (shared option_type & strike)."""
        # 1. Straddle (same strike for call & put)
        exp = "2020-02-01"
        straddle_df = pd.DataFrame(
            [
                # Entry 2020-01-02: call=5.0 + put=5.0 = 10.0
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 4.9, 5.1, 0.50
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 100.0, 4.9, 5.1, -0.50
                ),
                # Intermediate 2020-01-12: call=2.0 + put=2.0 = 4.0 (-60% -> hits stop_loss=-0.40)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-12", 100.0, 1.9, 2.1, 0.50
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-12", 100.0, 1.9, 2.1, -0.50
                ),
                # Expiration 2020-02-01: call=6.0 + put=6.0 = 12.0
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 100.0, 5.9, 6.1, 0.50
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 100.0, 5.9, 6.1, -0.50
                ),
            ]
        )
        entry_dates = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-02")]}
        )
        res_straddle = op.long_straddles(
            straddle_df,
            max_entry_dte=45,
            exit_dte=0,
            entry_dates=entry_dates,
            stop_loss=-0.40,
            slippage="mid",
            raw=True,
        )
        assert len(res_straddle) == 1
        assert res_straddle["exit_type"].iloc[0] == "stop_loss"
        assert res_straddle["_early_exit_date"].iloc[0] == pd.Timestamp("2020-01-12")
        assert res_straddle["pct_change"].iloc[0] == pytest.approx(-0.60)

        # 2. Calendar spread (same option_type & strike, different expirations)
        front_exp = "2020-01-25"
        back_exp = "2020-02-25"
        cal_df = pd.DataFrame(
            [
                # Entry 2020-01-05 (front DTE=20, back DTE=51): -front(2.0) + back(6.0) = 4.0
                _make_option_row(
                    "SPX", "call", front_exp, "2020-01-05", 100.0, 1.9, 2.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", back_exp, "2020-01-05", 100.0, 5.9, 6.1, 0.30
                ),
                # Intermediate 2020-01-12: -front(2.0) + back(4.0) = 2.0 (-50% -> hits stop_loss=-0.40)
                _make_option_row(
                    "SPX", "call", front_exp, "2020-01-12", 100.0, 1.9, 2.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", back_exp, "2020-01-12", 100.0, 3.9, 4.1, 0.30
                ),
                # Planned exit at front DTE=7 (2020-01-18): -front(1.0) + back(6.0) = 5.0
                _make_option_row(
                    "SPX", "call", front_exp, "2020-01-18", 100.0, 0.9, 1.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", back_exp, "2020-01-18", 100.0, 5.9, 6.1, 0.30
                ),
            ]
        )
        cal_entry = pd.DataFrame(
            {"underlying_symbol": ["SPX"], "quote_date": [pd.Timestamp("2020-01-05")]}
        )
        res_cal = op.long_call_calendar(
            cal_df,
            front_dte_min=10,
            front_dte_max=30,
            back_dte_min=35,
            back_dte_max=65,
            exit_dte=7,
            entry_dates=cal_entry,
            stop_loss=-0.40,
            slippage="mid",
            raw=True,
        )
        assert len(res_cal) == 1
        assert res_cal["exit_type"].iloc[0] == "stop_loss"
        assert res_cal["_early_exit_date"].iloc[0] == pd.Timestamp("2020-01-12")
        assert res_cal["pct_change"].iloc[0] == pytest.approx(-0.50)


# =============================================================================
# 3. Strategies Regressions
# =============================================================================


class TestStrategiesRegressions:
    def test_put_back_spread_and_put_front_spread_leg_economics(self):
        """put_back_spread must be +2 lower-strike puts / -1 higher-strike put, and put_front_spread the inverse."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Entry: lower strike 95 put (delta -0.15) = 2.0, higher strike 100 put (delta -0.50) = 3.5
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 95.0, 1.9, 2.1, -0.15
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 100.0, 3.4, 3.6, -0.50
                ),
                # Exit (crash to 85): strike 95 put = 10.0, strike 100 put = 15.0
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 95.0, 9.9, 10.1, -0.95
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 100.0, 14.9, 15.1, -0.99
                ),
            ]
        )
        back = op.put_back_spread(
            df, max_entry_dte=45, exit_dte=0, slippage="mid", raw=True
        )
        assert len(back) == 1
        # +2 * 2.0 - 1 * 3.5 = +0.5 debit entry; exit = +2 * 10.0 - 1 * 15.0 = +5.0 (profits on crash!)
        assert back["total_entry_cost"].iloc[0] == pytest.approx(0.5)
        assert back["total_exit_proceeds"].iloc[0] == pytest.approx(5.0)
        assert back["pct_change"].iloc[0] > 0

        front = op.put_front_spread(
            df, max_entry_dte=45, exit_dte=0, slippage="mid", raw=True
        )
        assert len(front) == 1
        # -2 * 2.0 + 1 * 3.5 = -0.5 credit entry; exit = -2 * 10.0 + 1 * 15.0 = -5.0 (loses on crash!)
        assert front["total_entry_cost"].iloc[0] == pytest.approx(-0.5)
        assert front["total_exit_proceeds"].iloc[0] == pytest.approx(-5.0)
        assert front["pct_change"].iloc[0] < 0

    def test_default_deltas_produce_valid_trades_for_put_spreads_butterflies_condors(
        self,
    ):
        """Calling put spreads, butterflies, and condors with default deltas must produce non-empty valid trades."""
        exp = "2020-02-01"
        strikes_calls = [
            (90.0, 0.80, 12.0, 13.0),
            (95.0, 0.70, 8.0, 9.0),
            (100.0, 0.50, 5.0, 5.5),
            (105.0, 0.30, 2.5, 2.7),
            (110.0, 0.20, 1.0, 1.1),
        ]
        strikes_puts = [
            (90.0, -0.20, 1.0, 1.1),
            (95.0, -0.30, 2.5, 2.7),
            (100.0, -0.50, 5.0, 5.5),
            (105.0, -0.70, 8.0, 9.0),
            (110.0, -0.80, 12.0, 13.0),
        ]
        rows = []
        for strike, delta, entry_mid, exit_mid in strikes_calls:
            rows.append(
                _make_option_row(
                    "SPX",
                    "call",
                    exp,
                    "2020-01-02",
                    strike,
                    entry_mid - 0.1,
                    entry_mid + 0.1,
                    delta,
                )
            )
            rows.append(
                _make_option_row(
                    "SPX",
                    "call",
                    exp,
                    "2020-02-01",
                    strike,
                    exit_mid - 0.1,
                    exit_mid + 0.1,
                    delta,
                )
            )
        for strike, delta, entry_mid, exit_mid in strikes_puts:
            rows.append(
                _make_option_row(
                    "SPX",
                    "put",
                    exp,
                    "2020-01-02",
                    strike,
                    entry_mid - 0.1,
                    entry_mid + 0.1,
                    delta,
                )
            )
            rows.append(
                _make_option_row(
                    "SPX",
                    "put",
                    exp,
                    "2020-02-01",
                    strike,
                    exit_mid - 0.1,
                    exit_mid + 0.1,
                    delta,
                )
            )
        df = pd.DataFrame(rows)

        for fn in (
            op.long_put_spread,
            op.short_put_spread,
            op.long_call_butterfly,
            op.long_put_butterfly,
            op.long_call_condor,
            op.long_put_condor,
        ):
            res = fn(df, max_entry_dte=45, exit_dte=0, raw=True)
            assert len(res) > 0, f"{fn.__name__} returned empty with default deltas"

    def test_synthetic_protective_put_allows_otm_put_below_deep_itm_call(self):
        """Synthetic protective_put must allow an OTM put strike below the deep-ITM call strike."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Deep ITM call at strike 90 (delta 0.90), OTM put at strike 85 (delta -0.15)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 90.0, 11.9, 12.1, 0.90
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 85.0, 0.9, 1.1, -0.15
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 90.0, 14.9, 15.1, 0.95
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 85.0, 0.1, 0.2, -0.02
                ),
            ]
        )
        res = op.protective_put(
            df,
            max_entry_dte=45,
            exit_dte=0,
            leg1_delta=_tr(0.90),
            leg2_delta=_tr(0.15),
            raw=True,
        )
        assert len(res) == 1
        assert res["strike_leg1"].iloc[0] == 90.0
        assert res["strike_leg2"].iloc[0] == 85.0

    def test_collar_rejects_inverted_put_call_strikes(self):
        """Both synthetic and stock-backed collars must reject put_strike >= call_strike."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                # Deep ITM call at 80, short call at 95, long put at 105 (inverted: put 105 > call 95!)
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 80.0, 20.9, 21.1, 0.90
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 95.0, 6.9, 7.1, 0.30
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 105.0, 6.9, 7.1, -0.30
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 80.0, 22.9, 23.1, 0.95
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 95.0, 7.9, 8.1, 0.40
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 105.0, 4.9, 5.1, -0.25
                ),
            ]
        )
        res_synth = op.collar(
            df,
            max_entry_dte=45,
            exit_dte=0,
            leg1_delta=_tr(0.90),
            leg2_delta=_tr(0.30),
            leg3_delta=_tr(0.30),
            raw=True,
        )
        assert len(res_synth) == 0

        sd = pd.DataFrame(
            {
                "underlying_symbol": ["SPX", "SPX"],
                "quote_date": pd.to_datetime(["2020-01-02", "2020-02-01"]),
                "close": [100.0, 102.0],
            }
        )
        res_stock = op.collar(
            df,
            stock_data=sd,
            max_entry_dte=45,
            exit_dte=0,
            leg2_delta=_tr(0.30),
            leg3_delta=_tr(0.30),
            raw=True,
        )
        assert len(res_stock) == 0

    def test_stock_backed_strategies_reject_early_exit_params(self):
        """Passing stop_loss/take_profit/max_hold_days with stock_data must raise NotImplementedError."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 105.0, 2.0, 2.2, 0.30
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 105.0, 1.0, 1.2, 0.20
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 95.0, 2.0, 2.2, -0.30
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 95.0, 1.0, 1.2, -0.20
                ),
            ]
        )
        sd = pd.DataFrame(
            {
                "underlying_symbol": ["SPX", "SPX"],
                "quote_date": pd.to_datetime(["2020-01-02", "2020-02-01"]),
                "close": [100.0, 102.0],
            }
        )
        with pytest.raises(NotImplementedError):
            op.covered_call(df, stock_data=sd, stop_loss=-0.50, raw=True)
        with pytest.raises(NotImplementedError):
            op.collar(df, stock_data=sd, take_profit=0.50, raw=True)

    def test_stock_data_multiindex_and_duplicate_dates(self):
        """Multi-ticker yfinance MultiIndex stock_data and duplicate dates must be normalized cleanly."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 105.0, 2.0, 2.2, 0.30
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 105.0, 1.0, 1.2, 0.20
                ),
            ]
        )
        idx = pd.to_datetime(["2020-01-02", "2020-02-01", "2020-02-01"])
        idx.name = "Date"
        cols = pd.MultiIndex.from_tuples(
            [("Close", "SPX"), ("Close", "QQQ"), ("Volume", "SPX"), ("Volume", "QQQ")]
        )
        yf_df = pd.DataFrame(
            [
                [100.0, 200.0, 1000, 2000],
                [103.0, 205.0, 1000, 2000],
                [103.0, 205.0, 1000, 2000],
            ],
            index=idx,
            columns=cols,
        )
        res = op.covered_call(
            df, stock_data=yf_df, max_entry_dte=45, exit_dte=0, raw=True
        )
        assert len(res) == 1


# =============================================================================
# 4. Simulator Regressions
# =============================================================================


class TestSimulatorRegressions:
    def test_max_drawdown_includes_initial_capital_and_all_loss_max_win_zero(self):
        """A losing first trade must register drawdown from initial capital and max_win == 0.0."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 100.0, 9.9, 10.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 100.0, 4.9, 5.1, 0.10
                ),
            ]
        )
        sim = simulate(
            df,
            op.long_calls,
            capital=10_000.0,
            quantity=1,
            max_entry_dte=45,
            exit_dte=0,
        )
        assert sim.summary["total_trades"] == 1
        assert sim.summary["total_pnl"] < 0
        assert sim.summary["max_drawdown"] < 0.0
        assert sim.summary["max_win"] == 0.0
        assert sim.summary["max_loss"] < 0.0

    def test_build_trade_log_sorts_equity_by_exit_date(self):
        """With overlapping concurrent positions, trade_log equity must be ordered by exit_date."""
        df = pd.DataFrame(
            [
                # Trade A: enters Jan 2, exits Feb 20 (later exit)
                _make_option_row(
                    "SPX", "call", "2020-02-20", "2020-01-02", 100.0, 4.9, 5.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", "2020-02-20", "2020-02-20", 100.0, 6.9, 7.1, 0.40
                ),
                # Trade B: enters Jan 5, exits Jan 25 (earlier exit!)
                _make_option_row(
                    "SPX", "call", "2020-01-25", "2020-01-05", 100.0, 4.9, 5.1, 0.30
                ),
                _make_option_row(
                    "SPX", "call", "2020-01-25", "2020-01-25", 100.0, 2.9, 3.1, 0.20
                ),
            ]
        )
        sim = simulate(
            df,
            op.long_calls,
            capital=100_000.0,
            quantity=1,
            max_positions=2,
            max_entry_dte=60,
            exit_dte=0,
        )
        assert len(sim.trade_log) == 2
        assert sim.trade_log["exit_date"].is_monotonic_increasing

    def test_multi_symbol_same_expiration_concurrent(self):
        """Two different symbols sharing the same expiration must be allowed concurrently."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPY", "call", exp, "2020-01-02", 300.0, 4.9, 5.1, 0.30
                ),
                _make_option_row(
                    "SPY", "call", exp, "2020-02-01", 300.0, 6.9, 7.1, 0.40
                ),
                _make_option_row(
                    "QQQ", "call", exp, "2020-01-02", 200.0, 4.9, 5.1, 0.30
                ),
                _make_option_row(
                    "QQQ", "call", exp, "2020-02-01", 200.0, 6.9, 7.1, 0.40
                ),
            ]
        )
        sim = simulate(
            df,
            op.long_calls,
            capital=100_000.0,
            quantity=1,
            max_positions=2,
            max_entry_dte=45,
            exit_dte=0,
        )
        assert sim.summary["total_trades"] == 2
        assert set(sim.trade_log["underlying_symbol"]) == {"SPY", "QQQ"}

    def test_partial_short_strategy_preserves_pnl_sign(self):
        """functools.partial(short_puts, ...) must be recognized as a short single-leg strategy."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPX", "put", exp, "2020-01-02", 95.0, 4.9, 5.1, -0.30
                ),
                _make_option_row(
                    "SPX", "put", exp, "2020-02-01", 95.0, 0.9, 1.1, -0.05
                ),
            ]
        )
        direct = simulate(df, op.short_puts, max_entry_dte=45, exit_dte=0)
        wrapped = simulate(
            df, functools.partial(op.short_puts, max_entry_dte=45), exit_dte=0
        )
        assert direct.summary["total_pnl"] > 0
        assert wrapped.summary["total_pnl"] == pytest.approx(
            direct.summary["total_pnl"]
        )

    def test_calendar_default_exit_dte_is_seven(self):
        """simulate() with a calendar strategy must default exit_dte to 7 (matching calendar API)."""
        front_exp = "2020-01-25"
        back_exp = "2020-02-25"
        df = pd.DataFrame(
            [
                # Entry 2020-01-05 (front DTE=20, back DTE=51)
                _make_option_row(
                    "SPX", "call", front_exp, "2020-01-05", 100.0, 2.0, 2.2, 0.30
                ),
                _make_option_row(
                    "SPX", "call", back_exp, "2020-01-05", 100.0, 5.0, 5.2, 0.30
                ),
                # Exit only at front DTE=7 (2020-01-18) — no DTE=0 row
                _make_option_row(
                    "SPX", "call", front_exp, "2020-01-18", 100.0, 1.0, 1.2, 0.30
                ),
                _make_option_row(
                    "SPX", "call", back_exp, "2020-01-18", 100.0, 4.8, 5.0, 0.30
                ),
            ]
        )
        sim = simulate(
            df,
            op.long_call_calendar,
            front_dte_min=10,
            front_dte_max=30,
            back_dte_min=35,
            back_dte_max=65,
        )
        assert sim.summary["total_trades"] == 1

    def test_all_nan_selectors_do_not_crash(self):
        """Candidate selectors must return the first row rather than raising on all-NaN columns."""
        group = pd.DataFrame(
            {
                "delta": [np.nan, np.nan],
                "entry": [np.nan, np.nan],
                "expiration": pd.to_datetime(["2020-02-01", "2020-02-08"]),
            }
        )
        assert _select_nearest(group).name == 0
        assert _select_highest_premium(group).name == 0
        assert _select_lowest_premium(group).name == 0

    def test_matched_exit_date_preserved_in_simulate_and_portfolio(self):
        """Matched tolerance exit dates must flow into both simulate and simulate_portfolio."""
        exp = "2020-01-20"
        df = pd.DataFrame(
            [
                # AAA: enters Jan 2 at $1.50, only exit quote is Jan 16 (DTE=4) at $0.30
                _make_option_row(
                    "AAA", "call", exp, "2020-01-02", 100.0, 1.50, 1.50, 0.30
                ),
                _make_option_row(
                    "AAA", "call", exp, "2020-01-16", 100.0, 0.30, 0.30, 0.10
                ),
                # BBB: enters Jan 4 at $1.50, exit quote is Jan 15 (DTE=5) at $2.00
                _make_option_row(
                    "BBB", "call", exp, "2020-01-04", 100.0, 1.50, 1.50, 0.30
                ),
                _make_option_row(
                    "BBB", "call", exp, "2020-01-15", 100.0, 2.00, 2.00, 0.50
                ),
            ]
        )

        sim = op.simulate(
            df,
            op.long_calls,
            capital=100.0,
            quantity=1,
            max_positions=2,
            multiplier=100,
            exit_dte=5,
            exit_dte_tolerance=1,
            slippage="mid",
        )
        port = op.simulate_portfolio(
            [
                {
                    "data": df,
                    "strategy": op.long_calls,
                    "weight": 1.0,
                    "quantity": 1,
                    "max_positions": 2,
                    "multiplier": 100,
                    "exit_dte": 5,
                    "exit_dte_tolerance": 1,
                    "slippage": "mid",
                }
            ],
            capital=100.0,
        )

        for result in (sim, port):
            log = result.trade_log
            assert len(log) == 2
            assert list(log["underlying_symbol"]) == ["BBB", "AAA"]
            assert list(pd.to_datetime(log["exit_date"])) == [
                pd.Timestamp("2020-01-15"),
                pd.Timestamp("2020-01-16"),
            ]
            assert list(log["equity"]) == pytest.approx([150.0, 30.0])

    def test_early_exit_takes_precedence_over_matched_exit_date(self):
        """Explicit early-exit dates must override matched tolerance exit dates."""
        exp = "2020-01-20"
        df = pd.DataFrame(
            [
                # AAA: enters Jan 2 at $2.00, stops out Jan 10 at $0.50 (-75%), matched exit Jan 16
                _make_option_row(
                    "AAA", "call", exp, "2020-01-02", 100.0, 2.00, 2.00, 0.30
                ),
                _make_option_row(
                    "AAA", "call", exp, "2020-01-10", 100.0, 0.50, 0.50, 0.15
                ),
                _make_option_row(
                    "AAA", "call", exp, "2020-01-16", 100.0, 1.80, 1.80, 0.25
                ),
                # BBB: enters Jan 4 at $2.00, no stop on Jan 10 ($2.10), exits on matched Jan 16 ($2.40)
                _make_option_row(
                    "BBB", "call", exp, "2020-01-04", 100.0, 2.00, 2.00, 0.30
                ),
                _make_option_row(
                    "BBB", "call", exp, "2020-01-10", 100.0, 2.10, 2.10, 0.35
                ),
                _make_option_row(
                    "BBB", "call", exp, "2020-01-16", 100.0, 2.40, 2.40, 0.45
                ),
            ]
        )
        entry_dates = pd.DataFrame(
            {
                "underlying_symbol": ["AAA", "BBB"],
                "quote_date": pd.to_datetime(["2020-01-02", "2020-01-04"]),
            }
        )
        sim = op.simulate(
            df,
            op.long_calls,
            capital=10_000.0,
            quantity=1,
            max_positions=2,
            multiplier=100,
            exit_dte=5,
            exit_dte_tolerance=1,
            entry_dates=entry_dates,
            stop_loss=-0.50,
            slippage="mid",
        )
        log = sim.trade_log
        assert len(log) == 2
        assert list(log["underlying_symbol"]) == ["AAA", "BBB"]
        assert list(pd.to_datetime(log["exit_date"])) == [
            pd.Timestamp("2020-01-10"),
            pd.Timestamp("2020-01-16"),
        ]
        assert list(log["exit_type"]) == ["stop_loss", "expiration"]

    def test_custom_strategy_without_matched_exit_date_uses_fallback(self):
        """Custom strategies omitting matched exit dates must keep the expiration - exit_dte fallback."""

        def _custom_strategy(_data: pd.DataFrame, *, raw: bool = True, **_kwargs):
            assert raw is True
            return pd.DataFrame(
                {
                    "underlying_symbol": ["AAA"],
                    "quote_date_entry": [pd.Timestamp("2020-01-02")],
                    "option_type": ["call"],
                    "expiration": [pd.Timestamp("2020-01-20")],
                    "dte_entry": [18],
                    "strike": [100.0],
                    "entry": [1.50],
                    "exit": [2.00],
                    "pct_change": [1.0 / 3.0],
                    "delta_entry": [0.30],
                }
            )

        dummy = pd.DataFrame({"x": [1]})
        sim_dte5 = op.simulate(dummy, _custom_strategy, capital=1_000.0, exit_dte=5)
        assert pd.Timestamp(sim_dte5.trade_log["exit_date"].iloc[0]) == pd.Timestamp(
            "2020-01-15"
        )

        sim_dte0 = op.simulate(dummy, _custom_strategy, capital=1_000.0, exit_dte=0)
        assert pd.Timestamp(sim_dte0.trade_log["exit_date"].iloc[0]) == pd.Timestamp(
            "2020-01-20"
        )


# =============================================================================
# 5. Signals Regressions
# =============================================================================


class TestSignalsRegressions:
    def test_atr_signals_are_strictly_causal(self):
        """Mutating future bars must NEVER alter past atr_above/atr_below signals."""
        rng = np.random.default_rng(42)
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=60))
        sd1 = _make_stock_series(closes.tolist())
        sd2 = sd1.copy()
        # Massive volatility explosion in bars 40..59
        sd2.loc[40:, "high"] = sd2.loc[40:, "close"] * 1.50
        sd2.loc[40:, "low"] = sd2.loc[40:, "close"] * 0.50

        for sig in (
            atr_above(period=14, multiplier=1.0),
            atr_below(period=14, multiplier=1.0),
        ):
            m1 = sig(sd1)
            m2 = sig(sd2)
            pd.testing.assert_series_equal(m1.iloc[:40], m2.iloc[:40])

    def test_donchian_breakout_signals_fire(self):
        """donchian_above_upper and donchian_below_lower must fire on true breakouts."""
        up_closes = [100.0] * 15 + [100.0 + i * 2.0 for i in range(1, 15)]
        down_closes = [100.0] * 15 + [100.0 - i * 2.0 for i in range(1, 15)]
        assert donchian_above_upper(10, 10)(_make_stock_series(up_closes)).any()
        assert donchian_below_lower(10, 10)(_make_stock_series(down_closes)).any()

    def test_squeeze_on_and_squeeze_off_fire_and_are_disjoint(self):
        """squeeze_on and squeeze_off must read SQZ_ON/SQZ_OFF flags and never overlap."""
        closes = (
            [100.0 + ((-1) ** i) * 5.0 for i in range(30)]
            + [100.0 + ((-1) ** i) * 0.05 for i in range(30)]
            + [100.0 + i * 3.0 for i in range(1, 25)]
        )
        sd = _make_stock_series(closes, spread=0.005)
        sd.loc[:29, "high"] = sd.loc[:29, "close"] * 1.04
        sd.loc[:29, "low"] = sd.loc[:29, "close"] * 0.96
        sd.loc[30:59, "high"] = sd.loc[30:59, "close"] * 1.001
        sd.loc[30:59, "low"] = sd.loc[30:59, "close"] * 0.999
        sd.loc[60:, "high"] = sd.loc[60:, "close"] * 1.04
        sd.loc[60:, "low"] = sd.loc[60:, "close"] * 0.96

        on = squeeze_on()(sd)
        off = squeeze_off()(sd)
        assert not (on & off).any()
        assert on.any()
        assert off.any()

    def test_tsi_smi_stoch_fisher_signals(self):
        """TSI, SMI, Stoch(d_period), and Fisher(high/low) must compute properly."""
        closes = [120.0 - i * 1.2 for i in range(40)] + [
            72.0 + i * 1.8 for i in range(45)
        ]
        sd = _make_stock_series(closes, spread=0.02)

        assert tsi_cross_above(long=25, short=13, signal_period=7)(sd).any()
        assert smi_cross_above(fast=5, slow=20, signal_period=5)(sd).any()

        # stoch_below must actually use d_period (smooth_k=d_period)
        s_d1 = stoch_below(k_period=14, d_period=1, threshold=30)(sd)
        s_d5 = stoch_below(k_period=14, d_period=5, threshold=30)(sd)
        assert not s_d1.equals(s_d5)

        # fisher_cross_above must use real high/low (asymmetric so hl2 != close)
        sd_asym = sd.copy()
        sd_asym.loc[35:50, "high"] = sd_asym.loc[35:50, "close"] * 1.35
        sd_asym.loc[35:50, "low"] = sd_asym.loc[35:50, "close"] * 0.99
        f_ohlcv = fisher_cross_above(period=9)(sd_asym)
        f_close = fisher_cross_above(period=9)(sd_asym.drop(columns=["high", "low"]))
        assert f_ohlcv.any()
        assert not f_ohlcv.equals(f_close)

    def test_direction_signals_do_not_fire_on_first_bars(self):
        """PSAR and Supertrend signals must require a genuine direction flip, not fire on bars 0-1."""
        closes = [100.0 + i * 0.5 for i in range(30)]
        sd = _make_stock_series(closes)
        for sig in (psar_buy(), psar_sell(), supertrend_buy(), supertrend_sell()):
            mask = sig(sd)
            assert not mask.iloc[0]
            assert not mask.iloc[1]

    def test_signal_dates_preserves_per_strike_iv_rank_and_combinators(self):
        """signal_dates() must keep per-strike rows for iv_rank_above and combinators wrapping it."""
        dates = pd.bdate_range("2020-01-02", periods=30)
        rows = []
        for i, d in enumerate(dates):
            exp = d + pd.Timedelta(days=30)
            iv_val = 0.15 + 0.01 * i
            for strike, delta, iv in (
                (100.0, 0.50, iv_val),
                (120.0, 0.10, iv_val + 0.3),
            ):
                row = _make_option_row(
                    "SPX",
                    "call",
                    exp,
                    d,
                    strike,
                    4.9,
                    5.1,
                    delta,
                    implied_volatility=iv,
                )
                row["underlying_price"] = 100.0
                rows.append(row)
        opt_df = pd.DataFrame(rows)
        sig = iv_rank_above(threshold=0.5, window=10)
        combined = and_signals(sig, day_of_week(0))
        assert sig.requires_per_strike is True
        assert combined.requires_per_strike is True

        dates_iv = signal_dates(opt_df, sig)
        dates_combined = signal_dates(opt_df, combined)
        assert len(dates_iv) > 0
        assert len(dates_combined) > 0
        assert len(dates_combined) <= len(dates_iv)

    def test_unsorted_quote_date_and_zero_price_safety(self):
        """Direct signal evaluation must sort unsorted quote_date rows and guard zero-price division."""
        closes = [100.0 + i * 1.0 for i in range(25)]
        sd_sorted = _make_stock_series(closes)
        sd_shuffled = sd_sorted.iloc[::-1].copy()

        m_sorted = daily_return_above(0.5)(sd_sorted)
        m_shuffled = daily_return_above(0.5)(sd_shuffled)
        pd.testing.assert_series_equal(m_sorted, m_shuffled.sort_index())

        zero_sd = _make_stock_series([0.0, 0.0, 10.0, 0.0, 5.0])
        for sig in (
            daily_return_above(0.5),
            daily_return_below(-0.5),
            gap_up(0.5),
            drawdown_from_high(period=3, pct=10.0),
            rally_from_low(period=3, pct=10.0),
        ):
            res = sig(zero_sd)
            assert res.dtype == bool
            assert len(res) == len(zero_sd)
        assert not daily_return_above(50.0)(zero_sd).iloc[2]
        assert not rally_from_low(period=3, pct=50.0)(zero_sd).iloc[2]

    def test_butterfly_rejects_broken_wing_at_large_strikes(self):
        """Butterfly rule must use rtol=0.0 so large strikes do not admit broken-wing spreads."""
        df = pd.DataFrame(
            {
                "strike_leg1": [1000.0],
                "strike_leg2": [2000.0],
                "strike_leg3": [3000.01],
            }
        )
        filtered = _rule_butterfly_strikes(df, [(1, None), (-1, None), (1, None)])
        assert filtered.empty

    def test_omega_ratio_object_series_with_pd_na(self):
        """Object Series containing pd.NA must drop missing entries before float coercion."""
        returns = pd.Series([0.01, pd.NA, -0.005, 0.02], dtype=object)
        val = omega_ratio(returns, risk_free_rate=0.0, trading_days=252)
        assert np.isfinite(val)
        assert val > 1.0

    def test_calendar_tolerance_snaps_per_symbol(self):
        """Another symbol's exit date must not shadow a valid tolerance exit for a different symbol."""
        front_exp = "2020-01-25"
        back_exp = "2020-02-25"
        rows = [
            # SPY entry Jan 10, exit signal only on Jan 16 (front DTE=9)
            _make_option_row(
                "SPY", "call", front_exp, "2020-01-10", 100.0, 2.0, 2.2, 0.30
            ),
            _make_option_row(
                "SPY", "call", back_exp, "2020-01-10", 100.0, 4.0, 4.2, 0.30
            ),
            _make_option_row(
                "SPY", "call", front_exp, "2020-01-16", 100.0, 1.0, 1.2, 0.30
            ),
            _make_option_row(
                "SPY", "call", back_exp, "2020-01-16", 100.0, 3.8, 4.0, 0.30
            ),
            # QQQ entry Jan 10, exit signal on Jan 18 (front DTE=7, exact target)
            _make_option_row(
                "QQQ", "call", front_exp, "2020-01-10", 200.0, 2.0, 2.2, 0.30
            ),
            _make_option_row(
                "QQQ", "call", back_exp, "2020-01-10", 200.0, 4.0, 4.2, 0.30
            ),
            _make_option_row(
                "QQQ", "call", front_exp, "2020-01-18", 200.0, 0.8, 1.0, 0.30
            ),
            _make_option_row(
                "QQQ", "call", back_exp, "2020-01-18", 200.0, 3.9, 4.1, 0.30
            ),
        ]
        df = pd.DataFrame(rows)
        exit_dates = pd.DataFrame(
            {
                "underlying_symbol": ["SPY", "QQQ"],
                "quote_date": pd.to_datetime(["2020-01-16", "2020-01-18"]),
            }
        )
        res = op.long_call_calendar(
            df,
            front_dte_min=10,
            front_dte_max=20,
            back_dte_min=30,
            back_dte_max=60,
            exit_dte=7,
            exit_dte_tolerance=3,
            exit_dates=exit_dates,
            raw=True,
        )
        assert set(res["underlying_symbol"]) == {"SPY", "QQQ"}
        by_sym = res.set_index("underlying_symbol")["exit_date"]
        assert pd.Timestamp(by_sym["SPY"]) == pd.Timestamp("2020-01-16")
        assert pd.Timestamp(by_sym["QQQ"]) == pd.Timestamp("2020-01-18")

    def test_iv_rank_direct_call_normalizes_timestamp_and_tz_aware_dates(self):
        """Direct iv_rank_above calls with tz-aware market-close timestamps must match ATM keys."""
        dates = pd.date_range("2020-01-02 16:00:00", periods=20, freq="B", tz="UTC")
        rows = []
        for i, d in enumerate(dates):
            exp = d + pd.Timedelta(days=30)
            row = _make_option_row(
                "SPX",
                "call",
                exp,
                d,
                100.0,
                4.9,
                5.1,
                0.50,
                implied_volatility=0.15 + 0.01 * i,
            )
            row["underlying_price"] = 100.0
            rows.append(row)
        opt_df = pd.DataFrame(rows, index=range(100, 100 + len(rows)))
        mask = iv_rank_above(threshold=0.5, window=10)(opt_df)
        assert mask.index.equals(opt_df.index)
        assert mask.any()

    def test_iv_rank_combined_with_historical_price_signal_counts_dates_not_contracts(
        self,
    ):
        """Combining iv_rank_above with sma_above(20) on 10 dates x 5 strikes must NOT warm up SMA."""
        dates = pd.bdate_range("2020-01-02", periods=10)
        rows = []
        for i, d in enumerate(dates):
            exp = d + pd.Timedelta(days=30)
            price = 100.0 + i * 2.0
            for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
                row = _make_option_row(
                    "SPX",
                    "call",
                    exp,
                    d,
                    strike,
                    4.9,
                    5.1,
                    0.50,
                    implied_volatility=0.15 + 0.02 * i,
                )
                row["close"] = price
                rows.append(row)
        opt_df = pd.DataFrame(rows)

        combined_fn = and_signals(iv_rank_above(threshold=0.5, window=5), sma_above(20))
        combined_op = signal(iv_rank_above(threshold=0.5, window=5)) & signal(
            sma_above(20)
        )
        assert signal_dates(opt_df, combined_fn).empty
        assert signal_dates(opt_df, combined_op).empty

        # Sanity check: with period=5 (<= 10 trading dates), both MUST fire
        combined_short = and_signals(
            iv_rank_above(threshold=0.5, window=5), sma_above(5)
        )
        assert len(signal_dates(opt_df, combined_short)) > 0

    def test_checks_reject_complex_dtype(self):
        """Schema validation must reject complex-valued numeric columns."""
        df = pd.DataFrame(
            {
                "underlying_symbol": ["SPX"],
                "option_type": ["call"],
                "expiration": pd.to_datetime(["2020-02-01"]),
                "quote_date": pd.to_datetime(["2020-01-02"]),
                "strike": [100.0 + 1j],
                "bid": [5.0],
                "ask": [5.2],
                "delta": [0.30],
            }
        )
        with pytest.raises(ValueError, match="does not match expected types"):
            _run_checks({}, df)

    def test_unsorted_signals_with_duplicate_index_labels(self):
        """Unsorted quote_date inputs with duplicate index labels must restore order by position."""
        closes = [100.0, 102.0, 104.0, 106.0, 108.0]
        sd_sorted = _make_stock_series(closes)
        sd_unsorted_dup = sd_sorted.iloc[[3, 1, 4, 0, 2]].copy()
        sd_unsorted_dup.index = pd.Index([0, 0, 1, 1, 2])

        sig = daily_return_above(1.0)
        sus = sustained(daily_return_above(1.0), days=2)

        res_sorted = sig(sd_sorted).to_numpy()
        res_dup = sig(sd_unsorted_dup)
        assert res_dup.index.equals(sd_unsorted_dup.index)
        assert list(res_dup.to_numpy()) == [res_sorted[i] for i in [3, 1, 4, 0, 2]]

        sus_sorted = sus(sd_sorted).to_numpy()
        sus_dup = sus(sd_unsorted_dup)
        assert sus_dup.index.equals(sd_unsorted_dup.index)
        assert list(sus_dup.to_numpy()) == [sus_sorted[i] for i in [3, 1, 4, 0, 2]]

    def test_missing_symbol_excluded_from_single_symbol_fast_path(self):
        """Rows with missing underlying_symbol must not be grouped into the first symbol's fast path."""
        df = pd.DataFrame(
            {
                "underlying_symbol": pd.Series(["SPX", pd.NA], dtype="string"),
                "quote_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "close": [100.0, 110.0],
            }
        )
        mask = daily_return_above(5.0)(df)
        assert list(mask.to_numpy()) == [False, False]

    def test_stock_data_multiindex_unnamed_datetime_index(self):
        """Multi-ticker yfinance MultiIndex stock_data with unnamed DatetimeIndex must normalize cleanly."""
        exp = "2020-02-01"
        df = pd.DataFrame(
            [
                _make_option_row(
                    "SPX", "call", exp, "2020-01-02", 105.0, 2.0, 2.2, 0.30
                ),
                _make_option_row(
                    "SPX", "call", exp, "2020-02-01", 105.0, 1.0, 1.2, 0.20
                ),
            ]
        )
        idx = pd.to_datetime(["2020-01-02", "2020-02-01"])  # unnamed DatetimeIndex
        cols = pd.MultiIndex.from_tuples(
            [("Close", "SPX"), ("Close", "QQQ"), ("Volume", "SPX"), ("Volume", "QQQ")]
        )
        yf_df = pd.DataFrame(
            [[100.0, 200.0, 1000, 2000], [103.0, 205.0, 1000, 2000]],
            index=idx,
            columns=cols,
        )
        res = op.covered_call(
            df, stock_data=yf_df, max_entry_dte=45, exit_dte=0, raw=True
        )
        assert len(res) == 1
