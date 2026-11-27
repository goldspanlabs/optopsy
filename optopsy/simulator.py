"""Lightweight backtest simulation layer for optopsy strategies.

Sits on top of the existing strategy engine. Calls any strategy function with
``raw=True`` to get individual trades, then simulates executing them
chronologically with capital tracking, position limits, and equity curve
generation.

Example::

    import optopsy as op

    data = op.csv_data("SPX_2018.csv")
    result = op.simulate(data, op.short_puts, max_entry_dte=45, exit_dte=14)
    print(result.summary)
    print(result.trade_log)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, Callable, Literal, Union

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationResult:
    """Container for simulation output.

    Attributes:
        trade_log: One row per completed trade with P&L details.
        equity_curve: Indexed by exit date; value is equity after each trade close.
        summary: Flat dict of performance metrics.
    """

    trade_log: pd.DataFrame
    equity_curve: pd.Series
    summary: dict[str, Any]


@dataclass(frozen=True)
class PortfolioResult:
    """Container for multi-leg portfolio simulation output.

    Attributes:
        trade_log: Combined trade log across all legs, with a ``leg`` column.
        equity_curve: Daily portfolio equity (sum of per-leg equity curves).
        summary: Portfolio-level performance metrics (same keys as single sim).
        leg_results: Per-leg :class:`SimulationResult` keyed by leg name.
    """

    trade_log: pd.DataFrame
    equity_curve: pd.Series
    summary: dict[str, Any]
    leg_results: dict[str, SimulationResult]


# ---------------------------------------------------------------------------
# Trade log schema
# ---------------------------------------------------------------------------

_TRADE_LOG_COLUMNS = [
    "trade_id",
    "underlying_symbol",
    "entry_date",
    "exit_date",
    "days_held",
    "expiration",
    "entry_cost",
    "exit_proceeds",
    "quantity",
    "multiplier",
    "dollar_cost",
    "dollar_proceeds",
    "commission",
    "realized_pnl",
    "pct_change",
    "cumulative_pnl",
    "equity",
    "description",
    "exit_type",
]


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------

# Columns that identify a single-leg raw trade
_SINGLE_LEG_COLS = {"entry", "exit", "quote_date_entry"}

# Columns that identify a multi-leg (non-calendar) raw trade
_MULTI_LEG_COST_COLS = {"total_entry_cost", "total_exit_proceeds"}

# Calendar/diagonal strategies have per-leg expirations
_CALENDAR_MARKER = "expiration_leg1"

# Single-leg short strategies — raw output has unsigned option prices (same as
# long), so we negate entry/exit during normalisation to convert to signed cash
# flows: negative entry_cost = credit received, negative exit_proceeds = paid.
_SHORT_SINGLE_LEG = frozenset({"short_calls", "short_puts", "cash_secured_put"})


def _is_single_leg(columns: pd.Index) -> bool:
    return _SINGLE_LEG_COLS.issubset(columns) and "total_entry_cost" not in columns


def _is_calendar(columns: pd.Index) -> bool:
    return _CALENDAR_MARKER in columns


def _scale_commission_for_quantity(
    commission: Any, quantity: int, multiplier: int
) -> Any:
    """Scale variable fees to the simulated lot size.

    Base and minimum fees apply once per side of the simulated trade, so they
    must not be multiplied by the number of contracts. Per-share fees scale
    with the simulated share count (quantity * multiplier).
    """
    if commission is None:
        return commission

    if isinstance(commission, (int, float)) and not isinstance(commission, bool):
        return commission * quantity

    if hasattr(commission, "model_dump"):
        config = commission.model_dump()
    elif isinstance(commission, dict):
        from .types import Commission

        # Coerce accepted fee values (including numeric strings) before scaling.
        config = Commission.model_validate(commission).model_dump()
    else:
        return commission

    if "per_contract" in config:
        config["per_contract"] *= quantity
    if "per_share" in config:
        config["per_share"] *= quantity * multiplier / 100
    return config


def _adjust_candidate_returns(raw: pd.DataFrame, lot_size: int) -> pd.DataFrame:
    """Recompute candidate returns after applying lot-sized commission.

    Strategy functions express prices per standard 100-share contract and
    deduct commission in those same units. Recover gross per-contract P&L,
    then apply the simulated lot size and dollar commission before selectors
    inspect ``pct_change``.
    """
    if "total_commission" not in raw.columns or "pct_change" not in raw.columns:
        return raw

    entry_col = "total_entry_cost" if "total_entry_cost" in raw.columns else "entry"
    if entry_col not in raw.columns:
        return raw

    basis = pd.to_numeric(raw[entry_col], errors="coerce").abs()
    fee = pd.to_numeric(raw["total_commission"], errors="coerce").fillna(0.0)
    pct_change = pd.to_numeric(raw["pct_change"], errors="coerce")
    gross_pnl_per_contract = pct_change * basis + fee / 100
    denominator = basis * lot_size
    raw["pct_change"] = np.where(
        denominator > 0,
        (gross_pnl_per_contract * lot_size - fee) / denominator,
        np.nan,
    )
    return raw


# ---------------------------------------------------------------------------
# Trade normalisation
# ---------------------------------------------------------------------------


def _normalise_trades(
    raw: pd.DataFrame, *, is_short_single: bool = False, exit_dte: int = 0
) -> pd.DataFrame:
    """Map raw strategy output to a uniform schema for the simulation loop.

    Returns a DataFrame with columns:
        entry_date, exit_date, expiration, underlying_symbol,
        entry_cost, exit_proceeds, pct_change, description, exit_type

    After normalisation, ``entry_cost`` and ``exit_proceeds`` are *signed cash
    flows* using Optopsy's convention: negative values are credits received
    (cash inflow), positive values are debits/amounts paid (cash outflow).
    For multi-leg strategies the raw output already encodes this; for
    single-leg short strategies, the unsigned option prices are negated.

    When *exit_dte* > 0, exit_date is set to ``expiration - exit_dte`` days
    instead of expiration itself.  When an early exit (stop-loss/take-profit)
    triggers before that date, the early exit date takes priority.
    """
    cols = raw.columns

    if _is_single_leg(cols):
        df = _normalise_single_leg(raw, negate=is_short_single)
    elif _is_calendar(cols):
        df = _normalise_calendar(raw)
    else:
        # Multi-leg (spreads, butterflies, condors, straddles, strangles)
        df = _normalise_multi_leg(raw)

    # Propagate exit_type from strategy output (set by early exit logic)
    if "exit_type" in raw.columns:
        df["exit_type"] = raw["exit_type"].values
    else:
        df["exit_type"] = "expiration"

    # Prefer explicit exit date columns when present in raw output; otherwise
    # derive from expiration - exit_dte. Early exit dates always take priority.
    explicit_exit_col = next(
        (
            col
            for col in ("quote_date_exit", "quote_date_exit_leg1", "exit_date")
            if col in raw.columns
        ),
        None,
    )
    fallback_exit = (
        pd.to_datetime(df["expiration"] - pd.Timedelta(days=exit_dte))
        if exit_dte > 0
        else pd.to_datetime(df["exit_date"])
    )
    if explicit_exit_col is not None:
        matched_exit = pd.to_datetime(raw[explicit_exit_col].to_numpy())
        base_exit = pd.to_datetime(
            np.where(
                pd.notna(matched_exit),
                matched_exit.to_numpy(),
                fallback_exit.to_numpy(),
            )
        )
    else:
        base_exit = fallback_exit

    if "_early_exit_date" in raw.columns:
        early = pd.to_datetime(raw["_early_exit_date"].to_numpy())
        has_early = pd.notna(early)
        df["exit_date"] = pd.to_datetime(
            np.where(has_early, early.to_numpy(), base_exit.to_numpy())
        )
    else:
        df["exit_date"] = base_exit.to_numpy()

    df["commission"] = (
        raw["total_commission"].fillna(0.0).values
        if "total_commission" in raw.columns
        else 0.0
    )
    return df


def _normalise_single_leg(raw: pd.DataFrame, *, negate: bool = False) -> pd.DataFrame:
    entry_date = _resolve_entry_date(raw)
    expiration = _resolve_expiration(raw)

    # Derive exit date: if dte_entry exists we can compute exit_dte from the
    # strategy params, but the raw output doesn't carry exit_dte directly.
    # For single-leg trades the exit is at expiration (exit_dte=0) or at
    # expiration minus the remaining DTE.  Since we don't have exit quote_date
    # in the raw output, approximate exit_date = expiration.
    exit_date = expiration

    option_type = raw.get("option_type", pd.Series([""] * len(raw), index=raw.index))
    strike = raw.get("strike", pd.Series([np.nan] * len(raw), index=raw.index))
    desc = option_type.astype(str) + " " + strike.astype(str)  # type: ignore[operator]

    # For short single-leg strategies, negate prices to signed cash flows:
    # selling at entry = credit (negative cost), buying back at exit = debit
    # (negative proceeds).  This aligns with multi-leg convention where
    # total_entry_cost < 0 means credit.
    sign = -1 if negate else 1
    entry_cost = raw["entry"] * sign
    exit_proceeds = raw["exit"] * sign

    return pd.DataFrame(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiration": expiration,
            "underlying_symbol": raw["underlying_symbol"],
            "entry_cost": entry_cost,
            "exit_proceeds": exit_proceeds,
            "pct_change": raw["pct_change"],
            "description": desc,
        }
    )


def _normalise_multi_leg(raw: pd.DataFrame) -> pd.DataFrame:
    entry_date = _resolve_entry_date(raw)
    expiration = _resolve_expiration(raw)
    exit_date = expiration

    # Build a human-readable description from available strike/type columns
    desc_parts = [
        raw[f"option_type_leg{i}"].astype(str) + " " + raw[f"strike_leg{i}"].astype(str)  # type: ignore[operator]
        for i in range(1, 5)
        if f"option_type_leg{i}" in raw.columns and f"strike_leg{i}" in raw.columns
    ]
    desc = (
        reduce(lambda a, b: a + "/" + b, desc_parts)
        if desc_parts
        else pd.Series(["multi-leg"] * len(raw), index=raw.index)
    )

    return pd.DataFrame(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiration": expiration,
            "underlying_symbol": raw["underlying_symbol"],
            "entry_cost": raw["total_entry_cost"],
            "exit_proceeds": raw["total_exit_proceeds"],
            "pct_change": raw["pct_change"],
            "description": desc,
        }
    )


def _normalise_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    entry_date = _resolve_entry_date(raw)

    # Exit date = front expiration (leg1) — the spread is typically closed
    # near front expiration
    exit_date = pd.to_datetime(raw["expiration_leg1"])
    expiration = exit_date

    strike_col = "strike" if "strike" in raw.columns else "strike_leg1"
    strike = raw.get(strike_col, pd.Series([np.nan] * len(raw), index=raw.index))
    option_type = raw.get("option_type", pd.Series([""] * len(raw), index=raw.index))
    desc = "cal " + option_type.astype(str) + " " + strike.astype(str)  # type: ignore[operator]

    return pd.DataFrame(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "expiration": expiration,
            "underlying_symbol": raw["underlying_symbol"],
            "entry_cost": raw["total_entry_cost"],
            "exit_proceeds": raw["total_exit_proceeds"],
            "pct_change": raw["pct_change"],
            "description": desc,
        }
    )


# ---------------------------------------------------------------------------
# Built-in selectors
# ---------------------------------------------------------------------------

# Each selector receives a DataFrame of candidate trades for a single entry
# date (in the raw/pre-normalised schema) and returns a single-row Series.


def _select_one(candidates: pd.DataFrame, idx: Any) -> pd.Series:
    """Return a single row from *candidates* at the given index label."""
    result = candidates.loc[idx]
    if isinstance(result, pd.DataFrame):
        return result.iloc[0]
    assert isinstance(result, pd.Series)
    return result


def _select_by_extremum(
    candidates: pd.DataFrame, metric: pd.Series, *, pick_max: bool = False
) -> pd.Series:
    """Select the candidate row minimizing (or maximizing) *metric*, guarding all-NaN."""
    if not metric.notna().any():
        return candidates.iloc[0]
    idx = metric.idxmax() if pick_max else metric.idxmin()
    return _select_one(candidates, idx)


def _select_nearest(candidates: pd.DataFrame) -> pd.Series:
    """Select the trade closest to ATM (delta closest to 0.50)."""
    # Stock-backed strategies (covered call, protective put, collar with stock_data)
    # set delta_entry_leg1 = 1.0 for the stock leg; target the option leg instead.
    if (
        "option_type_leg1" in candidates.columns
        and bool((candidates["option_type_leg1"] == "stock").all())
        and "delta_entry_leg2" in candidates.columns
        and candidates["delta_entry_leg2"].notna().any()
    ):
        return _select_by_extremum(
            candidates, (candidates["delta_entry_leg2"].abs() - 0.50).abs()
        )

    for delta_col in ("delta_entry", "delta_entry_leg1", "delta_leg1"):
        if delta_col in candidates.columns and candidates[delta_col].notna().any():
            return _select_by_extremum(
                candidates, (candidates[delta_col].abs() - 0.50).abs()
            )

    # Fallback: use strike - underlying_price distance
    strike_col = "strike" if "strike" in candidates.columns else None
    if strike_col is None and "strike_leg1" in candidates.columns:
        strike_col = "strike_leg1"

    underlying_col: str | None = "underlying_price_entry"
    if underlying_col not in candidates.columns:
        if "underlying_price_entry_leg1" in candidates.columns:
            underlying_col = "underlying_price_entry_leg1"
        else:
            underlying_col = None

    if strike_col is not None and underlying_col is not None:
        distance = (candidates[strike_col] - candidates[underlying_col]).abs()
        return _select_by_extremum(candidates, distance)

    return candidates.iloc[0]


def _select_highest_premium(candidates: pd.DataFrame) -> pd.Series:
    """Select the trade with the highest credit (most negative entry cost).

    For multi-leg strategies, ``total_entry_cost`` is already signed (negative
    = credit), so ``idxmin`` picks the largest credit.  For single-leg
    strategies, ``entry`` is an unsigned option price, so ``idxmax`` picks the
    highest premium.
    """
    cost_col = _find_cost_col(candidates)
    if cost_col == "entry":
        # Unsigned prices: highest premium = max value
        return _select_by_extremum(candidates, candidates[cost_col], pick_max=True)
    # Signed costs: highest credit = most negative = min value
    return _select_by_extremum(candidates, candidates[cost_col], pick_max=False)


def _select_lowest_premium(candidates: pd.DataFrame) -> pd.Series:
    """Select the trade with the lowest debit (cheapest absolute entry cost)."""
    cost_col = _find_cost_col(candidates)
    return _select_by_extremum(candidates, candidates[cost_col].abs(), pick_max=False)


def _select_first(candidates: pd.DataFrame) -> pd.Series:
    """Select the first candidate (deterministic, for testing)."""
    return candidates.iloc[0]


_BUILTIN_SELECTORS: dict[str, Callable[..., pd.Series]] = {
    "nearest": _select_nearest,
    "highest_premium": _select_highest_premium,
    "lowest_premium": _select_lowest_premium,
    "first": _select_first,
}


def _find_cost_col(df: pd.DataFrame) -> str:
    """Find the entry cost column in the dataframe."""
    if "total_entry_cost" in df.columns:
        return "total_entry_cost"
    return "entry"


# ---------------------------------------------------------------------------
# Entry-date column detection
# ---------------------------------------------------------------------------


def _resolve_entry_date(raw: pd.DataFrame) -> pd.Series:
    """Resolve the entry date from raw strategy output.

    Checks for explicit date columns first, then falls back to deriving
    from expiration - DTE.
    """
    for col in ("quote_date_entry", "quote_date_entry_leg1", "quote_date"):
        if col in raw.columns:
            return pd.to_datetime(raw[col])
    return _derive_entry_date(raw)


def _derive_entry_date(raw: pd.DataFrame) -> pd.Series:
    """Derive entry date from expiration and dte_entry when no date column exists."""
    if "expiration" in raw.columns and "dte_entry" in raw.columns:
        exp = pd.to_datetime(raw["expiration"])
        dte = raw["dte_entry"]
    elif "expiration_leg1" in raw.columns and "dte_entry_leg1" in raw.columns:
        exp = pd.to_datetime(raw["expiration_leg1"])
        dte = raw["dte_entry_leg1"]
    else:
        raise ValueError("Cannot derive entry date: no expiration/dte column found")
    return exp - pd.to_timedelta(dte, unit="D")


def _resolve_expiration(raw: pd.DataFrame) -> pd.Series:
    """Resolve the expiration date from raw strategy output."""
    for col in ("expiration", "expiration_leg1"):
        if col in raw.columns:
            return pd.to_datetime(raw[col])
    raise ValueError(f"Cannot determine expiration: columns={list(raw.columns)}")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


def _compute_summary(trade_log: pd.DataFrame, capital: float) -> dict[str, Any]:
    """Compute summary statistics from the trade log.

    Includes both basic trade statistics and risk-adjusted metrics
    (Sharpe, Sortino, VaR, CVaR, Calmar) from the ``metrics`` module.
    """
    from .metrics import (
        calmar_ratio,
        conditional_value_at_risk,
        omega_ratio,
        sharpe_ratio,
        sortino_ratio,
        tail_ratio,
        value_at_risk,
    )
    from .metrics import (
        max_drawdown as _max_drawdown,
    )
    from .metrics import (
        profit_factor as _profit_factor,
    )
    from .metrics import (
        win_rate as _win_rate,
    )

    _empty_risk = {
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "var_95": 0.0,
        "cvar_95": 0.0,
        "calmar_ratio": 0.0,
        "omega_ratio": 0.0,
        "tail_ratio": 0.0,
    }

    if trade_log.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "avg_days_in_trade": 0.0,
            **_empty_risk,
        }

    pnl = trade_log["realized_pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    # Max drawdown from equity curve (prepend starting capital so losses on
    # the very first trade are captured relative to initial equity).
    equity = trade_log["equity"]
    equity_with_start = (
        np.concatenate(([capital], equity.to_numpy(dtype=float)))
        if capital > 0
        else equity.to_numpy(dtype=float)
    )
    max_dd = _max_drawdown(equity_with_start)

    # Derive true daily returns from the equity curve for annualised
    # metrics (Sharpe, Sortino, Calmar, Omega).  The equity curve is indexed
    # by trade exit dates which are NOT evenly spaced.  Resampling to calendar
    # daily frequency ("D") with forward-fill preserves weekend exit dates and
    # produces 365-day series, so we annualise with annual_days=365.
    _has_dates = "exit_date" in trade_log.columns and "entry_date" in trade_log.columns
    if _has_dates:
        # When multiple trades exit on the same date, use groupby to get
        # the final equity value for each unique exit date. This ensures
        # a unique datetime index required by resample("D").
        equity_daily = trade_log.groupby("exit_date")["equity"].last()
        equity_daily.index = pd.to_datetime(equity_daily.index)
        # Prepend initial capital at (or one day before, for same-day closes)
        # the earliest entry date so the full period and trade 1's return are
        # represented.
        first_entry = pd.to_datetime(trade_log["entry_date"].min())
        seed_date = (
            first_entry - pd.Timedelta(days=1)
            if first_entry in equity_daily.index
            else first_entry
        )
        equity_daily.loc[seed_date] = capital
        equity_daily = equity_daily.sort_index().resample("D").ffill()
        daily_returns = equity_daily.pct_change().dropna()
        annual_days = 365
    else:
        # Fallback for minimal trade logs (e.g. unit tests) without date
        # columns — use inter-trade equity changes.
        daily_returns = equity.pct_change().dropna()
        annual_days = 252

    # Per-trade returns are still correct for VaR/CVaR (no annualisation).
    per_trade_returns = trade_log["pct_change"]

    return {
        "total_trades": len(trade_log),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": _win_rate(pnl),
        "total_pnl": float(pnl.sum()),
        "total_return": float(pnl.sum()) / capital if capital > 0 else 0.0,
        "avg_pnl": float(pnl.mean()),
        "avg_win": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) > 0 else 0.0,
        "max_win": float(wins.max()) if len(wins) > 0 else 0.0,
        "max_loss": float(losses.min()) if len(losses) > 0 else 0.0,
        "profit_factor": _profit_factor(pnl),
        "max_drawdown": max_dd,
        "avg_days_in_trade": float(trade_log["days_held"].mean()),
        "sharpe_ratio": sharpe_ratio(daily_returns, trading_days=annual_days),
        "sortino_ratio": sortino_ratio(daily_returns, trading_days=annual_days),
        "var_95": value_at_risk(per_trade_returns, 0.95),
        "cvar_95": conditional_value_at_risk(per_trade_returns, 0.95),
        "calmar_ratio": calmar_ratio(daily_returns, trading_days=annual_days),
        "omega_ratio": omega_ratio(daily_returns, trading_days=annual_days),
        "tail_ratio": tail_ratio(daily_returns),
    }


def _filter_trades(
    trades: pd.DataFrame,
    max_positions: int,
) -> pd.DataFrame:
    """Filter trades based on position limits and overlap rules.

    Light O(n) scan over trades sorted by entry_date.  For each trade:

    - ``max_positions=1``: keep trade only if its entry_date >= the previous
      kept trade's exit_date (greedy non-overlap).
    - ``max_positions > 1``: track open intervals.  Keep trade if fewer than
      *max_positions* are open **and** no open position for the same symbol
      shares the same expiration.

    Returns the filtered DataFrame of trades that will execute.
    """
    if trades.empty:
        return trades

    n = len(trades)
    entry_dates = trades["entry_date"].values
    exit_dates = trades["exit_date"].values
    expirations = trades["expiration"].values
    symbols = (
        trades["underlying_symbol"].values
        if "underlying_symbol" in trades.columns
        else None
    )

    keep = np.zeros(n, dtype=bool)

    if max_positions == 1:
        # Fast path: greedy non-overlap
        prev_exit = None
        for i in range(n):
            if prev_exit is None or entry_dates[i] >= prev_exit:
                keep[i] = True
                prev_exit = exit_dates[i]
    else:
        # Track open positions; in-place compaction keeps only positions whose
        # exit_date is still in the future. Key duplicate-expiration checks by
        # (symbol, expiration) so distinct symbols can share an expiration date.
        open_positions: list[tuple[Any, Any]] = []
        for i in range(n):
            entry_dt = entry_dates[i]
            open_positions = [p for p in open_positions if p[0] > entry_dt]

            if len(open_positions) < max_positions:
                exp_key = (
                    (symbols[i], expirations[i])
                    if symbols is not None
                    else expirations[i]
                )
                if all(p[1] != exp_key for p in open_positions):
                    keep[i] = True
                    open_positions.append((exit_dates[i], exp_key))

    return trades.loc[keep].reset_index(drop=True)


def _build_trade_log(
    trades: pd.DataFrame,
    capital: float,
    quantity: int,
    multiplier: int,
) -> pd.DataFrame:
    """Compute P&L columns vectorially on filtered trades.

    Trades are ordered chronologically by ``exit_date`` (when P&L is realized),
    with ``entry_date`` as a deterministic tie-breaker, so ``cumulative_pnl``,
    ``equity``, and ruin truncation reflect true realization order even when
    concurrent positions close out of entry order.

    Returns a DataFrame with :data:`_TRADE_LOG_COLUMNS`.  If equity drops to
    zero or below (ruin), the log is truncated at that trade.
    """
    if trades.empty:
        return pd.DataFrame(columns=_TRADE_LOG_COLUMNS)

    # Sort by exit_date (when P&L is realized) before computing cumulative equity
    # and ruin truncation (matches _combine_trade_logs).
    trades = trades.sort_values(
        ["exit_date", "entry_date"], kind="mergesort"
    ).reset_index(drop=True)

    commission = (
        trades["commission"].fillna(0.0).to_numpy(dtype=float)
        if "commission" in trades.columns
        else np.zeros(len(trades), dtype=float)
    )

    trade_log = pd.DataFrame(
        {
            "trade_id": np.arange(1, len(trades) + 1),
            "underlying_symbol": trades["underlying_symbol"].values,
            "entry_date": trades["entry_date"].values,
            "exit_date": trades["exit_date"].values,
            "expiration": trades["expiration"].values,
            "entry_cost": trades["entry_cost"].values,
            "exit_proceeds": trades["exit_proceeds"].values,
            "commission": commission,
            "quantity": quantity,
            "multiplier": multiplier,
            "pct_change": trades["pct_change"].values,
            "description": trades["description"].values,
            "exit_type": trades["exit_type"].values
            if "exit_type" in trades.columns
            else "expiration",
        }
    )

    lot_size = quantity * multiplier
    trade_log["dollar_cost"] = trade_log["entry_cost"].abs() * lot_size
    trade_log["dollar_proceeds"] = trade_log["exit_proceeds"] * lot_size
    trade_log["realized_pnl"] = (
        trade_log["exit_proceeds"] - trade_log["entry_cost"]
    ) * lot_size - trade_log["commission"]
    trade_log["pct_change"] = np.where(
        # Keep strategy-specific return guards (e.g. near-zero calendar entries).
        trade_log["pct_change"].notna() & (trade_log["dollar_cost"] > 0),
        trade_log["realized_pnl"] / trade_log["dollar_cost"],
        np.nan,
    )
    trade_log["days_held"] = (
        pd.to_datetime(trade_log["exit_date"]) - pd.to_datetime(trade_log["entry_date"])
    ).dt.days
    trade_log["cumulative_pnl"] = trade_log["realized_pnl"].cumsum()
    trade_log["equity"] = capital + trade_log["cumulative_pnl"]

    # Ruin check: truncate at first trade where equity <= 0
    ruin_mask = trade_log["equity"] <= 0
    if ruin_mask.any():
        ruin_idx = ruin_mask.idxmax()  # first True
        trade_log = trade_log.loc[:ruin_idx]

    return trade_log


def _unwrap_strategy_name(strategy: Callable[..., Any]) -> str:
    """Resolve the underlying strategy function name, unwrapping functools.partial."""
    fn: Any = strategy
    while isinstance(fn, partial):
        fn = fn.func
    return getattr(fn, "__name__", "")


def _extract_partial_kwarg(strategy: Callable[..., Any], key: str) -> tuple[bool, Any]:
    """Return ``(True, value)`` if *key* is bound in a ``functools.partial`` chain."""
    fn: Any = strategy
    while isinstance(fn, partial):
        if fn.keywords and key in fn.keywords:
            return True, fn.keywords[key]
        fn = fn.func
    return False, None


def simulate(
    data: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    capital: float = 100_000.0,
    quantity: int = 1,
    max_positions: int = 1,
    multiplier: int = 100,
    selector: Union[
        Literal["nearest", "highest_premium", "lowest_premium", "first"],
        Callable[[pd.DataFrame], pd.Series],
    ] = "nearest",
    **strategy_kwargs: Any,
) -> SimulationResult:
    """Run a chronological simulation of an options strategy.

    Args:
        data: Option chain DataFrame (same format as strategy functions expect).
        strategy: Any optopsy strategy function (e.g. ``op.long_calls``).
        capital: Starting capital in dollars.
        quantity: Number of contracts per trade.
        max_positions: Maximum concurrent open positions.
        multiplier: Contract multiplier (100 for standard equity options).
        selector: How to pick one trade when multiple candidates exist for a
            date.  One of ``"nearest"``, ``"highest_premium"``,
            ``"lowest_premium"``, ``"first"``, or a custom callable.
        **strategy_kwargs: Passed through to the strategy function.

    Returns:
        A :class:`SimulationResult` with trade log, equity curve, and summary.
    """
    # Validate arguments via Pydantic model and use validated values
    from pydantic import ValidationError

    from .checks import _format_validation_error
    from .types import SimulatorParams

    try:
        validated = SimulatorParams(
            capital=capital,
            quantity=quantity,
            max_positions=max_positions,
            multiplier=multiplier,
        )
    except ValidationError as e:
        raise ValueError(_format_validation_error(e)) from e

    capital = validated.capital
    quantity = validated.quantity
    max_positions = validated.max_positions
    multiplier = validated.multiplier

    # Resolve selector
    if isinstance(selector, str):
        if selector not in _BUILTIN_SELECTORS:
            raise ValueError(
                f"Unknown selector '{selector}'. "
                f"Choose from: {list(_BUILTIN_SELECTORS.keys())}"
            )
        select_fn = _BUILTIN_SELECTORS[selector]
    else:
        select_fn = selector

    # Scale only per-contract and per-share fees for the simulated lot size.
    # Base/minimum fees remain one charge per side of the simulated trade.
    # Also inspect commissions bound via functools.partial so lot scaling is
    # not skipped when callers pass partial(op.short_puts, commission=...).
    strategy_kwargs = strategy_kwargs.copy()
    if "commission" in strategy_kwargs:
        strategy_kwargs["commission"] = _scale_commission_for_quantity(
            strategy_kwargs["commission"], quantity, multiplier
        )
    else:
        has_bound_comm, bound_comm = _extract_partial_kwarg(strategy, "commission")
        if has_bound_comm and bound_comm is not None:
            strategy_kwargs["commission"] = _scale_commission_for_quantity(
                bound_comm, quantity, multiplier
            )

    # Generate raw trades — return empty result for empty input
    if data.empty:
        raw = pd.DataFrame()
    else:
        raw = strategy(data, raw=True, **strategy_kwargs)

    if raw.empty:
        empty_log = pd.DataFrame(columns=_TRADE_LOG_COLUMNS)
        empty_curve = pd.Series(dtype=float, name="equity")
        return SimulationResult(
            trade_log=empty_log,
            equity_curve=empty_curve,
            summary=_compute_summary(empty_log, capital),
        )

    raw = _adjust_candidate_returns(raw.reset_index(drop=True), quantity * multiplier)

    # Select one trade per entry date (on raw data so OTM% columns are
    # available to selectors)
    raw["_entry_date"] = _resolve_entry_date(raw)
    group_col = "_entry_date"

    # Group by symbol + entry date so multi-symbol data picks one trade per
    # symbol per date, not one trade across all symbols.
    if "underlying_symbol" in raw.columns:
        group_cols = ["underlying_symbol", group_col]
    else:
        group_cols = [group_col]

    selected_raw = pd.DataFrame(
        [select_fn(group) for _, group in raw.groupby(group_cols)]
    )

    # Detect short single-leg strategies so normalisation can negate prices
    strategy_name = _unwrap_strategy_name(strategy)
    is_short_single = strategy_name in _SHORT_SINGLE_LEG

    # Normalise to uniform schema; calendar/diagonal strategies default to
    # exit_dte=7 in CalendarStrategyParams (vs 0 for standard strategies).
    default_exit_dte = 7 if _is_calendar(selected_raw.columns) else 0
    exit_dte = int(strategy_kwargs.get("exit_dte", default_exit_dte))
    trades = _normalise_trades(
        selected_raw, is_short_single=is_short_single, exit_dte=exit_dte
    )
    sort_cols = ["entry_date"] + (
        ["underlying_symbol"] if "underlying_symbol" in trades.columns else []
    )
    trades = trades.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Filter trades by position limits and overlap rules
    filtered = _filter_trades(trades, max_positions)

    # Build trade log with vectorized P&L computation
    trade_log = _build_trade_log(filtered, capital, quantity, multiplier)

    # Build equity curve
    if not trade_log.empty:
        equity_curve = trade_log.set_index("exit_date")["equity"]
        equity_curve.name = "equity"
    else:
        equity_curve = pd.Series(dtype=float, name="equity")

    summary = _compute_summary(trade_log, capital)

    return SimulationResult(
        trade_log=trade_log,
        equity_curve=equity_curve,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

_PORTFOLIO_RESERVED_KEYS = frozenset({"data", "strategy", "weight", "name"})


def simulate_portfolio(
    legs: list[dict],
    capital: float = 100_000.0,
) -> PortfolioResult:
    """Run a weighted portfolio simulation across multiple strategy legs.

    Each leg runs independently via :func:`simulate` with its share of capital.
    Results are combined into a single portfolio equity curve and trade log.

    Args:
        legs: List of leg dicts.  Each must contain ``data`` (DataFrame),
            ``strategy`` (callable), and ``weight`` (float, 0-1).  Optional
            ``name`` (str) labels the leg (defaults to strategy function name).
            All other keys are forwarded to :func:`simulate` and the strategy.
        capital: Total starting capital in dollars.

    Returns:
        A :class:`PortfolioResult` with combined and per-leg results.

    Example::

        result = op.simulate_portfolio(
            legs=[
                {"data": spy, "strategy": op.short_puts, "weight": 0.6,
                 "max_entry_dte": 45, "exit_dte": 14},
                {"data": qqq, "strategy": op.iron_condor, "weight": 0.4,
                 "max_entry_dte": 30, "exit_dte": 7},
            ],
            capital=100_000,
        )
    """
    if not legs:
        raise ValueError("legs must be a non-empty list")

    if (
        isinstance(capital, bool)
        or not isinstance(capital, (int, float))
        or not np.isfinite(capital)
        or capital <= 0
    ):
        raise ValueError(
            "Invalid setting for capital, must be a positive finite int or float"
        )

    # Validate each leg has required keys
    for i, leg in enumerate(legs):
        for key in ("data", "strategy", "weight"):
            if key not in leg:
                raise ValueError(f"leg[{i}] missing required key '{key}'")
        w = leg["weight"]
        if not isinstance(w, (int, float)) or isinstance(w, bool) or w <= 0 or w > 1:
            raise ValueError(f"leg[{i}] weight must be a number in (0, 1], got {w!r}")
        if not callable(leg["strategy"]):
            raise ValueError(f"leg[{i}] strategy must be callable")

    # Validate weights sum to ~1.0
    total_weight = sum(leg["weight"] for leg in legs)
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"weights must sum to 1.0 (got {total_weight:.4f})")

    # Normalize weights so allocated capital sums exactly to `capital`
    norm_factor = 1.0 / total_weight if total_weight != 1.0 else 1.0

    # Run each leg
    leg_results: dict[str, SimulationResult] = {}
    seen_names: set[str] = set()

    for i, leg in enumerate(legs):
        data = leg["data"]
        strategy = leg["strategy"]
        weight = leg["weight"] * norm_factor
        base_name = leg.get("name") or getattr(strategy, "__name__", f"leg{i}")
        name = base_name

        # Deduplicate names with incrementing suffix to avoid collisions
        if name in seen_names:
            suffix = 1
            while f"{base_name}_{suffix}" in seen_names:
                suffix += 1
            name = f"{base_name}_{suffix}"
        seen_names.add(name)

        leg_capital = capital * weight

        # Build kwargs for simulate(): everything except reserved keys
        sim_kwargs: dict[str, Any] = {
            k: v for k, v in leg.items() if k not in _PORTFOLIO_RESERVED_KEYS
        }
        sim_kwargs["capital"] = leg_capital

        leg_results[name] = simulate(data, strategy, **sim_kwargs)

    # Combine trade logs (already applies portfolio-level ruin truncation)
    combined_log = _combine_trade_logs(leg_results, capital)

    # Build daily portfolio equity curve from the retained combined trade log so
    # it always agrees with trade_log and summary (including after ruin).
    combined_curve = _build_portfolio_equity_curve(combined_log)

    # Compute portfolio summary from combined trade log
    summary = _compute_summary(combined_log, capital)

    return PortfolioResult(
        trade_log=combined_log,
        equity_curve=combined_curve,
        summary=summary,
        leg_results=leg_results,
    )


def _combine_trade_logs(
    leg_results: dict[str, SimulationResult], capital: float
) -> pd.DataFrame:
    """Merge per-leg trade logs into a single portfolio trade log."""
    all_logs = []
    for name, result in leg_results.items():
        if result.trade_log.empty:
            continue
        log = result.trade_log.copy()
        log["leg"] = name
        all_logs.append(log)

    if not all_logs:
        cols = _TRADE_LOG_COLUMNS + ["leg"]
        return pd.DataFrame(columns=cols)

    combined = pd.concat(all_logs, ignore_index=True)
    # Sort by exit_date (when P&L is realized) for correct cumulative equity,
    # with entry_date and leg as deterministic tie-breakers.
    combined = combined.sort_values(
        ["exit_date", "entry_date", "leg"], kind="mergesort"
    ).reset_index(drop=True)
    combined["trade_id"] = np.arange(1, len(combined) + 1)
    combined["cumulative_pnl"] = combined["realized_pnl"].cumsum()
    combined["equity"] = capital + combined["cumulative_pnl"]

    # Ruin check: truncate at first trade where portfolio equity <= 0
    # (prevents sign-inverted pct_change() returns on negative equity).
    ruin_mask = combined["equity"] <= 0
    if ruin_mask.any():
        ruin_idx = ruin_mask.idxmax()
        combined = combined.loc[:ruin_idx]

    return combined


def _build_portfolio_equity_curve(combined_log: pd.DataFrame) -> pd.Series:
    """Build a daily portfolio equity curve from the combined trade log.

    ``combined_log["equity"]`` already incorporates total starting capital
    across all legs (including idle cash) and reflects portfolio ruin
    truncation. When multiple trades close on the same exit date, the final
    retained equity for that date is used, then resampled to daily frequency
    with forward-fill.
    """
    if combined_log.empty:
        return pd.Series(dtype=float, name="equity")

    daily = combined_log.groupby("exit_date")["equity"].last()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index().resample("D").last().ffill()
    daily.name = "equity"
    return daily
