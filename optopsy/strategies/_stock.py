"""Stock-backed strategy support (covered calls, protective puts, collars).

Normalizes user-supplied stock prices and provides ``_stock_with_options``,
the shared pipeline that pairs real stock closes with one or two option legs.
Private to the ``strategies`` package.
"""

import inspect
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..checks import _run_checks
from ..core import (
    _evaluate_leg,
    _has_early_exit,
    _keep_closest_exit_per_position,
    _prepare_data,
)
from ..definitions import describe_cols
from ..output import _format_output
from ..pricing import (
    _calculate_commission,
    _calculate_fill_price,
    _compute_net_pct_change,
)
from ..timestamps import normalize_dates


def _normalize_stock_data(
    stock_data: pd.DataFrame, options_data: pd.DataFrame
) -> pd.DataFrame:
    """Normalize stock data to the internal format expected by the pipeline.

    Accepts stock data from various sources (yfinance, CSV, user-provided
    DataFrames) and returns a deduplicated DataFrame with columns:
    ``[underlying_symbol, quote_date, close]``.

    Normalization steps:

    1. Flatten or reshape ``MultiIndex`` columns (yfinance single- or
       multi-ticker downloads).
    2. If the index is a ``DatetimeIndex``, reset it to a column (handles
       yfinance output where dates are the index).
    3. Lowercase all column names (``Close`` → ``close``).
    4. Map ``date`` → ``quote_date`` if ``quote_date`` is missing.
    5. If ``underlying_symbol`` is absent, infer it from *options_data*
       (works for single-symbol datasets, raises for multi-symbol).
    """
    df = stock_data.copy()

    # Handle MultiIndex columns (yfinance downloads: level 0 = field, level 1 = ticker)
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.get_level_values(-1).unique()
        non_empty_tickers = [t for t in tickers if t != ""]
        if len(non_empty_tickers) > 1:
            ticker_level_name = df.columns.names[-1]
            if isinstance(df.index, pd.DatetimeIndex) and df.index.name is None:
                df.index = df.index.rename("date")
            # Multi-ticker yfinance DataFrame: stack ticker level into rows.
            # future_stack was added in pandas 2.1; fall back for pandas 2.0.
            if "future_stack" in inspect.signature(df.stack).parameters:
                df = df.stack(level=-1, future_stack=True).reset_index()
            else:
                df = df.stack(level=-1).reset_index()
            # Rename stacked ticker column to underlying_symbol
            ticker_col = (
                ticker_level_name if ticker_level_name is not None else "level_1"
            )
            if ticker_col in df.columns:
                df = df.rename(columns={ticker_col: "underlying_symbol"})
            elif "Ticker" in df.columns:
                df = df.rename(columns={"Ticker": "underlying_symbol"})
            elif "level_1" in df.columns:
                df = df.rename(columns={"level_1": "underlying_symbol"})
        else:
            df.columns = [c[0] for c in df.columns]
            if len(non_empty_tickers) == 1 and "underlying_symbol" not in [
                str(c).lower() for c in df.columns
            ]:
                df["underlying_symbol"] = non_empty_tickers[0]

    # Reset DatetimeIndex to a column (yfinance uses date as index)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Lowercase all column names
    df.columns = [str(c).lower() for c in df.columns]

    # Map common date column names to quote_date
    if "quote_date" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "quote_date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "quote_date"})

    # Infer underlying_symbol from options data if missing
    if "underlying_symbol" not in df.columns:
        symbols = options_data["underlying_symbol"].unique()
        if len(symbols) == 1:
            df["underlying_symbol"] = symbols[0]
        else:
            raise KeyError(
                "stock_data has no 'underlying_symbol' column and the options "
                "data contains multiple symbols. Please add an "
                "'underlying_symbol' column to your stock data."
            )

    # Validate required columns are present after normalization
    required = {"underlying_symbol", "quote_date", "close"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"stock_data is missing required columns after normalization: "
            f"{', '.join(sorted(missing))}. "
            f"Expected columns: underlying_symbol (or inferred), "
            f"quote_date (or date/DatetimeIndex), close."
        )

    df["quote_date"] = normalize_dates(pd.to_datetime(df["quote_date"]))
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    return (
        df[["underlying_symbol", "quote_date", "close"]]
        .dropna(subset=["close"])
        .drop_duplicates(subset=["underlying_symbol", "quote_date"], keep="last")
    )


def _match_stock_prices(
    result: pd.DataFrame,
    stock_data: pd.DataFrame,
    options_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge stock close prices by entry and exit date onto a result DataFrame.

    Normalizes *stock_data* via ``_normalize_stock_data()`` and inner-joins
    on ``(underlying_symbol, quote_date_entry)`` and
    ``(underlying_symbol, quote_date_exit)``, adding ``_stock_entry`` and
    ``_stock_exit`` columns.
    """
    stock_prices = _normalize_stock_data(stock_data, options_data)

    entry_map = stock_prices.rename(
        columns={"quote_date": "quote_date_entry", "close": "_stock_entry"}
    )
    result = result.merge(
        entry_map, on=["underlying_symbol", "quote_date_entry"], how="inner"
    )

    exit_map = stock_prices.rename(
        columns={"quote_date": "quote_date_exit", "close": "_stock_exit"}
    )
    result = result.merge(
        exit_map, on=["underlying_symbol", "quote_date_exit"], how="inner"
    )
    return result


def _apply_option_slippage(
    result: pd.DataFrame,
    side_value: int,
    params: dict,
    suffix: str = "",
    num_legs: int = 1,
) -> pd.DataFrame:
    """Apply slippage to entry/exit prices on a single option leg.

    *suffix* distinguishes legs when multiple option legs are present
    (e.g. ``"_opt1"``, ``"_opt2"``).  When the required bid/ask columns
    are missing or slippage is ``"mid"``, the DataFrame is returned
    unchanged.
    """
    slippage = params["slippage"]
    bid_entry_col = f"bid_entry{suffix}"
    ask_entry_col = f"ask_entry{suffix}"
    bid_exit_col = f"bid_exit{suffix}"
    ask_exit_col = f"ask_exit{suffix}"
    if not all(
        c in result.columns
        for c in (bid_entry_col, ask_entry_col, bid_exit_col, ask_exit_col)
    ):
        return result
    if slippage == "mid":
        return result

    result = result.copy()
    volume_entry = (
        result[f"volume_entry{suffix}"]
        if f"volume_entry{suffix}" in result.columns
        else None
    )
    result[f"entry{suffix}"] = _calculate_fill_price(
        result[bid_entry_col],
        result[ask_entry_col],
        side_value,
        slippage,
        params["fill_ratio"],
        volume_entry,
        params["reference_volume"],
        params["per_leg_slippage"],
        num_legs=num_legs,
    )
    volume_exit = (
        result[f"volume_exit{suffix}"]
        if f"volume_exit{suffix}" in result.columns
        else None
    )
    result[f"exit{suffix}"] = _calculate_fill_price(
        result[bid_exit_col],
        result[ask_exit_col],
        -side_value,
        slippage,
        params["fill_ratio"],
        volume_exit,
        params["reference_volume"],
        params["per_leg_slippage"],
        num_legs=num_legs,
    )
    return result


def _reject_unsupported_stock_early_exits(params: dict) -> None:
    """Raise NotImplementedError if early exits are requested alongside stock_data."""
    if _has_early_exit(params):
        raise NotImplementedError(
            "Early exits (stop_loss, take_profit, max_hold_days) are not yet "
            "supported when using stock_data."
        )


def _stock_with_options(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stock_data: pd.DataFrame,
    kwargs: dict,
    internal_cols: List[str],
    external_cols: List[str],
) -> pd.DataFrame:
    """Shared pipeline for strategies pairing real stock prices with options.

    ``leg_def[0]`` is the stock/synthetic leg and is ignored; ``leg_def[1:]``
    are the option legs (one for covered call/protective put, two for collar).
    *kwargs* must already map the option legs' deltas onto ``leg1_delta`` ..
    ``legN_delta`` (option-leg numbering), since the stock leg has no delta
    targeting.  Each option leg runs through the normal evaluation pipeline;
    stock closes are then matched by entry/exit date to compute the combined
    entry cost, exit proceeds, and percentage change.

    Two option legs are merged on common keys with the collar strike rule
    (long put strike < short call strike) enforced.
    """
    option_legs = leg_def[1:]
    n_opt = len(option_legs)
    # Single option leg keeps unsuffixed columns; two legs are suffixed on merge.
    suffixes = [""] if n_opt == 1 else [f"_opt{i}" for i in range(1, n_opt + 1)]

    params = _run_checks(dict(kwargs), data)
    _reject_unsupported_stock_early_exits(params)

    data = _prepare_data(data)
    select_closest_exit = n_opt == 1 or params["exit_dte_tolerance"] == 0

    def _empty_result() -> pd.DataFrame:
        """Return a correctly shaped empty DataFrame for both raw and aggregated."""
        if params["raw"]:
            return pd.DataFrame(columns=internal_cols)
        return pd.DataFrame(columns=external_cols + describe_cols)

    # --- evaluate each option leg ---
    evaluated = [
        _evaluate_leg(
            leg[1](data), params, params[f"leg{i}_delta"], select_closest_exit
        )
        for i, leg in enumerate(option_legs, start=1)
    ]
    if any(ev.empty for ev in evaluated):
        return _empty_result()

    if n_opt == 1:
        result = evaluated[0]
    else:
        merge_keys = [
            "underlying_symbol",
            "expiration",
            "quote_date_entry",
            "quote_date_exit",
            "dte_entry",
        ]
        if all("dte_range" in ev.columns for ev in evaluated):
            merge_keys.append("dte_range")
        result = evaluated[0].merge(
            evaluated[1], on=merge_keys, suffixes=("_opt1", "_opt2"), how="inner"
        )
        # Collar strike ordering: long put strike < short call strike
        result = result[result["strike_opt2"] < result["strike_opt1"]]
        if result.empty:
            return _empty_result()

    # --- match stock prices ---
    result = _match_stock_prices(result, stock_data, data)
    if result.empty:
        return _empty_result()

    if not select_closest_exit and "quote_date_exit" in result.columns:
        pos_cols = [
            "underlying_symbol",
            "expiration",
            "quote_date_entry",
            "dte_entry",
            *(f"strike{s}" for s in suffixes),
        ]
        result = _keep_closest_exit_per_position(result, pos_cols, params["exit_dte"])

    # --- apply slippage to option legs ---
    for suffix, leg in zip(suffixes, option_legs):
        result = _apply_option_slippage(
            result, leg[0].value, params, suffix, num_legs=n_opt
        )

    # --- compute combined P&L ---
    total_entry = result["_stock_entry"]
    total_exit = result["_stock_exit"]
    for suffix, leg in zip(suffixes, option_legs):
        total_entry = total_entry + result[f"entry{suffix}"] * leg[0].value
        total_exit = total_exit + result[f"exit{suffix}"] * leg[0].value
    net_pnl = total_exit - total_entry

    # commission is already a plain dict after _run_checks() -> model_dump()
    commission_obj = params.get("commission")
    total_commission = 0.0
    if commission_obj is not None:
        # Option legs only; the stock leg is handled via has_stock_leg
        comm_per_side = _calculate_commission(
            option_legs, commission_obj, has_stock_leg=True, num_shares=100
        )
        total_commission = comm_per_side * 2

    columns = {
        "underlying_symbol": result["underlying_symbol"].values,
        "underlying_price_entry_leg1": result["_stock_entry"].values,
        "expiration": result["expiration"].values,
        "quote_date_exit": result["quote_date_exit"].values,
        "dte_entry": result["dte_entry"].values,
        "option_type_leg1": "stock",
        "strike_leg1": result["_stock_entry"].values,
        "total_entry_cost": total_entry.values,
        "total_exit_proceeds": total_exit.values,
        "pct_change": _compute_net_pct_change(net_pnl, total_entry, total_commission),
        "delta_entry_leg1": 1.0,
    }
    # Stock delta is always 1.0, so only option legs get delta/range columns.
    for leg_no, suffix in enumerate(suffixes, start=2):
        columns[f"option_type_leg{leg_no}"] = result[f"option_type{suffix}"].values
        columns[f"strike_leg{leg_no}"] = result[f"strike{suffix}"].values
        columns[f"delta_entry_leg{leg_no}"] = (
            result[f"delta_entry{suffix}"].values
            if f"delta_entry{suffix}" in result.columns
            else np.nan
        )
    if "dte_range" in result.columns:
        columns["dte_range"] = result["dte_range"].values
    elif "dte_range" in internal_cols:
        columns["dte_range"] = np.nan

    output = pd.DataFrame(columns)
    if commission_obj is not None:
        output["total_commission"] = total_commission

    # Carry over grouping columns produced by the evaluation pipeline
    for leg_no, suffix in enumerate(suffixes, start=2):
        if f"delta_range{suffix}" in result.columns:
            output[f"delta_range_leg{leg_no}"] = result[f"delta_range{suffix}"].values

    return _format_output(output, params, internal_cols, external_cols)
