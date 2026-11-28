"""Internal helpers for options strategy processing.

Contains the ``Side`` enum and private helper functions that assemble leg
definitions and call the core strategy engine.  These are not part of the
public API.

Parameter defaults and validation are handled by the Pydantic models in
``types.py`` (``StrategyParams``, ``CalendarStrategyParams``).  Helpers
pass raw user ``kwargs`` through to ``_process_strategy()`` /
``_process_calendar_strategy()``, which validate and apply defaults.
"""

import inspect
from enum import Enum
from typing import List, Optional, Tuple, Unpack

import numpy as np
import pandas as pd

from ..checks import _run_checks
from ..core import (
    _evaluate_leg,
    _has_early_exit,
    _keep_closest_exit_per_position,
    _prepare_data,
    _process_calendar_strategy,
    _process_strategy,
)
from ..definitions import (
    calendar_spread_external_cols,
    calendar_spread_internal_cols,
    describe_cols,
    diagonal_spread_external_cols,
    diagonal_spread_internal_cols,
    double_strike_internal_cols,
    quadruple_strike_internal_cols,
    single_strike_internal_cols,
    straddle_internal_cols,
    triple_strike_internal_cols,
)
from ..evaluation import _puts
from ..output import _format_output
from ..pricing import (
    _calculate_commission,
    _calculate_fill_price,
    _compute_net_pct_change,
)
from ..rules import (
    _rule_butterfly_strikes,
    _rule_collar_strikes,
    _rule_diagonal_spread,
    _rule_expiration_ordering,
    _rule_iron_butterfly_strikes,
    _rule_iron_condor_strikes,
    _rule_non_overlapping_strike,
)
from ..timestamps import normalize_dates
from ..types import CalendarStrategyParamsDict, StrategyParamsDict


class Side(Enum):
    """Enum representing long or short position side with multiplier values."""

    long = 1
    short = -1


# ---------------------------------------------------------------------------
# Default delta TargetRange dicts for each role
# ---------------------------------------------------------------------------
_DEFAULT_DELTA = {"target": 0.30, "min": 0.20, "max": 0.40}
_DEFAULT_ATM_DELTA = {"target": 0.50, "min": 0.40, "max": 0.60}
_DEFAULT_OTM_DELTA = {"target": 0.10, "min": 0.05, "max": 0.20}
_DEFAULT_WING_DELTA = {"target": 0.20, "min": 0.10, "max": 0.30}
_DEFAULT_DEEP_ITM_DELTA = {"target": 0.80, "min": 0.60, "max": 0.95}
_DEFAULT_ITM_WING_DELTA = {"target": 0.70, "min": 0.55, "max": 0.85}
_DEFAULT_OTM_WING_DELTA = {"target": 0.30, "min": 0.15, "max": 0.45}
_DEFAULT_CONDOR_INNER_ITM = {"target": 0.65, "min": 0.55, "max": 0.75}
_DEFAULT_CONDOR_INNER_OTM = {"target": 0.35, "min": 0.25, "max": 0.45}

_DEFAULT_JOIN_ON = [
    "underlying_symbol",
    "expiration",
    "dte_entry",
    "dte_range",
    "quote_date_exit",
]


def _singles(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process single-leg option strategies (calls or puts)."""
    kwargs.setdefault("leg1_delta", _DEFAULT_DELTA)
    return _process_strategy(
        data,
        internal_cols=single_strike_internal_cols,
        leg_def=leg_def,
        params=kwargs,
    )


def _straddles(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process straddle strategies (call and put at same strike)."""
    kwargs.setdefault("leg1_delta", _DEFAULT_ATM_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)
    return _process_strategy(
        data,
        internal_cols=straddle_internal_cols,
        leg_def=leg_def,
        join_on=[
            "underlying_symbol",
            "expiration",
            "strike",
            "dte_entry",
            "dte_range",
            "quote_date_exit",
        ],
        params=kwargs,
    )


def _strangles(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process strangle strategies (call and put at different strikes)."""
    kwargs.setdefault("leg1_delta", _DEFAULT_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_DELTA)
    return _process_strategy(
        data,
        internal_cols=double_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_non_overlapping_strike,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _spread(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process vertical spread strategies (call or put spreads at different strikes)."""
    # Strike-ordering rule enforces strike_leg1 < strike_leg2:
    # - Calls: lower strike has higher |delta| (leg1 ATM 0.50, leg2 OTM 0.10)
    # - Puts:  lower strike has lower  |delta| (leg1 OTM 0.10, leg2 ATM 0.50)
    if leg_def[0][1] is _puts:
        kwargs.setdefault("leg1_delta", _DEFAULT_OTM_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)
    else:
        kwargs.setdefault("leg1_delta", _DEFAULT_ATM_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_OTM_DELTA)
    return _process_strategy(
        data,
        internal_cols=double_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_non_overlapping_strike,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _butterfly(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process butterfly strategies (3 legs at different strikes)."""
    # Ascending strikes: strike_leg1 < strike_leg2 < strike_leg3
    if leg_def[0][1] is _puts:
        kwargs.setdefault("leg1_delta", _DEFAULT_OTM_WING_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)
        kwargs.setdefault("leg3_delta", _DEFAULT_ITM_WING_DELTA)
    else:
        kwargs.setdefault("leg1_delta", _DEFAULT_ITM_WING_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)
        kwargs.setdefault("leg3_delta", _DEFAULT_OTM_WING_DELTA)
    return _process_strategy(
        data,
        internal_cols=triple_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_butterfly_strikes,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _iron_condor(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process iron condor strategies (4 legs at different strikes)."""
    kwargs.setdefault("leg1_delta", _DEFAULT_OTM_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_DELTA)
    kwargs.setdefault("leg3_delta", _DEFAULT_DELTA)
    kwargs.setdefault("leg4_delta", _DEFAULT_OTM_DELTA)
    return _process_strategy(
        data,
        internal_cols=quadruple_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_iron_condor_strikes,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _iron_butterfly(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process iron butterfly strategies (4 legs, middle legs share strike)."""
    kwargs.setdefault("leg1_delta", _DEFAULT_OTM_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)
    kwargs.setdefault("leg3_delta", _DEFAULT_ATM_DELTA)
    kwargs.setdefault("leg4_delta", _DEFAULT_OTM_DELTA)
    return _process_strategy(
        data,
        internal_cols=quadruple_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_iron_butterfly_strikes,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _covered_call(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stock_data: Optional[pd.DataFrame] = None,
    apply_strike_rule: bool = True,
    **kwargs: Unpack[StrategyParamsDict],
) -> pd.DataFrame:
    """
    Process covered call or protective put strategy.

    When *stock_data* is ``None`` (default), the underlying position is
    approximated via a deep ITM call.  When a DataFrame of stock prices
    is provided, actual stock close prices are used for the underlying
    leg instead.

    Args:
        data: DataFrame containing option chain data.
        leg_def: Leg definitions – ``[(Side, filter), ...]``.
        stock_data: Optional DataFrame of stock prices.  Accepts
            ``yf.download()`` output directly; normalized internally
            via ``_normalize_stock_data()``.
        apply_strike_rule: Whether to enforce ``strike_leg1 < strike_leg2``
            in synthetic mode (True for covered call, False for protective put).
        **kwargs: Strategy parameters forwarded to the processing pipeline.
    """
    if stock_data is not None:
        return _covered_with_stock(data, leg_def, stock_data, **kwargs)

    kwargs.setdefault("leg1_delta", _DEFAULT_DEEP_ITM_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_DELTA)
    return _process_strategy(
        data,
        internal_cols=double_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_non_overlapping_strike if apply_strike_rule else None,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


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


def _covered_with_stock(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stock_data: pd.DataFrame,
    **kwargs: Unpack[StrategyParamsDict],
) -> pd.DataFrame:
    """Process a covered strategy using actual stock data for the underlying.

    The option leg (``leg_def[1]``) is evaluated through the normal
    single-leg pipeline; see ``_stock_with_options``.
    """
    # Prefer leg2_delta (which controls the option leg in the 2-leg public API)
    # when provided, falling back to leg1_delta or _DEFAULT_DELTA.
    opt_delta = kwargs.pop("leg2_delta", None)
    if opt_delta is not None:
        kwargs["leg1_delta"] = opt_delta
    else:
        kwargs.setdefault("leg1_delta", _DEFAULT_DELTA)
    return _stock_with_options(
        data,
        leg_def,
        stock_data,
        kwargs,
        internal_cols=double_strike_internal_cols,
        external_cols=["dte_range", "delta_range_leg2"],
    )


def _collar(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stock_data: Optional[pd.DataFrame] = None,
    **kwargs: Unpack[StrategyParamsDict],
) -> pd.DataFrame:
    """
    Process collar strategy.

    A collar consists of long underlying + short OTM call + long OTM put.

    When *stock_data* is ``None`` (default), the underlying position is
    approximated via a deep ITM call (3-leg synthetic).  When a DataFrame
    of stock prices is provided, actual stock close prices are used for
    the underlying leg instead (2 option legs merged with stock).

    Args:
        data: DataFrame containing option chain data.
        leg_def: Leg definitions – ``[(Side, filter), ...]``.
        stock_data: Optional DataFrame of stock prices.
        **kwargs: Strategy parameters forwarded to the processing pipeline.
    """
    if stock_data is not None:
        return _collar_with_stock(data, leg_def, stock_data, **kwargs)

    # Synthetic collar: deep ITM call (stock proxy) + short OTM call + long OTM put
    kwargs.setdefault("leg1_delta", _DEFAULT_DEEP_ITM_DELTA)
    kwargs.setdefault("leg2_delta", _DEFAULT_DELTA)
    kwargs.setdefault("leg3_delta", _DEFAULT_DELTA)
    return _process_strategy(
        data,
        internal_cols=triple_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_collar_strikes,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _collar_with_stock(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stock_data: pd.DataFrame,
    **kwargs: Unpack[StrategyParamsDict],
) -> pd.DataFrame:
    """Process a collar strategy using actual stock data for the underlying.

    The two option legs (short call + long put) are evaluated through the
    normal pipeline; see ``_stock_with_options``.
    """
    # Map collar-style 3-leg deltas to the 2-leg option pipeline.
    # Public API: leg2_delta = short call, leg3_delta = long put.
    # Internal pipeline uses leg1_delta (opt leg 1) and leg2_delta (opt leg 2).
    opt1_delta = kwargs.pop("leg2_delta", None)
    opt2_delta = kwargs.pop("leg3_delta", None)
    kwargs.pop("leg1_delta", None)  # stock leg delta — not used here
    kwargs["leg1_delta"] = opt1_delta if opt1_delta is not None else _DEFAULT_DELTA
    kwargs["leg2_delta"] = opt2_delta if opt2_delta is not None else _DEFAULT_DELTA
    return _stock_with_options(
        data,
        leg_def,
        stock_data,
        kwargs,
        internal_cols=triple_strike_internal_cols,
        external_cols=["dte_range", "delta_range_leg2", "delta_range_leg3"],
    )


def _condor(
    data: pd.DataFrame, leg_def: List[Tuple], **kwargs: Unpack[StrategyParamsDict]
) -> pd.DataFrame:
    """Process condor strategies (4 legs at different strikes, same option type)."""
    # Ascending strikes: strike_leg1 < strike_leg2 < strike_leg3 < strike_leg4
    if leg_def[0][1] is _puts:
        kwargs.setdefault("leg1_delta", _DEFAULT_WING_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_CONDOR_INNER_OTM)
        kwargs.setdefault("leg3_delta", _DEFAULT_CONDOR_INNER_ITM)
        kwargs.setdefault("leg4_delta", _DEFAULT_DEEP_ITM_DELTA)
    else:
        kwargs.setdefault("leg1_delta", _DEFAULT_DEEP_ITM_DELTA)
        kwargs.setdefault("leg2_delta", _DEFAULT_CONDOR_INNER_ITM)
        kwargs.setdefault("leg3_delta", _DEFAULT_CONDOR_INNER_OTM)
        kwargs.setdefault("leg4_delta", _DEFAULT_WING_DELTA)
    return _process_strategy(
        data,
        internal_cols=quadruple_strike_internal_cols,
        leg_def=leg_def,
        rules=_rule_iron_condor_strikes,
        join_on=_DEFAULT_JOIN_ON,
        params=kwargs,
    )


def _calendar_spread(
    data: pd.DataFrame,
    leg_def: List[Tuple],
    same_strike: bool = True,
    **kwargs: Unpack[CalendarStrategyParamsDict],
) -> pd.DataFrame:
    """
    Process calendar or diagonal spread strategies.

    Calendar spreads have the same strike but different expirations.
    Diagonal spreads have different strikes and different expirations.

    Args:
        data: DataFrame containing option chain data
        leg_def: List of tuples defining strategy legs [(side, option_filter), ...]
        same_strike: True for calendar spreads, False for diagonal spreads
        **kwargs: Optional strategy parameters

    Returns:
        DataFrame with calendar/diagonal spread strategy results
    """
    kwargs.setdefault("leg1_delta", _DEFAULT_DELTA)
    if not same_strike:
        kwargs.setdefault("leg2_delta", _DEFAULT_ATM_DELTA)

    internal_cols = (
        calendar_spread_internal_cols if same_strike else diagonal_spread_internal_cols
    )
    external_cols = (
        calendar_spread_external_cols if same_strike else diagonal_spread_external_cols
    )

    return _process_calendar_strategy(
        data,
        internal_cols=internal_cols,
        external_cols=external_cols,
        leg_def=leg_def,
        rules=_rule_expiration_ordering if same_strike else _rule_diagonal_spread,
        same_strike=same_strike,
        params=kwargs,
    )
