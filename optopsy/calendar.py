"""Calendar and diagonal spread evaluation, leg preparation, and exit matching.

Calendar spreads involve options with the same strike but different
expirations; diagonal spreads allow different strikes as well.  This
module handles the specialised pipeline that differs from the regular
strategy path: per-leg DTE filtering, two-leg merging with independent
expirations, exit-price lookup keyed on the front-leg expiration, and
calendar-specific P&L calculation.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .filters import (
    _apply_signal_filter,
    _remove_min_bid_ask,
    _select_closest_delta,
    _trim,
)
from .pricing import (
    _calculate_commission,
    _calculate_fill_price,
    _compute_net_pct_change,
    _get_leg_quantity,
)


def _evaluate_calendar_options(
    data: pd.DataFrame, dte_min: int, dte_max: int, **kwargs: Any
) -> pd.DataFrame:
    """
    Evaluate options for a single leg of a calendar/diagonal spread.

    Args:
        data: DataFrame containing option chain data with DTE assigned
        dte_min: Minimum DTE for this leg
        dte_max: Maximum DTE for this leg
        **kwargs: Additional parameters including min_bid_ask, delta_target

    Returns:
        DataFrame with evaluated options for this leg
    """
    # Filter by DTE range for this leg
    leg_data = _trim(data, "dte", dte_min, dte_max)

    # Remove options with bid/ask below minimum
    leg_data = _remove_min_bid_ask(leg_data, kwargs["min_bid_ask"])

    # Apply delta targeting
    dt = kwargs["delta_target"]
    leg_data = _select_closest_delta(leg_data, dt["target"], dt["min"], dt["max"])

    return leg_data


def _get_strike_column(same_strike: bool, leg_num: int) -> str:
    """Return the appropriate strike column name based on spread type."""
    return "strike" if same_strike else f"strike_leg{leg_num}"


def _prepare_calendar_leg(
    options: pd.DataFrame, leg_num: int, same_strike: bool
) -> pd.DataFrame:
    """
    Rename columns for a calendar/diagonal spread leg.

    Args:
        options: DataFrame with option data for this leg
        leg_num: Leg number (1 for front, 2 for back)
        same_strike: True for calendar spreads, False for diagonal

    Returns:
        DataFrame with renamed columns
    """
    strike_col = _get_strike_column(same_strike, leg_num)

    rename_map = {
        "expiration": f"expiration_leg{leg_num}",
        "dte": f"dte_entry_leg{leg_num}",
        "strike": strike_col,
        "bid": f"bid_leg{leg_num}",
        "ask": f"ask_leg{leg_num}",
        "delta": f"delta_leg{leg_num}",
        "volume": f"volume_leg{leg_num}",
    }
    return options.rename(columns=rename_map)


def _get_calendar_leg_columns(leg_num: int, same_strike: bool) -> List[str]:
    """Return the columns needed for a calendar spread leg."""
    strike_col = _get_strike_column(same_strike, leg_num)
    cols = [
        "underlying_symbol",
        "quote_date",
        "option_type",
        f"expiration_leg{leg_num}",
        f"dte_entry_leg{leg_num}",
        f"bid_leg{leg_num}",
        f"ask_leg{leg_num}",
        f"delta_leg{leg_num}",
    ]
    cols.append(strike_col)
    return cols


def _merge_calendar_legs(
    front: pd.DataFrame, back: pd.DataFrame, same_strike: bool
) -> pd.DataFrame:
    """
    Merge front and back legs of a calendar/diagonal spread.

    Args:
        front: DataFrame with front leg data
        back: DataFrame with back leg data
        same_strike: True for calendar spreads, False for diagonal

    Returns:
        Merged DataFrame
    """
    join_cols = ["underlying_symbol", "quote_date", "option_type"]
    if same_strike:
        join_cols.append("strike")

    front_cols = _get_calendar_leg_columns(1, same_strike)
    back_cols = _get_calendar_leg_columns(2, same_strike)

    if "volume_leg1" in front.columns:
        front_cols = front_cols + ["volume_leg1"]
    if "volume_leg2" in back.columns:
        back_cols = back_cols + ["volume_leg2"]

    return pd.merge(front[front_cols], back[back_cols], on=join_cols, how="inner")


def _get_exit_leg_subset(
    exit_data: pd.DataFrame, leg_num: int, same_strike: bool
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare exit data for joining with a specific leg.

    Args:
        exit_data: DataFrame with exit date prices
        leg_num: Leg number (1 or 2)
        same_strike: True for calendar spreads, False for diagonal

    Returns:
        Tuple of (subset DataFrame, join columns)
    """
    strike_col = _get_strike_column(same_strike, leg_num)

    renamed = exit_data.rename(
        columns={
            "quote_date": "exit_date",
            "expiration": f"expiration_leg{leg_num}",
            "bid": f"exit_bid_leg{leg_num}",
            "ask": f"exit_ask_leg{leg_num}",
            "volume": f"exit_volume_leg{leg_num}",
        }
    )

    if not same_strike:
        renamed = renamed.rename(columns={"strike": strike_col})

    join_cols = [
        "underlying_symbol",
        "exit_date",
        "option_type",
        f"expiration_leg{leg_num}",
        strike_col,
    ]

    subset_cols = join_cols + [f"exit_bid_leg{leg_num}", f"exit_ask_leg{leg_num}"]
    if f"exit_volume_leg{leg_num}" in renamed.columns:
        subset_cols.append(f"exit_volume_leg{leg_num}")

    subset = renamed[subset_cols].drop_duplicates(subset=join_cols, keep="first")
    return subset, join_cols


def _find_calendar_exit_prices(
    merged: pd.DataFrame,
    data: pd.DataFrame,
    exit_dte: int,
    same_strike: bool,
    exit_dte_tolerance: int = 0,
    exit_dates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Find exit prices for calendar/diagonal spread positions.

    Args:
        merged: DataFrame with merged entry positions
        data: Original DataFrame with all option data (with DTE assigned)
        exit_dte: Days before front expiration to exit
        same_strike: True for calendar spreads, False for diagonal
        exit_dte_tolerance: Maximum days of deviation from target exit date (default 0)
        exit_dates: Optional signal DataFrame restricting eligible exit dates

    Returns:
        DataFrame with exit prices merged in, or empty DataFrame if no exit data
    """
    # Filter candidate exit data by exit_dates BEFORE tolerance snapping so a
    # valid signal date within tolerance is not shadowed by a closer non-signal date.
    exit_pool = (
        _apply_signal_filter(data, exit_dates) if exit_dates is not None else data
    )
    if exit_pool.empty:
        return merged.iloc[:0]

    if exit_dte_tolerance == 0:
        # Exact date matching: exit timed relative to front leg (leg1) expiration.
        merged = merged.copy()
        merged["exit_date"] = merged["expiration_leg1"] - pd.Timedelta(days=exit_dte)
        # Never allow exit on or before entry date (prevents look-backward trades)
        merged = merged[merged["exit_date"] > merged["quote_date"]]
        if merged.empty:
            return merged.iloc[:0]

        all_exit_dates = merged["exit_date"].unique()
        exit_data = exit_pool[exit_pool["quote_date"].isin(all_exit_dates)]
        if exit_data.empty:
            return merged.iloc[:0]

        for leg_num in [1, 2]:
            exit_subset, join_cols = _get_exit_leg_subset(
                exit_data, leg_num, same_strike
            )
            merged = pd.merge(merged, exit_subset, on=join_cols, how="inner")
            if merged.empty:
                return merged

        return merged

    # Tolerance-based matching: consider only exit dates within
    # [max(0, exit_dte - exit_dte_tolerance), exit_dte + exit_dte_tolerance]
    # relative to front expiration (which also guarantees exit_date <= expiration_leg1),
    # strictly after entry quote_date, and where BOTH legs have quotes, before
    # selecting the single closest exit per position.
    if "dte" not in exit_pool.columns:
        exit_pool = exit_pool.assign(
            dte=(exit_pool["expiration"] - exit_pool["quote_date"]).dt.days
        )

    lower_dte = max(0, exit_dte - exit_dte_tolerance)
    upper_dte = exit_dte + exit_dte_tolerance
    front_exit_candidates = _trim(exit_pool, "dte", lower_dte, upper_dte)
    if front_exit_candidates.empty:
        return merged.iloc[:0]

    leg1_subset, leg1_join_cols = _get_exit_leg_subset(
        front_exit_candidates, 1, same_strike
    )
    leg1_merge_cols = [c for c in leg1_join_cols if c != "exit_date"]
    merged = pd.merge(merged, leg1_subset, on=leg1_merge_cols, how="inner")

    # Never allow exit on or before entry date (prevents look-backward trades)
    merged = merged[merged["exit_date"] > merged["quote_date"]]
    if merged.empty:
        return merged.iloc[:0]

    leg2_pool = exit_pool[exit_pool["quote_date"].isin(merged["exit_date"].unique())]
    leg2_subset, leg2_join_cols = _get_exit_leg_subset(leg2_pool, 2, same_strike)
    merged = pd.merge(merged, leg2_subset, on=leg2_join_cols, how="inner")
    if merged.empty:
        return merged

    pos_cols = [
        "underlying_symbol",
        "quote_date",
        "option_type",
        "expiration_leg1",
        "expiration_leg2",
    ] + (["strike"] if same_strike else ["strike_leg1", "strike_leg2"])
    actual_exit_dte = (merged["expiration_leg1"] - merged["exit_date"]).dt.days
    merged = (
        merged.assign(
            _exit_dte_diff=(actual_exit_dte - exit_dte).abs(),
            _exit_dte_actual=actual_exit_dte,
        )
        .sort_values(
            ["_exit_dte_diff", "_exit_dte_actual", "exit_date"],
            kind="mergesort",
        )
        .drop_duplicates(subset=pos_cols, keep="first")
        .drop(columns=["_exit_dte_diff", "_exit_dte_actual"])
        .reset_index(drop=True)
    )
    return merged


def _calculate_calendar_pnl(
    merged: pd.DataFrame,
    leg_def: List[Tuple],
    slippage: str = "spread",
    fill_ratio: float = 0.5,
    reference_volume: int = 1000,
    commission: Optional[Dict[str, Any]] = None,
    per_leg_slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate P&L for calendar/diagonal spread positions.

    Args:
        merged: DataFrame with entry and exit prices
        leg_def: List of tuples defining strategy legs
        slippage: Slippage mode - "mid", "spread", "liquidity", or "per_leg"
        fill_ratio: Base fill ratio (0.0-1.0)
        reference_volume: Volume threshold for liquid options
        per_leg_slippage: Additive penalty per additional leg (per_leg mode)

    Returns:
        DataFrame with P&L columns added
    """
    front_side = leg_def[0][0].value
    back_side = leg_def[1][0].value
    front_qty = _get_leg_quantity(leg_def[0])
    back_qty = _get_leg_quantity(leg_def[1])

    # Calculate entry prices based on slippage model
    volume_leg1 = merged["volume_leg1"] if "volume_leg1" in merged.columns else None
    volume_leg2 = merged["volume_leg2"] if "volume_leg2" in merged.columns else None

    num_legs = len(leg_def)
    merged["entry_leg1"] = _calculate_fill_price(
        merged["bid_leg1"],
        merged["ask_leg1"],
        front_side,
        slippage,
        fill_ratio,
        volume_leg1,
        reference_volume,
        per_leg_slippage,
        num_legs,
    )
    merged["entry_leg2"] = _calculate_fill_price(
        merged["bid_leg2"],
        merged["ask_leg2"],
        back_side,
        slippage,
        fill_ratio,
        volume_leg2,
        reference_volume,
        per_leg_slippage,
        num_legs,
    )

    # Calculate exit prices (reverse sides for closing positions)
    exit_vol_leg1 = (
        merged["exit_volume_leg1"] if "exit_volume_leg1" in merged.columns else None
    )
    exit_vol_leg2 = (
        merged["exit_volume_leg2"] if "exit_volume_leg2" in merged.columns else None
    )
    merged["exit_leg1"] = _calculate_fill_price(
        merged["exit_bid_leg1"],
        merged["exit_ask_leg1"],
        -front_side,
        slippage,
        fill_ratio,
        exit_vol_leg1,
        reference_volume,
        per_leg_slippage,
        num_legs,
    )
    merged["exit_leg2"] = _calculate_fill_price(
        merged["exit_bid_leg2"],
        merged["exit_ask_leg2"],
        -back_side,
        slippage,
        fill_ratio,
        exit_vol_leg2,
        reference_volume,
        per_leg_slippage,
        num_legs,
    )

    # Apply position multipliers (direction x quantity) based on leg definition
    front_multiplier = front_side * front_qty
    back_multiplier = back_side * back_qty

    merged["entry_leg1"] = merged["entry_leg1"] * front_multiplier
    merged["exit_leg1"] = merged["exit_leg1"] * front_multiplier
    merged["entry_leg2"] = merged["entry_leg2"] * back_multiplier
    merged["exit_leg2"] = merged["exit_leg2"] * back_multiplier

    # Calculate totals
    merged["total_entry_cost"] = merged["entry_leg1"] + merged["entry_leg2"]
    merged["total_exit_proceeds"] = merged["exit_leg1"] + merged["exit_leg2"]

    net_pnl = merged["total_exit_proceeds"] - merged["total_entry_cost"]

    comm_dollars: pd.Series | float = 0.0
    if commission is not None:
        comm_per_side = _calculate_commission(leg_def, commission)
        merged["total_commission"] = comm_per_side * 2
        comm_dollars = merged["total_commission"]

    # Calculate percentage change.
    # Use a minimum threshold to avoid misleading percentages from near-zero entries.
    merged["pct_change"] = _compute_net_pct_change(
        net_pnl,
        merged["total_entry_cost"],
        comm_dollars,
        min_entry_threshold=0.01,
    )

    return merged
