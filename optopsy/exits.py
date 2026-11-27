"""Early exit logic (stop-loss, take-profit, and max-hold-days).

This module provides vectorized early exit detection for options strategies.
It scans intermediate quote dates between each trade's entry and planned exit
to find the first date where unrealized P&L crosses a stop-loss or take-profit
threshold, or the position has been held for ``max_hold_days`` calendar days.

The main entry point is ``_apply_early_exits()``, called from
``core._process_strategy()`` when ``stop_loss``, ``take_profit``, or
``max_hold_days`` is set.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .pricing import (
    _calculate_fill_price,
    _compute_net_pct_change,
    _get_leg_quantity,
)


def _scalar_float(val: Any) -> float:
    """Extract a scalar float from a DataFrame.at[] value."""
    return float(cast(float, val))


def _resolve_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first column name in *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _apply_early_exits(
    result: pd.DataFrame,
    data: pd.DataFrame,
    leg_def: List[Tuple],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Apply early exit logic to strategy results.

    Dispatcher that handles both single-leg and multi-leg strategies.
    Only modifies trades where an early exit threshold is crossed.

    Args:
        result: Strategy result DataFrame from _strategy_engine / _merge_legs.
        data: Normalized option chain data with all quote dates.
        leg_def: Strategy leg definitions [(Side, filter_fn, qty), ...].
        params: Validated strategy parameters dict.

    Returns:
        Modified result DataFrame with early exits applied where triggered.
    """
    stop_loss = params.get("stop_loss")
    take_profit = params.get("take_profit")
    max_hold_days = params.get("max_hold_days")

    if stop_loss is None and take_profit is None and max_hold_days is None:
        return result

    if result.empty:
        result["exit_type"] = pd.Series(dtype="object")
        return result

    result = result.copy()

    if len(leg_def) == 1:
        return _apply_single_leg_exits(
            result, data, leg_def, stop_loss, take_profit, max_hold_days, params
        )
    else:
        return _apply_multi_leg_exits(
            result, data, leg_def, stop_loss, take_profit, max_hold_days, params
        )


def _apply_single_leg_exits(
    result: pd.DataFrame,
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    max_hold_days: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply early exits for single-leg strategies.

    For each trade, find intermediate quotes between entry and planned exit dates,
    compute unrealized P&L as a percentage, and check against thresholds.
    """
    side_value = leg_def[0][0].value  # +1 (long) or -1 (short)

    # Assign trade IDs for tracking
    result["_trade_id"] = np.arange(len(result))
    result["exit_type"] = "expiration"
    result["_early_exit_date"] = pd.NaT

    # Get intermediate snapshots strictly between entry and planned exit
    contract_cols = ["underlying_symbol", "option_type", "expiration", "strike"]
    exit_date_col = _resolve_col(result, ("quote_date_exit", "exit_date"))
    intermediates = _get_intermediate_snapshots(
        data,
        result,
        contract_cols,
        "quote_date_entry",
        exit_date_col=exit_date_col,
    )

    if intermediates.empty:
        result.drop(columns=["_trade_id"], inplace=True)
        return result

    # Merge entry price and entry date onto intermediates
    trade_entry = result[["_trade_id", "entry", "quote_date_entry"]].rename(
        columns={"entry": "_entry_price", "quote_date_entry": "_entry_date"}
    )
    intermediates = intermediates.merge(trade_entry, on="_trade_id")

    # Compute unrealized P&L percentage (mid_now is the intermediate midpoint price)
    intermediates["_unrealized_pct"] = _compute_net_pct_change(
        side_value * (intermediates["_mid"] - intermediates["_entry_price"]),
        intermediates["_entry_price"],
    )

    # Find first threshold crossing per trade
    triggered = _find_first_threshold_crossing(
        intermediates, stop_loss, take_profit, max_hold_days
    )

    if not triggered.empty:
        result = _replace_exits_single_leg(
            result, triggered, data, side_value, params or {}
        )

    result.drop(columns=["_trade_id"], inplace=True)
    return result


def _apply_multi_leg_exits(
    result: pd.DataFrame,
    data: pd.DataFrame,
    leg_def: List[Tuple],
    stop_loss: Optional[float],
    take_profit: Optional[float],
    max_hold_days: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Apply early exits for multi-leg strategies (including straddles and calendars).

    For multi-leg strategies, all legs must have quotes on the same
    intermediate date. Total unrealized P&L is computed across all legs
    before checking thresholds.
    """
    n_legs = len(leg_def)
    result["_trade_id"] = np.arange(len(result))
    result["exit_type"] = "expiration"
    result["_early_exit_date"] = pd.NaT

    contract_cols = ["underlying_symbol", "option_type", "expiration", "strike"]

    # Determine whether trades already carry a planned exit date column; only
    # precompute fallback max_dates if no planned exit date column exists.
    global_exit_col = _resolve_col(
        result, ("quote_date_exit", "quote_date_exit_leg1", "exit_date")
    )
    precomputed_max_dates = None
    if global_exit_col is None:
        available_cols = [c for c in contract_cols if c in data.columns]
        precomputed_max_dates = (
            data.groupby(available_cols, observed=True)["quote_date"]
            .max()
            .reset_index()
            .rename(columns={"quote_date": "_max_date"})
        )

    # Build per-leg intermediate DataFrames
    leg_intermediates = []
    for idx in range(1, n_legs + 1):
        leg = leg_def[idx - 1]
        side_value = leg[0].value
        quantity = _get_leg_quantity(leg)
        multiplier = side_value * quantity

        exp_col = _resolve_col(result, (f"expiration_leg{idx}", "expiration"))
        entry_col_name = _resolve_col(
            result,
            (
                f"quote_date_entry_leg{idx}",
                "quote_date_entry",
                "quote_date_entry_leg1",
                "quote_date",
            ),
        )
        exit_col_name = _resolve_col(
            result,
            (
                "exit_date",
                f"quote_date_exit_leg{idx}",
                "quote_date_exit",
                "quote_date_exit_leg1",
            ),
        )
        opt_type_col = _resolve_col(result, (f"option_type_leg{idx}", "option_type"))
        strike_col = _resolve_col(result, (f"strike_leg{idx}", "strike"))

        if exp_col is None or entry_col_name is None:
            result.drop(columns=["_trade_id"], inplace=True)
            return result

        # Build a clean lookup DataFrame with generic contract column names
        trades_lookup = pd.DataFrame(
            {
                "_trade_id": result["_trade_id"].values,
                "underlying_symbol": result["underlying_symbol"].values,
                "expiration": result[exp_col].values,
                "quote_date_entry": result[entry_col_name].values,
            }
        )
        if opt_type_col is not None:
            trades_lookup["option_type"] = result[opt_type_col].values
        if strike_col is not None:
            trades_lookup["strike"] = result[strike_col].values
        if exit_col_name is not None:
            trades_lookup["_planned_exit_date"] = result[exit_col_name].values

        intermediates = _get_intermediate_snapshots(
            data,
            trades_lookup,
            contract_cols,
            "quote_date_entry",
            exit_date_col="_planned_exit_date" if exit_col_name is not None else None,
            max_dates=precomputed_max_dates,
        )

        if intermediates.empty:
            result.drop(columns=["_trade_id"], inplace=True)
            return result

        # Get entry price for this leg
        entry_col = f"entry_leg{idx}"
        trade_entry = result[["_trade_id", entry_col]].rename(
            columns={entry_col: "_entry_price"}
        )
        intermediates = intermediates.merge(trade_entry, on="_trade_id")

        # Per-leg unrealized P&L contribution
        # _entry_price already has multiplier baked in (from _apply_ratios)
        intermediates["_leg_entry_cost"] = intermediates["_entry_price"]
        intermediates["_leg_exit_value"] = intermediates["_mid"] * multiplier

        leg_intermediates.append(
            intermediates[
                ["_trade_id", "quote_date", "_leg_entry_cost", "_leg_exit_value"]
            ].rename(
                columns={
                    "_leg_entry_cost": f"_leg_entry_{idx}",
                    "_leg_exit_value": f"_leg_exit_{idx}",
                }
            )
        )

    # Merge all legs on (_trade_id, quote_date) — inner join ensures all legs present
    combined = leg_intermediates[0]
    for li in leg_intermediates[1:]:
        combined = combined.merge(li, on=["_trade_id", "quote_date"], how="inner")

    if combined.empty:
        result.drop(columns=["_trade_id"], inplace=True)
        return result

    # Merge entry date for max_hold_days calculation
    entry_date_col = _resolve_col(
        result, ("quote_date_entry", "quote_date_entry_leg1", "quote_date")
    )
    if entry_date_col is not None:
        trade_dates = result[["_trade_id", entry_date_col]].rename(
            columns={entry_date_col: "_entry_date"}
        )
        combined = combined.merge(trade_dates, on="_trade_id")

    # Compute total unrealized P&L
    entry_cols = [f"_leg_entry_{i}" for i in range(1, n_legs + 1)]
    exit_cols = [f"_leg_exit_{i}" for i in range(1, n_legs + 1)]

    combined["_total_entry"] = combined[entry_cols].sum(axis=1)
    combined["_total_exit"] = combined[exit_cols].sum(axis=1)

    is_calendar = "expiration_leg1" in result.columns
    combined["_unrealized_pct"] = _compute_net_pct_change(
        combined["_total_exit"] - combined["_total_entry"],
        combined["_total_entry"],
        min_entry_threshold=0.01 if is_calendar else 0.0,
    )

    # Find first threshold crossing
    triggered = _find_first_threshold_crossing(
        combined, stop_loss, take_profit, max_hold_days
    )

    if not triggered.empty:
        result = _replace_exits_multi_leg(
            result, triggered, data, leg_def, params or {}
        )

    result.drop(columns=["_trade_id"], inplace=True)
    return result


def _get_intermediate_snapshots(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    contract_cols: List[str],
    entry_date_col: str,
    exit_date_col: Optional[str] = None,
    max_dates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract intermediate quotes strictly between entry and planned exit dates.

    Args:
        data: Full option chain data with all quote dates.
        trades: Trade DataFrame with _trade_id and contract columns.
        contract_cols: Columns identifying a contract (symbol, type, exp, strike).
        entry_date_col: Column name for entry date in trades.
        exit_date_col: Optional column name for each trade's planned exit date
            (e.g. ``quote_date_exit`` or ``exit_date``). When provided, only
            intermediate quotes strictly before this date are considered.
        max_dates: Fallback pre-computed max quote_date per contract group when
            *exit_date_col* is unavailable.

    Returns:
        DataFrame with _trade_id, quote_date, and _mid (midpoint price)
        for each intermediate date.
    """
    if trades.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    available_cols = [c for c in contract_cols if c in trades.columns]
    if not available_cols:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    use_planned_exit = exit_date_col is not None and exit_date_col in trades.columns
    extra_cols = [entry_date_col] + ([exit_date_col] if use_planned_exit else [])
    trade_contracts = trades[["_trade_id"] + extra_cols + available_cols].copy()

    # Pre-filter raw chain to traded expirations/symbols before joining
    data_subset = data
    if "expiration" in available_cols and "expiration" in data_subset.columns:
        data_subset = data_subset[
            data_subset["expiration"].isin(trade_contracts["expiration"].unique())
        ]
    if (
        "underlying_symbol" in available_cols
        and "underlying_symbol" in data_subset.columns
    ):
        data_subset = data_subset[
            data_subset["underlying_symbol"].isin(
                trade_contracts["underlying_symbol"].unique()
            )
        ]

    if data_subset.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    keep_data_cols = list(dict.fromkeys(available_cols + ["quote_date", "bid", "ask"]))
    dedup_keys = list(dict.fromkeys(available_cols + ["quote_date"]))
    deduped_data = data_subset[keep_data_cols].drop_duplicates(
        subset=dedup_keys, keep="first"
    )
    merged = deduped_data.merge(trade_contracts, on=available_cols, how="inner")

    if merged.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    # Filter to dates strictly after entry
    merged = merged[merged["quote_date"] > merged[entry_date_col]]
    if merged.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    # Filter to dates strictly before planned exit
    if use_planned_exit and exit_date_col is not None:
        merged = merged[merged["quote_date"] < merged[exit_date_col]]
    else:
        if max_dates is None:
            max_dates = (
                data.groupby(available_cols, observed=True)["quote_date"]
                .max()
                .reset_index()
                .rename(columns={"quote_date": "_max_date"})
            )
        merged = merged.merge(max_dates, on=available_cols, how="left")
        merged = merged[merged["quote_date"] < merged["_max_date"]]

    if merged.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_mid"])

    # Compute midpoint
    merged["_mid"] = (
        merged["bid"].astype("float64") + merged["ask"].astype("float64")
    ) / 2

    return merged[["_trade_id", "quote_date", "_mid"]].reset_index(drop=True)


def _find_first_threshold_crossing(
    intermediates: pd.DataFrame,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    max_hold_days: Optional[int] = None,
) -> pd.DataFrame:
    """Find the first date per trade where an early-exit condition is met.

    Conditions checked (priority when multiple trigger on same date):
    1. stop_loss — unrealized P&L <= threshold
    2. take_profit — unrealized P&L >= threshold
    3. max_hold — calendar days held >= max_hold_days

    Args:
        intermediates: DataFrame with _trade_id, quote_date, _unrealized_pct,
            and optionally _entry_date (required when max_hold_days is set).
        stop_loss: Negative float threshold (e.g., -0.50).
        take_profit: Positive float threshold (e.g., 0.50).
        max_hold_days: Maximum calendar days to hold a position.

    Returns:
        DataFrame with _trade_id, quote_date, _exit_type for triggered trades.
    """
    # Build crossing mask
    mask = pd.Series(False, index=intermediates.index)

    if stop_loss is not None:
        mask = mask | (intermediates["_unrealized_pct"] <= stop_loss)
    if take_profit is not None:
        mask = mask | (intermediates["_unrealized_pct"] >= take_profit)
    if max_hold_days is not None and "_entry_date" in intermediates.columns:
        hold_duration = (
            intermediates["quote_date"] - intermediates["_entry_date"]
        ).dt.days
        mask = mask | (hold_duration >= max_hold_days)

    crossed = intermediates[mask].copy()

    if crossed.empty:
        return pd.DataFrame(columns=["_trade_id", "quote_date", "_exit_type"])

    # For each trade, take the first crossing date (chronologically)
    crossed = crossed.sort_values(["_trade_id", "quote_date"])
    first_per_trade = crossed.drop_duplicates(subset=["_trade_id"], keep="first")

    # Determine exit type — priority: stop_loss > take_profit > max_hold
    conditions = []
    choices = []
    if stop_loss is not None:
        conditions.append(first_per_trade["_unrealized_pct"] <= stop_loss)
        choices.append("stop_loss")
    if take_profit is not None:
        conditions.append(first_per_trade["_unrealized_pct"] >= take_profit)
        choices.append("take_profit")
    if max_hold_days is not None:
        conditions.append(pd.Series(True, index=first_per_trade.index))
        choices.append("max_hold")

    first_per_trade["_exit_type"] = np.select(conditions, choices, default="")

    return first_per_trade[["_trade_id", "quote_date", "_exit_type"]].reset_index(
        drop=True
    )


def _replace_exits_single_leg(
    result: pd.DataFrame,
    triggered: pd.DataFrame,
    data: pd.DataFrame,
    side_value: int,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Replace exit data for triggered single-leg trades.

    Updates bid_exit, ask_exit, exit, dte_exit (if present), pct_change,
    exit_type, and _early_exit_date for each triggered trade. The planned
    exit date in quote_date_exit (if present) is not modified; the actual
    early-exit quote date is stored in _early_exit_date.
    """
    trig_trades = triggered[["_trade_id", "quote_date", "_exit_type"]].rename(
        columns={"quote_date": "_exit_quote_date"}
    )

    result_cols = ["_trade_id", "underlying_symbol", "expiration", "strike", "entry"]
    if "total_commission" in result.columns:
        result_cols.append("total_commission")
    if "option_type" in result.columns:
        result_cols.append("option_type")

    lookup = trig_trades.merge(result[result_cols], on="_trade_id", how="inner")

    data_merge_cols = ["underlying_symbol", "expiration", "strike"]
    if "option_type" in lookup.columns and "option_type" in data.columns:
        data_merge_cols.append("option_type")

    lookup = lookup.rename(columns={"_exit_quote_date": "quote_date"})
    data_exit_keys = ["quote_date"] + data_merge_cols
    exit_price_cols = ["bid", "ask"] + (["volume"] if "volume" in data.columns else [])

    # Pre-filter raw chain to triggered exit dates before deduplicating
    trig_dates = lookup["quote_date"].unique()
    data_exit = data.loc[
        data["quote_date"].isin(trig_dates), data_exit_keys + exit_price_cols
    ].drop_duplicates(subset=data_exit_keys, keep="first")

    lookup = lookup.merge(data_exit, on=data_exit_keys, how="inner")

    if lookup.empty:
        return result

    volume = lookup["volume"] if "volume" in lookup.columns else None
    lookup["_new_exit"] = _calculate_fill_price(
        lookup["bid"],
        lookup["ask"],
        -side_value,
        params.get("slippage", "mid"),
        params.get("fill_ratio", 0.5),
        volume,
        params.get("reference_volume", 1000),
        params.get("per_leg_slippage", 0.0),
        num_legs=1,
    )
    comm_dollars = (
        lookup["total_commission"] if "total_commission" in lookup.columns else 0.0
    )
    lookup["_new_pct"] = _compute_net_pct_change(
        side_value * (lookup["_new_exit"] - lookup["entry"]),
        lookup["entry"],
        comm_dollars,
    )

    result_indexed = result.set_index("_trade_id")
    updates = lookup.set_index("_trade_id")

    for col_name, lookup_col in (("bid_exit", "bid"), ("ask_exit", "ask")):
        if col_name in result_indexed.columns:
            result_indexed.loc[updates.index, col_name] = updates[lookup_col].values

    result_indexed.loc[updates.index, "exit"] = updates["_new_exit"].values
    result_indexed.loc[updates.index, "pct_change"] = updates["_new_pct"].values
    result_indexed.loc[updates.index, "exit_type"] = updates["_exit_type"].values
    result_indexed.loc[updates.index, "_early_exit_date"] = pd.to_datetime(
        updates["quote_date"]
    ).values

    if "dte_exit" in result_indexed.columns and "expiration" in result_indexed.columns:
        new_dte = (updates["expiration"] - updates["quote_date"]).dt.days
        result_indexed.loc[updates.index, "dte_exit"] = new_dte.values

    return result_indexed.reset_index()


def _replace_exits_multi_leg(
    result: pd.DataFrame,
    triggered: pd.DataFrame,
    data: pd.DataFrame,
    leg_def: List[Tuple],
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Replace exit data for triggered multi-leg trades (including straddles and calendars)."""
    n_legs = len(leg_def)

    trig_trades = triggered[["_trade_id", "quote_date", "_exit_type"]].rename(
        columns={"quote_date": "_exit_quote_date"}
    )

    # Collect result columns needed for per-leg contract lookup
    result_cols = ["_trade_id", "underlying_symbol"]
    if "expiration" in result.columns:
        result_cols.append("expiration")
    for leg_idx in range(1, n_legs + 1):
        for col in (
            f"expiration_leg{leg_idx}",
            f"option_type_leg{leg_idx}",
            f"strike_leg{leg_idx}",
            f"entry_leg{leg_idx}",
        ):
            if col in result.columns:
                result_cols.append(col)
    for shared_col in ("option_type", "strike", "total_entry_cost"):
        if shared_col in result.columns:
            result_cols.append(shared_col)

    lookup = trig_trades.merge(
        result[list(dict.fromkeys(result_cols))], on="_trade_id", how="inner"
    )

    if lookup.empty:
        return result

    # Pre-filter and deduplicate raw chain ONCE across triggered exit dates
    has_opt_type = "option_type" in data.columns
    merge_cols = ["quote_date", "underlying_symbol", "expiration", "strike"]
    if has_opt_type:
        merge_cols.append("option_type")
    data_cols = (
        merge_cols + ["bid", "ask"] + (["volume"] if "volume" in data.columns else [])
    )
    trig_dates = lookup["_exit_quote_date"].unique()
    data_merge = data.loc[
        data["quote_date"].isin(trig_dates), data_cols
    ].drop_duplicates(subset=merge_cols, keep="first")

    if data_merge.empty:
        return result

    # Track which trade_ids have all legs found
    valid_ids = set(lookup["_trade_id"].values)
    leg_updates: Dict[int, pd.DataFrame] = {}

    for leg_idx in range(1, n_legs + 1):
        leg = leg_def[leg_idx - 1]
        side_value = leg[0].value
        quantity = _get_leg_quantity(leg)
        multiplier = side_value * quantity

        exp_col = _resolve_col(lookup, (f"expiration_leg{leg_idx}", "expiration"))
        strike_col = _resolve_col(lookup, (f"strike_leg{leg_idx}", "strike"))
        opt_type_col = _resolve_col(
            lookup, (f"option_type_leg{leg_idx}", "option_type")
        )

        if exp_col is None or strike_col is None:
            return result

        # Build a clean per-leg lookup DataFrame with unique column names
        # (prevents duplicate 'strike'/'option_type' columns on straddles & calendars)
        active = lookup[lookup["_trade_id"].isin(valid_ids)]
        leg_lookup_dict: Dict[str, Any] = {
            "_trade_id": active["_trade_id"].values,
            "quote_date": active["_exit_quote_date"].values,
            "underlying_symbol": active["underlying_symbol"].values,
            "expiration": active[exp_col].values,
            "strike": active[strike_col].values,
        }
        if has_opt_type:
            if opt_type_col is None:
                return result
            leg_lookup_dict["option_type"] = active[opt_type_col].values

        leg_lookup_merge = pd.DataFrame(leg_lookup_dict)
        merged = leg_lookup_merge.merge(data_merge, on=merge_cols, how="inner")

        found_ids = set(merged["_trade_id"].values)
        valid_ids &= found_ids
        if not valid_ids:
            return result

        volume = merged["volume"] if "volume" in merged.columns else None
        merged["_new_exit"] = _calculate_fill_price(
            merged["bid"],
            merged["ask"],
            -side_value,
            params.get("slippage", "mid"),
            params.get("fill_ratio", 0.5),
            volume,
            params.get("reference_volume", 1000),
            params.get("per_leg_slippage", 0.0),
            num_legs=n_legs,
        )

        leg_updates[leg_idx] = (
            merged[["_trade_id", "bid", "ask", "_new_exit"]]
            .rename(columns={"bid": "_bid", "ask": "_ask"})
            .copy()
        )
        leg_updates[leg_idx]["_multiplier"] = multiplier

    if not valid_ids:
        return result

    # Apply updates to result using _trade_id index
    result_indexed = result.set_index("_trade_id")
    valid_trig = trig_trades[trig_trades["_trade_id"].isin(valid_ids)].set_index(
        "_trade_id"
    )
    new_exit_totals = pd.Series(0.0, index=valid_trig.index)

    for leg_idx in range(1, n_legs + 1):
        updates = leg_updates[leg_idx]
        updates = updates[updates["_trade_id"].isin(valid_ids)].set_index("_trade_id")

        for bid_col in (f"bid_exit_leg{leg_idx}", f"exit_bid_leg{leg_idx}"):
            if bid_col in result_indexed.columns:
                result_indexed.loc[updates.index, bid_col] = updates["_bid"].values
        for ask_col in (f"ask_exit_leg{leg_idx}", f"exit_ask_leg{leg_idx}"):
            if ask_col in result_indexed.columns:
                result_indexed.loc[updates.index, ask_col] = updates["_ask"].values

        leg_exit_values = np.asarray(
            (updates["_new_exit"] * updates["_multiplier"]).values
        )
        exit_col = f"exit_leg{leg_idx}"
        if exit_col in result_indexed.columns:
            result_indexed.loc[updates.index, exit_col] = leg_exit_values

        new_exit_totals.loc[updates.index] = (
            new_exit_totals.loc[updates.index].values + leg_exit_values
        )

    # Update totals and pct_change
    if "total_entry_cost" in result_indexed.columns:
        entry_totals = result_indexed.loc[valid_trig.index, "total_entry_cost"]
    else:
        entry_cols = [f"entry_leg{i}" for i in range(1, n_legs + 1)]
        available_entry_cols = [c for c in entry_cols if c in result_indexed.columns]
        entry_totals = result_indexed.loc[valid_trig.index, available_entry_cols].sum(
            axis=1
        )

    if "total_exit_proceeds" in result_indexed.columns:
        result_indexed.loc[valid_trig.index, "total_exit_proceeds"] = (
            new_exit_totals.values
        )

    comm_dollars = (
        result_indexed.loc[valid_trig.index, "total_commission"]
        if "total_commission" in result_indexed.columns
        else 0.0
    )
    is_calendar = "expiration_leg1" in result_indexed.columns
    result_indexed.loc[valid_trig.index, "pct_change"] = _compute_net_pct_change(
        np.asarray(new_exit_totals.values) - np.asarray(entry_totals.values),
        np.asarray(entry_totals.values),
        comm_dollars,
        min_entry_threshold=0.01 if is_calendar else 0.0,
    )

    result_indexed.loc[valid_trig.index, "exit_type"] = valid_trig["_exit_type"].values
    result_indexed.loc[valid_trig.index, "_early_exit_date"] = pd.to_datetime(
        valid_trig["_exit_quote_date"]
    ).values

    return result_indexed.reset_index()
