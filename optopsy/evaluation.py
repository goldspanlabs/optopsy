"""Option evaluation pipeline — entry/exit matching and option-type filters.

This module evaluates raw option chain data by filtering candidates,
matching entry rows with corresponding exit rows, and producing a
DataFrame ready for strategy construction.  It also provides the
``_calls`` and ``_puts`` convenience filters used by ``strategies.py``.
"""

from typing import Any

import pandas as pd

from .definitions import evaluated_cols
from .filters import (
    _apply_signal_filter,
    _assign_dte,
    _cut_options_by_delta,
    _cut_options_by_dte,
    _get,
    _ltrim,
    _remove_invalid_evaluated_options,
    _remove_min_bid_ask,
    _select_closest_delta,
    _trim,
)


def _get_exits(
    data: pd.DataFrame,
    exit_dte: int,
    exit_dte_tolerance: int = 0,
    *,
    select_closest: bool = True,
) -> pd.DataFrame:
    """
    Get exit rows from option data, optionally using tolerance-based DTE matching.

    When tolerance is 0 (default), only rows with exactly the target exit_dte are
    returned. When tolerance > 0, rows within [max(0, exit_dte - tolerance),
    exit_dte + tolerance] are considered. If *select_closest* is True (default),
    for each contract the row with DTE closest to exit_dte is selected; when False,
    all eligible exit rows (deduplicated per contract and quote_date) are returned
    so multi-leg strategies can match on a common exit date before selecting the
    closest exit.
    """
    contract_cols = ["underlying_symbol", "option_type", "expiration", "strike"]

    if exit_dte_tolerance == 0:
        exits = _get(data, "dte", exit_dte)
        if exits.empty:
            return exits
        return exits.drop_duplicates(subset=contract_cols, keep="first")

    lower = max(0, exit_dte - exit_dte_tolerance)
    upper = exit_dte + exit_dte_tolerance
    candidates = _trim(data, "dte", lower, upper)

    if candidates.empty:
        return candidates

    if not select_closest:
        dedup_cols = contract_cols + (
            ["quote_date"] if "quote_date" in candidates.columns else []
        )
        return candidates.drop_duplicates(subset=dedup_cols, keep="first")

    # For each contract, pick the row with DTE closest to exit_dte.
    # Reset index so duplicate input index labels cannot cause .loc[idx] to
    # return multiple rows, and sort deterministically by (dte, quote_date) so
    # equidistant ties always break the same way regardless of input row order.
    candidates = candidates.reset_index(drop=True)
    candidates["_dte_diff"] = (candidates["dte"] - exit_dte).abs()
    sort_cols = ["dte"] + (["quote_date"] if "quote_date" in candidates.columns else [])
    candidates = candidates.sort_values(sort_cols, kind="mergesort")
    idx = candidates.groupby(contract_cols, observed=True)["_dte_diff"].idxmin()
    exits = candidates.loc[idx].drop(columns=["_dte_diff"])
    return exits


def _match_entries_exits(
    entries: pd.DataFrame, data: pd.DataFrame, **kwargs: Any
) -> pd.DataFrame:
    """Match filtered entries with exit rows, compute midpoint prices, and select output columns."""
    entry_dates = kwargs.get("entry_dates")
    exit_dates = kwargs.get("exit_dates")
    exit_dte = kwargs["exit_dte"]
    exit_dte_tolerance = kwargs.get("exit_dte_tolerance", 0)
    select_closest_exit = kwargs.get("select_closest_exit", True)

    if entry_dates is not None:
        entries = _apply_signal_filter(entries, entry_dates)

    # Filter exit candidates by exit_dates BEFORE tolerance selection so that
    # a valid signal date within exit_dte_tolerance is not shadowed by a closer
    # non-signal date.
    exit_candidates = data
    if exit_dates is not None:
        exit_candidates = _apply_signal_filter(exit_candidates, exit_dates)

    # When exit_dte_tolerance > 0, retain all eligible exit candidates during
    # the merge so _remove_invalid_evaluated_options (dte_exit < dte_entry)
    # runs BEFORE selecting the closest exit per entry. Otherwise a globally
    # closer exit at dte_exit >= dte_entry would shadow a valid post-entry exit.
    exits = _get_exits(
        exit_candidates,
        exit_dte,
        exit_dte_tolerance,
        select_closest=False if exit_dte_tolerance > 0 else select_closest_exit,
    )

    merge_cols = ["underlying_symbol", "option_type", "expiration", "strike"]

    result = (
        entries.merge(right=exits, on=merge_cols, suffixes=("_entry", "_exit"))
        .assign(
            entry=lambda r: (
                (r["bid_entry"].astype("float64") + r["ask_entry"].astype("float64"))
                / 2
            ),
            exit=lambda r: (
                (r["bid_exit"].astype("float64") + r["ask_exit"].astype("float64")) / 2
            ),
        )
        .pipe(_remove_invalid_evaluated_options)
    )

    if select_closest_exit and exit_dte_tolerance > 0 and not result.empty:
        entry_key_cols = merge_cols + ["dte_entry"]
        sort_cols = ["_dte_diff", "dte_exit"] + (
            ["quote_date_exit"] if "quote_date_exit" in result.columns else []
        )
        result = (
            result.assign(_dte_diff=(result["dte_exit"] - exit_dte).abs())
            .sort_values(sort_cols, kind="mergesort")
            .drop_duplicates(subset=entry_key_cols, keep="first")
            .drop(columns=["_dte_diff"])
        )

    # Build output columns, including only those present in the result
    output_cols = [c for c in evaluated_cols if c in result.columns]
    for opt_col in (
        "delta_entry",
        "implied_volatility_entry",
        "volume_entry",
        "volume_exit",
    ):
        if opt_col in result.columns and opt_col not in output_cols:
            output_cols.append(opt_col)

    return result[output_cols]


def _evaluate_options(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """
    Evaluate options by filtering and selecting entries via delta targeting.

    Selects the closest-delta option per group within [min, max] range,
    then matches with exit rows.
    """
    # Entries must still be at or above exit_dte even when data retains lower
    # DTE rows for exit_dte_tolerance matching.
    entries = _ltrim(data, "dte", kwargs["exit_dte"])
    entries = _remove_min_bid_ask(entries, kwargs["min_bid_ask"])
    entries = _select_closest_delta(
        entries,
        kwargs["delta_target"],
        kwargs["delta_range_min"],
        kwargs["delta_range_max"],
    )
    return _match_entries_exits(entries, data, **kwargs)


def _evaluate_all_options(data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """
    Complete pipeline to evaluate all options with DTE and delta categorization.

    Filters by DTE, selects entries via delta targeting, then groups by
    DTE intervals and delta intervals.
    """
    exit_dte = kwargs["exit_dte"]
    exit_dte_tolerance = kwargs.get("exit_dte_tolerance", 0)
    lower_dte = max(0, exit_dte - exit_dte_tolerance)
    return (
        data.pipe(_assign_dte)
        .pipe(_trim, "dte", lower_dte, kwargs["max_entry_dte"])
        .pipe(_evaluate_options, **kwargs)
        .pipe(_cut_options_by_dte, kwargs["dte_interval"], kwargs["max_entry_dte"])
        .pipe(_cut_options_by_delta, kwargs.get("delta_interval", 0.05))
    )


def _calls(data: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for call options only."""
    return data[data["option_type"].str[0] == "c"]


def _puts(data: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe for put options only."""
    return data[data["option_type"].str[0] == "p"]
