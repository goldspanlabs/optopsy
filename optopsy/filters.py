"""Filtering primitives for option chain data.

Functions in this module apply row-level filters to DataFrames — by DTE,
OTM percentage, delta, bid/ask thresholds, and signal dates.  They are
pure functions with no side effects and no dependencies on other optopsy
modules.
"""

from typing import Any

import numpy as np
import pandas as pd

from .timestamps import normalize_dates


def _assign_dte(data: pd.DataFrame) -> pd.DataFrame:
    """Assign days to expiration (DTE) to the dataset."""
    return data.assign(dte=lambda r: (r["expiration"] - r["quote_date"]).dt.days)


def _trim(data: pd.DataFrame, col: str, lower: Any, upper: Any) -> pd.DataFrame:
    """Filter dataframe rows where column value is between lower and upper bounds."""
    return data.loc[(data[col] >= lower) & (data[col] <= upper)]


def _ltrim(data: pd.DataFrame, col: str, lower: Any) -> pd.DataFrame:
    """Filter dataframe rows where column value is greater than or equal to lower bound."""
    return data.loc[data[col] >= lower]


def _rtrim(data: pd.DataFrame, col: str, upper: Any) -> pd.DataFrame:
    """Filter dataframe rows where column value is less than or equal to upper bound."""
    return data.loc[data[col] <= upper]


def _get(data: pd.DataFrame, col: str, val: Any) -> pd.DataFrame:
    """Filter dataframe rows where column equals specified value."""
    return data.loc[data[col] == val]


def _remove_min_bid_ask(data: pd.DataFrame, min_bid_ask: float) -> pd.DataFrame:
    """Remove options with bid or ask prices below minimum threshold."""
    return data.loc[(data["bid"] > min_bid_ask) & (data["ask"] > min_bid_ask)]


def _remove_invalid_evaluated_options(data: pd.DataFrame) -> pd.DataFrame:
    """Keep evaluated options where entry DTE is greater than exit DTE."""
    return data.loc[
        (data["dte_exit"] <= data["dte_entry"])
        & (data["dte_entry"] != data["dte_exit"])
    ]


def _apply_signal_filter(
    data: pd.DataFrame,
    valid_dates: pd.DataFrame,
    date_col: str = "quote_date",
) -> pd.DataFrame:
    """
    Filter data to only include rows matching valid (symbol, date) pairs.

    Normalizes ``valid_dates["quote_date"]`` to date-only tz-naive timestamps,
    drops duplicate ``(underlying_symbol, quote_date)`` pairs, and selects only
    the join key columns so extra columns in caller-supplied DataFrames cannot
    cause suffix collisions or duplicate rows.

    Args:
        data: DataFrame to filter (already date-normalized)
        valid_dates: DataFrame with at least (underlying_symbol, quote_date)
            of valid dates
        date_col: Name of the date column in data to match against (default: quote_date)

    Returns:
        Filtered DataFrame
    """
    keys = valid_dates[["underlying_symbol", "quote_date"]].copy()
    keys["quote_date"] = normalize_dates(pd.to_datetime(keys["quote_date"]))
    keys = keys.drop_duplicates(subset=["underlying_symbol", "quote_date"])
    if date_col != "quote_date":
        keys = keys.rename(columns={"quote_date": date_col})
    return data.merge(keys, on=["underlying_symbol", date_col], how="inner")


def _select_closest_delta(
    data: pd.DataFrame, target: float, delta_min: float, delta_max: float
) -> pd.DataFrame:
    """Select the option with abs(delta) closest to target per group.

    Filters to options within [delta_min, delta_max] abs(delta) range,
    then picks the single closest-to-target row per
    (underlying_symbol, quote_date, expiration, option_type) group.
    """
    # Reset index so duplicate input index labels cannot cause .loc[idx] to
    # return multiple rows per group.
    data = data.reset_index(drop=True)
    data["_abs_delta"] = data["delta"].abs()
    data = data.loc[
        (data["_abs_delta"] >= delta_min) & (data["_abs_delta"] <= delta_max)
    ]

    if data.empty:
        return data.drop(columns=["_abs_delta"])

    data["_delta_diff"] = (data["_abs_delta"] - target).abs()
    group_cols = ["underlying_symbol", "quote_date", "expiration", "option_type"]
    # Sort by strike first so idxmin() deterministically picks lowest strike on ties
    data = data.sort_values("strike", kind="mergesort")
    idx = data.groupby(group_cols, observed=True)["_delta_diff"].idxmin()
    result = data.loc[idx].drop(columns=["_abs_delta", "_delta_diff"])
    return result


def _cut_options_by_dte(
    data: pd.DataFrame, dte_interval: int, max_entry_dte: int
) -> pd.DataFrame:
    """Categorize options into DTE intervals for grouping."""
    dte_intervals = list(range(0, max_entry_dte + dte_interval, dte_interval))
    data["dte_range"] = pd.cut(data["dte_entry"], dte_intervals)
    return data


def _cut_options_by_delta(
    data: pd.DataFrame,
    delta_interval: float,
    col: str = "delta_entry",
    out_col: str = "delta_range",
) -> pd.DataFrame:
    """
    Categorize options into delta intervals for grouping.

    Args:
        data: DataFrame with delta column
        delta_interval: Interval size for delta grouping
        col: Source delta column name (default: delta_entry)
        out_col: Output interval column name (default: delta_range)

    Returns:
        DataFrame with delta interval column added
    """
    # Delta ranges from -1 to 1 for puts and calls.
    # Round to 6 decimals so fine intervals (e.g. 0.005, 0.025) never collide
    # into duplicate bin edges, and include_lowest=True so delta == -1.0 is kept.
    delta_intervals = np.round(
        np.arange(-1.0, 1.0 + delta_interval, delta_interval), 6
    ).tolist()
    data[out_col] = pd.cut(data[col], delta_intervals, include_lowest=True)
    return data
