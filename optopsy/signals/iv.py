"""IV rank signals — require options data with implied_volatility column."""

import operator

import pandas as pd

from ..timestamps import normalize_dates
from ._helpers import SignalFunc

# ---------------------------------------------------------------------------
# IV rank computation
# ---------------------------------------------------------------------------


def _compute_atm_iv(options_data: pd.DataFrame) -> pd.DataFrame:
    """Compute the ATM implied volatility per (symbol, quote_date).

    For each quote_date, finds the option(s) with strike closest to
    the stock price (``close`` column) and averages their implied
    volatility.  Returns an empty DataFrame when the ``close`` column
    is absent.

    Callers should ensure ``resolve_price_column()`` or
    ``signal_dates()`` has been applied before invoking this function
    so that ``close`` is available.
    """
    if "close" in options_data.columns:
        price_col = "close"
    elif "underlying_price" in options_data.columns:
        price_col = "underlying_price"
    else:
        return pd.DataFrame(
            columns=["underlying_symbol", "quote_date", "implied_volatility"]
        )

    _empty_cols = [
        "underlying_symbol",
        "quote_date",
        "implied_volatility",
    ]
    df = options_data.dropna(
        subset=["implied_volatility", "strike", price_col]
    ).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=_empty_cols)

    df["quote_date"] = normalize_dates(pd.to_datetime(df["quote_date"]))

    if "expiration" in df.columns:
        df["expiration"] = normalize_dates(pd.to_datetime(df["expiration"]))
        df["_dte"] = (df["expiration"] - df["quote_date"]).dt.days
        df = df[df["_dte"] > 0]
        if df.empty:
            return pd.DataFrame(columns=_empty_cols)
        nearest_dte = df.groupby(["underlying_symbol", "quote_date"])["_dte"].transform(
            "min"
        )
        df = df[df["_dte"] == nearest_dte].drop(columns=["_dte"])

    df["_abs_otm"] = (df["strike"] - df[price_col]).abs()
    idx = df.groupby(["underlying_symbol", "quote_date"])["_abs_otm"].idxmin()
    atm = df.loc[idx, ["underlying_symbol", "quote_date", "strike"]].drop_duplicates()

    # Semi-join to ATM strikes first so groupby().mean() only aggregates ATM rows
    atm_rows = df.merge(
        atm, on=["underlying_symbol", "quote_date", "strike"], how="inner"
    )
    return (
        atm_rows.groupby(["underlying_symbol", "quote_date"])["implied_volatility"]
        .mean()
        .reset_index()
        .sort_values(["underlying_symbol", "quote_date"])
        .reset_index(drop=True)
    )


def _compute_iv_rank_series(atm_iv: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute IV rank: ``(current - min) / (max - min)`` over rolling window."""
    if atm_iv.empty:
        return pd.Series(float("nan"), index=atm_iv.index)

    def _rank_iv(iv: pd.Series) -> pd.Series:
        rolling_min = iv.rolling(window, min_periods=1).min()
        rolling_max = iv.rolling(window, min_periods=1).max()
        denom = rolling_max - rolling_min
        rank = (iv - rolling_min) / denom.replace(0, float("nan"))
        return rank.fillna(0.5)

    return (
        atm_iv.groupby("underlying_symbol", sort=False)["implied_volatility"]
        .transform(_rank_iv)
        .reindex(atm_iv.index, fill_value=float("nan"))
    )


def _iv_rank_signal(threshold: float, window: int, compare_op) -> SignalFunc:
    """Factory for IV rank signals."""

    def _signal(data: pd.DataFrame) -> "pd.Series[bool]":
        if "implied_volatility" not in data.columns:
            return pd.Series(False, index=data.index)
        atm_iv = _compute_atm_iv(data)
        if atm_iv.empty:
            return pd.Series(False, index=data.index)
        rank = _compute_iv_rank_series(atm_iv, window)
        rank_lookup = pd.Series(
            rank.values,
            index=pd.MultiIndex.from_arrays(
                [atm_iv["underlying_symbol"], atm_iv["quote_date"]]
            ),
        )
        norm_dates = normalize_dates(pd.to_datetime(data["quote_date"]))
        keys = pd.MultiIndex.from_arrays([data["underlying_symbol"], norm_dates])
        iv_rank_for_rows = pd.Series(rank_lookup.reindex(keys).values, index=data.index)
        return compare_op(iv_rank_for_rows, threshold).fillna(False)

    _signal.requires_per_strike = True  # type: ignore[attr-defined]
    return _signal


def iv_rank_above(threshold: float = 0.5, window: int = 252) -> SignalFunc:
    """True when IV rank exceeds a threshold."""
    return _iv_rank_signal(threshold, window, operator.gt)


def iv_rank_below(threshold: float = 0.5, window: int = 252) -> SignalFunc:
    """True when IV rank is below a threshold."""
    return _iv_rank_signal(threshold, window, operator.lt)
