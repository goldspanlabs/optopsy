"""Signal combinators, the Signal class, signal_dates, and custom_signal."""

import pandas as pd

from ..timestamps import normalize_dates
from ._helpers import SignalFunc, _groupby_symbol

# ---------------------------------------------------------------------------
# Signal combinators
# ---------------------------------------------------------------------------


def _requires_per_strike(sig: SignalFunc | "Signal") -> bool:
    """Return True if *sig* needs per-strike option rows rather than 1 row/date."""
    return bool(getattr(sig, "requires_per_strike", False))


def _eval_signal_on_data(
    sig: SignalFunc | "Signal",
    data: pd.DataFrame,
    *,
    per_strike_context: bool,
) -> "pd.Series[bool]":
    """Evaluate *sig* on *data*, deduplicating to 1 row per (symbol, date) for
    non-per-strike signals when *data* contains multiple option rows per date.
    """
    if (
        not per_strike_context
        or _requires_per_strike(sig)
        or data.empty
        or "underlying_symbol" not in data.columns
        or "quote_date" not in data.columns
    ):
        return sig(data)

    norm_dates = normalize_dates(pd.to_datetime(data["quote_date"]))
    dedup_keys = pd.DataFrame(
        {
            "underlying_symbol": data["underlying_symbol"].values,
            "_norm_date": norm_dates.values,
        },
        index=data.index,
    )
    first_mask = ~dedup_keys.duplicated(
        subset=["underlying_symbol", "_norm_date"], keep="first"
    )
    if first_mask.all():
        return sig(data)

    daily = data.loc[first_mask].copy()
    daily_norm_dates = norm_dates.loc[first_mask]
    daily_mask = sig(daily).fillna(False).astype(bool)

    lookup = pd.Series(
        daily_mask.values,
        index=pd.MultiIndex.from_arrays(
            [daily["underlying_symbol"].values, daily_norm_dates.values]
        ),
    )
    keys = pd.MultiIndex.from_arrays(
        [data["underlying_symbol"].values, norm_dates.values]
    )
    return pd.Series(
        lookup.reindex(keys).fillna(False).values,
        index=data.index,
        dtype=bool,
    )


def and_signals(*signals: SignalFunc) -> SignalFunc:
    """Combine multiple signals with logical AND.

    All signals must be True for a date to be valid.
    """
    per_strike = any(_requires_per_strike(s) for s in signals)

    def combined(data: pd.DataFrame) -> "pd.Series[bool]":
        result = pd.Series(True, index=data.index)
        for sig in signals:
            result = result & _eval_signal_on_data(
                sig, data, per_strike_context=per_strike
            )
            if not result.any():
                return result
        return result

    if per_strike:
        combined.requires_per_strike = True  # type: ignore[attr-defined]
    return combined


def or_signals(*signals: SignalFunc) -> SignalFunc:
    """Combine multiple signals with logical OR.

    At least one signal must be True for a date to be valid.
    """
    per_strike = any(_requires_per_strike(s) for s in signals)

    def combined(data: pd.DataFrame) -> "pd.Series[bool]":
        result = pd.Series(False, index=data.index)
        for sig in signals:
            result = result | _eval_signal_on_data(
                sig, data, per_strike_context=per_strike
            )
            if result.all():
                return result
        return result

    if per_strike:
        combined.requires_per_strike = True  # type: ignore[attr-defined]
    return combined


# ---------------------------------------------------------------------------
# Sustained combinator
# ---------------------------------------------------------------------------


def sustained(signal_func: SignalFunc, days: int = 5) -> SignalFunc:
    """True only when signal_func has been True for at least ``days`` consecutive bars.

    Uses a rolling minimum over the boolean output of the inner signal: the
    window minimum is 1 only when every bar in the window was True.

    Args:
        signal_func: Any SignalFunc to wrap
        days: Minimum consecutive True bars required (default 5)

    Raises:
        ValueError: If days < 1
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")

    per_strike = _requires_per_strike(signal_func)

    def _signal(data: pd.DataFrame) -> "pd.Series[bool]":
        if data.empty:
            return pd.Series(False, index=data.index, dtype=bool)
        if per_strike:
            # Evaluate on full per-strike data, collapse to 1 bool per (symbol, date),
            # compute rolling minimum across consecutive dates, then map back.
            raw = signal_func(data).fillna(False).astype(bool)
            norm_dates = normalize_dates(pd.to_datetime(data["quote_date"]))
            daily = (
                pd.DataFrame(
                    {
                        "underlying_symbol": data["underlying_symbol"].values,
                        "quote_date": norm_dates.values,
                        "_flag": raw.values,
                    }
                )
                .groupby(["underlying_symbol", "quote_date"], sort=True)["_flag"]
                .any()
                .reset_index()
            )
            daily["_sustained"] = (
                daily.groupby("underlying_symbol", sort=False)["_flag"]
                .transform(lambda s: s.astype(int).rolling(days).min().fillna(0))
                .astype(bool)
            )
            lookup = pd.Series(
                daily["_sustained"].values,
                index=pd.MultiIndex.from_arrays(
                    [daily["underlying_symbol"], daily["quote_date"]]
                ),
            )
            keys = pd.MultiIndex.from_arrays([data["underlying_symbol"], norm_dates])
            return pd.Series(
                lookup.reindex(keys).fillna(False).values,
                index=data.index,
                dtype=bool,
            )

        def _compute_group(group: pd.DataFrame) -> "pd.Series[bool]":
            raw = signal_func(group).fillna(False).astype(int)
            return raw.rolling(days).min().fillna(0).astype(bool)

        return _groupby_symbol(data, _compute_group)

    if per_strike:
        _signal.requires_per_strike = True  # type: ignore[attr-defined]
    return _signal


# ---------------------------------------------------------------------------
# Custom signal from pre-flagged DataFrame
# ---------------------------------------------------------------------------


def custom_signal(df: pd.DataFrame, flag_col: str = "signal") -> SignalFunc:
    """Create a signal function from a DataFrame with a boolean flag column.

    Args:
        df: DataFrame with at least ``underlying_symbol``, ``quote_date``, and
            the boolean flag column.
        flag_col: Name of the column whose truthy values mark valid signal dates.
    """
    required = {"underlying_symbol", "quote_date", flag_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    flags = df[flag_col].fillna(False).astype(bool)
    valid = df.loc[flags, ["underlying_symbol", "quote_date"]].copy()
    valid["quote_date"] = normalize_dates(pd.to_datetime(valid["quote_date"]))
    valid_idx = pd.MultiIndex.from_arrays(
        [valid["underlying_symbol"], valid["quote_date"]]
    )

    def _signal(data: pd.DataFrame) -> "pd.Series[bool]":
        dates = normalize_dates(pd.to_datetime(data["quote_date"]))
        lookup = pd.MultiIndex.from_arrays([data["underlying_symbol"], dates])
        return pd.Series(lookup.isin(valid_idx), index=data.index, dtype=bool)

    return _signal


# ---------------------------------------------------------------------------
# Fluent Signal class
# ---------------------------------------------------------------------------


class Signal:
    """Composable wrapper around a SignalFunc with & and | operator support."""

    def __init__(self, func: SignalFunc) -> None:
        self._func = func
        if _requires_per_strike(func):
            self.requires_per_strike = True

    def __call__(self, data: pd.DataFrame) -> "pd.Series[bool]":
        return self._func(data)

    def __and__(self, other: "Signal | SignalFunc") -> "Signal":
        per_strike = _requires_per_strike(self) or _requires_per_strike(other)

        def combined(data: pd.DataFrame) -> "pd.Series[bool]":
            left = _eval_signal_on_data(self, data, per_strike_context=per_strike)
            if not left.any():
                return left
            return left & _eval_signal_on_data(
                other, data, per_strike_context=per_strike
            )

        if per_strike:
            combined.requires_per_strike = True  # type: ignore[attr-defined]
        return Signal(combined)

    def __or__(self, other: "Signal | SignalFunc") -> "Signal":
        per_strike = _requires_per_strike(self) or _requires_per_strike(other)

        def combined(data: pd.DataFrame) -> "pd.Series[bool]":
            left = _eval_signal_on_data(self, data, per_strike_context=per_strike)
            if left.all():
                return left
            return left | _eval_signal_on_data(
                other, data, per_strike_context=per_strike
            )

        if per_strike:
            combined.requires_per_strike = True  # type: ignore[attr-defined]
        return Signal(combined)

    def __repr__(self) -> str:
        return f"Signal({self._func!r})"


def signal(func: SignalFunc) -> Signal:
    """Wrap a signal function in a Signal for fluent operator chaining."""
    return Signal(func)


def signal_dates(
    stock_data: pd.DataFrame,
    signal_func: SignalFunc,
) -> pd.DataFrame:
    """Run a signal on stock data and return valid (symbol, date) pairs.

    This is the recommended way to generate ``entry_dates`` / ``exit_dates``
    for strategy functions.  Signals are computed entirely from stock price
    data — no options data is involved (except for per-strike IV rank signals,
    which preserve all option rows when ``requires_per_strike`` is set).

    Args:
        stock_data: OHLCV DataFrame with at least ``underlying_symbol``,
            ``quote_date``, and ``close``.  Extra columns (``high``, ``low``,
            ``volume``) are used by signals that need them.  Accepts output
            from ``load_cached_stocks()`` directly.
        signal_func: A signal function (e.g. ``op.rsi_below()``) or a
            composed signal using ``&`` / ``|`` operators on ``Signal``
            objects.

    Returns:
        DataFrame with columns ``(underlying_symbol, quote_date)`` for
        dates where the signal is True.  Pass this directly to a strategy
        via ``entry_dates`` or ``exit_dates``.

    Example::

        stocks = op.load_cached_stocks("SPY")
        entry = op.signal_dates(stocks, op.rsi_below(threshold=30))
        results = op.long_calls(options, entry_dates=entry)

    Example with composed signals::

        entry = op.signal_dates(
            stocks,
            op.signal(op.rsi_below()) & op.signal(op.sma_above(200)),
        )
    """
    df = stock_data.copy()
    df["quote_date"] = normalize_dates(pd.to_datetime(df["quote_date"]))
    if not _requires_per_strike(signal_func):
        df = df.drop_duplicates(["underlying_symbol", "quote_date"])
    df = df.sort_values(["underlying_symbol", "quote_date"]).reset_index(drop=True)
    mask = signal_func(df)
    return (
        df.loc[mask, ["underlying_symbol", "quote_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
