"""Internal helpers for options strategy processing.

Contains the ``Side`` enum and private helper functions that assemble leg
definitions and call the core strategy engine.  These are not part of the
public API.

Parameter defaults and validation are handled by the Pydantic models in
``types.py`` (``StrategyParams``, ``CalendarStrategyParams``).  Helpers
pass raw user ``kwargs`` through to ``_process_strategy()`` /
``_process_calendar_strategy()``, which validate and apply defaults.
"""

from enum import Enum
from typing import List, Optional, Tuple, Unpack

import pandas as pd

from ..core import (
    _process_calendar_strategy,
    _process_strategy,
)
from ..definitions import (
    calendar_spread_external_cols,
    calendar_spread_internal_cols,
    diagonal_spread_external_cols,
    diagonal_spread_internal_cols,
    double_strike_internal_cols,
    quadruple_strike_internal_cols,
    single_strike_internal_cols,
    straddle_internal_cols,
    triple_strike_internal_cols,
)
from ..evaluation import _puts
from ..rules import (
    _rule_butterfly_strikes,
    _rule_collar_strikes,
    _rule_diagonal_spread,
    _rule_expiration_ordering,
    _rule_iron_butterfly_strikes,
    _rule_iron_condor_strikes,
    _rule_non_overlapping_strike,
)
from ..types import CalendarStrategyParamsDict, StrategyParamsDict
from ._stock import _stock_with_options


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
        join_on=[*_DEFAULT_JOIN_ON[:2], "strike", *_DEFAULT_JOIN_ON[2:]],
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
            via ``_stock._normalize_stock_data()``.
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
