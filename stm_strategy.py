"""STM PVIP and STM PVIP 2 as configurable, documented rules.

---------------------------------------------------------------- STM PVIP
Trend  : confirmed swing structure (main.swing_structure). UP only if the last two
         CONFIRMED swing highs AND lows are both rising; DOWN if both falling;
         otherwise RANGE (no signal). A swing is usable only `right` bars after it
         formed, so there is no look-ahead.
Setup  : uptrend, the `pvip_run` (3..10, default 3) candles BEFORE the last candle
         are all bullish, and the last (closed) candle is a bearish, non-doji
         pullback -> CALL for the next eligible candle. Downtrend is mirrored.
Filters: ATR ratio inside [vol_min_ratio, vol_max_ratio]; indicators warmed up.

------------------------------------------------------------- STM PVIP 2
Body % : body / (high - low) of the breakout candle (NOT close position).
Setup  : candles A = [-2] and B = [-1] (adjacent). They share a boundary price when
         |A.close - B.open| <= pvip2_boundary_atr * ATR.
         CALL: A and B bodies are both fully above EMA10 (min(open, close) > EMA10 on
               their own rows), B is bullish, B.body% in [pvip2_body_min, pvip2_body_max]
               (cap only if pvip2_cap_enabled) and B closes above A.high (breakout).
         PUT : mirrored (bodies below EMA10, B bearish, B closes below A.low).
Dupes  : context["last_trigger_ts"] (open time of the last triggering candle) stops the
         same candle from firing twice; the engine also dedupes in the database.
"""
from __future__ import annotations

from typing import Optional

from strategy_base import (CALL, PUT, Candidate, Strategy, alignment_bonus,
                           build_candidate, cfg_get, clamp, isnan, volatility_ok)

