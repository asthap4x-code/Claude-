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


class StmPvipStrategy(Strategy):
    name = "STM_PVIP"

    def evaluate(self, candles, context) -> Optional[Candidate]:
        run = int(clamp(int(cfg_get(context, "pvip_run", 3)), 3, 10))
        if len(candles) < max(60, run + 5):
            return None
        last = candles.iloc[-1]
        trend = last["trend"]
        if isnan(trend) or int(trend) == 0:
            return None
        trend = int(trend)
        if bool(last["is_doji"]) or int(last["color"]) != -trend:
            return None  # last candle must be a real pullback against the trend
        if not volatility_ok(candles, context):
            return None
        colors = candles["color"].iloc[:-1].astype(int).tolist()
        count = 0
        for c in reversed(colors):
            if c == trend:
                count += 1
            else:
                break
        if count < run:
            return None
        direction = CALL if trend > 0 else PUT
        score = 50 + min(count, 8) * 3 + alignment_bonus(candles, direction) * 0.8
        reasons = [f"{'up' if trend > 0 else 'down'}trend (confirmed swings), "
                   f"{count} same-colour candles then pullback"]
        return build_candidate(candles, context, self.name, direction, score, reasons)


class StmPvip2Strategy(Strategy):
    name = "STM_PVIP2"

    def evaluate(self, candles, context) -> Optional[Candidate]:
        if len(candles) < 60:
            return None
        a, b = candles.iloc[-2], candles.iloc[-1]
        atr = b["atr"]
        if isnan(atr) or atr <= 0 or isnan(b["ema10"]) or isnan(a["ema10"]):
            return None
        if context.get("last_trigger_ts") is not None and context["last_trigger_ts"] == b["time"]:
            return None
        if not volatility_ok(candles, context):
            return None
        tol = cfg_get(context, "pvip2_boundary_atr", 0.1) * float(atr)
        if abs(float(a["close"]) - float(b["open"])) > tol:
            return None
        body_min = cfg_get(context, "pvip2_body_min", 0.55)
        body_max = cfg_get(context, "pvip2_body_max", 0.85)
        cap = cfg_get(context, "pvip2_cap_enabled", True)
        br = float(b["body_ratio"])
        if br < body_min or (cap and br > body_max):
            return None
        a_lo, a_hi = min(a["open"], a["close"]), max(a["open"], a["close"])
        b_lo, b_hi = min(b["open"], b["close"]), max(b["open"], b["close"])
        direction = None
        if (int(b["color"]) > 0 and a_lo > a["ema10"] and b_lo > b["ema10"]
                and float(b["close"]) > float(a["high"])):
            direction = CALL
        elif (int(b["color"]) < 0 and a_hi < a["ema10"] and b_hi < b["ema10"]
              and float(b["close"]) < float(a["low"])):
            direction = PUT
        if direction is None:
            return None
        score = 52 + 25 * (br - body_min) / max(1e-9, (body_max - body_min)) \
            + alignment_bonus(candles, direction) * 0.7
        reasons = [f"EMA10 side + shared boundary + breakout, body {br:.0%} of range"]
        return build_candidate(candles, context, self.name, direction, score, reasons)
