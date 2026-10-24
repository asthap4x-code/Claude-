"""OPX4O: opposite-direction rule for doji candles.

Colour convention (documented, configurable by editing this file):
  * green doji  = close > open (tiny bullish body)  -> PUT  (opposite)
  * red doji    = close < open (tiny bearish body)  -> CALL (opposite)
  * TRUE doji   = close == open exactly             -> no signal (colour undefined)
  * non-doji candle                                  -> no signal (rule is doji-only)
A doji means body/range <= DOJI_THRESHOLD (default 0.10). Zero-range candles
(no movement at all) are skipped because they carry no information.
"""
from __future__ import annotations

from typing import Optional

from strategy_base import (CALL, PUT, Candidate, Strategy, build_candidate,
                           cfg_get, isnan)


class Opx4oStrategy(Strategy):
    name = "OPX4O"

    def evaluate(self, candles, context) -> Optional[Candidate]:
        if len(candles) < 60:
            return None
        row = candles.iloc[-1]
        if not bool(row["is_doji"]):
            return None
        if float(row["rng"]) <= 0:
            return None
        color = int(row["color"])
        if color == 0:
            return None  # true doji: ambiguous colour, skipped by design
        atr = row["atr"]
        if isnan(atr) or atr <= 0:
            return None
        min_rng = cfg_get(context, "opx4o_min_range_atr", 0.3)
        if float(row["rng"]) < min_rng * float(atr):
            return None  # too small to be a meaningful candle
        direction = PUT if color > 0 else CALL
        score = 45 + 20 * min(float(row["rng"]) / float(atr), 1.5) / 1.5
        rsi = row["rsi"]
        if not isnan(rsi):
            if (direction == CALL and rsi < 45) or (direction == PUT and rsi > 55):
                score += 5
        reasons = [f"{'green' if color > 0 else 'red'} doji "
                   f"(body {row['body_ratio']:.0%} of range) -> opposite"]
        return build_candidate(candles, context, self.name, direction, score, reasons)
