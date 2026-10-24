"""BLACK4X: copy the colour of the previous fully closed candle.

Rule (exact):
  * last candle is a doji (body/range <= DOJI_THRESHOLD, or open == close, or zero range) -> no signal
  * last candle bullish (close > open) -> CALL
  * last candle bearish (close < open) -> PUT
"""
from __future__ import annotations

from typing import Optional

from strategy_base import (CALL, PUT, Candidate, Strategy, alignment_bonus,
                           build_candidate, volatility_ok)


class Black4xStrategy(Strategy):
    name = "BLACK4X"

    def evaluate(self, candles, context) -> Optional[Candidate]:
        if len(candles) < 60:
            return None
        row = candles.iloc[-1]
        if bool(row["is_doji"]) or int(row["color"]) == 0:
            return None
        if not volatility_ok(candles, context):
            return None
        direction = CALL if int(row["color"]) > 0 else PUT
        score = 35 + 35 * min(float(row["body_ratio"]), 1.0) + alignment_bonus(candles, direction)
        reasons = [f"previous candle {'bullish' if direction == CALL else 'bearish'} "
                   f"(body {row['body_ratio']:.0%} of range)"]
        return build_candidate(candles, context, self.name, direction, score, reasons)
