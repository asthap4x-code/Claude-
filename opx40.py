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
