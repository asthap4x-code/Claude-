"""AI strategy: transparent, deterministic confluence scoring (no ML yet).

Each component votes a value in [-1, +1] (+1 = bullish / CALL). The weighted sum of
votes is in [-100, +100]. Direction = sign of the sum. The reported score is
min(99, |sum| * 1.25): a RULE-AGREEMENT score, not a win probability. Use the
backtest's bucket table (python main.py --backtest) to see realised hit-rates.

Weights (sum 100): trend 15, ema 10, stoch 10, rsi 5, macd 10, sr 15, zone 10,
liquidity 8, structure 7, price_action 10.
A signal needs |sum| >= ai_min_total (default 40), at least ai_min_agree (default 4)
components agreeing, and the volatility filter to pass.

Only confirmed swing levels (sh1, sh2, sl1, sl2 - available `right` bars after the
swing formed) are used as support/resistance. A plain touch of a level never signals
on its own; breakout/retest and failed-breakout need a confirmed rejection candle.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from strategy_base import (CALL, PUT, Candidate, Strategy, build_candidate,
                           cfg_get, clamp, isnan, sign, volatility_ok)

WEIGHTS = {"trend": 15, "ema": 10, "stoch": 10, "rsi": 5, "macd": 10,
           "sr": 15, "zone": 10, "liquidity": 8, "structure": 7, "price_action": 10}


def retest_vote(df, atr: float, lookback: int = 10):
