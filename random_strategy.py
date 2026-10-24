"""RANDOM strategy: EXPERIMENTAL, for software testing only.

Outputs a coin-flip direction. It has no predictive value and must never be
presented as an AI prediction or a verified signal. It is disabled unless
ENABLE_RANDOM=true, and every message carries an EXPERIMENTAL label.
"""
from __future__ import annotations

import random
from typing import Optional

from strategy_base import CALL, PUT, Candidate, Strategy, build_candidate


class RandomStrategy(Strategy):
    name = "RANDOM"
    experimental = True

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def evaluate(self, candles, context) -> Optional[Candidate]:
        if len(candles) < 2:
            return None
        direction = self.rng.choice([CALL, PUT])
        return build_candidate(candles, context, self.name, direction, 50.0,
                               ["random coin flip (experimental, no edge)"],
                               experimental=True)
