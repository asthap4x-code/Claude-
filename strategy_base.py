"""Shared strategy interface and tiny helpers (no indicator maths lives here).

Every strategy receives a candles DataFrame that main.prepare() has already
enriched with indicator columns, plus a context dict {"pair": str, "cfg": obj}.
Strategies only look at CLOSED candles (the last row is the newest candle the
engine decided to hand over) and never see anything newer than that row.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

CALL = "CALL"
PUT = "PUT"


@dataclass
class Candidate:
    pair: str
    strategy: str
    direction: str            # "CALL" or "PUT"
    score: float              # rule-agreement score 0..100 (NOT a win probability)
    ref_price: float          # close of the last candle used
    candle_ts: pd.Timestamp   # open time of the last candle used
    reasons: list = field(default_factory=list)
    experimental: bool = False

