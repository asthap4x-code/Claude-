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


class Strategy(ABC):
    name: str = "BASE"
    experimental: bool = False

    @abstractmethod
    def evaluate(self, candles: pd.DataFrame, context: dict) -> Optional[Candidate]:
        """Return a Candidate or None (no signal)."""


def cfg_get(context: dict, key: str, default: Any) -> Any:
    cfg = context.get("cfg")
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def isnan(x: Any) -> bool:
    try:
        return x is None or math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def sign(x: float) -> int:
    if isnan(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def direction_sign(direction: str) -> int:
    return 1 if direction == CALL else -1


def alignment_bonus(df: pd.DataFrame, direction: str) -> float:
    """0..18 points when trend, EMA20 side and MACD histogram agree with direction."""
    row = df.iloc[-1]
    s = direction_sign(direction)
    bonus = 0.0
    if not isnan(row.get("ema20")) and sign(row["close"] - row["ema20"]) == s:
        bonus += 5
    if not isnan(row.get("trend")) and int(row["trend"]) == s:
        bonus += 10
    if not isnan(row.get("macd_hist")) and sign(row["macd_hist"]) == s:
        bonus += 3
    return bonus


def volatility_ok(df: pd.DataFrame, context: dict) -> bool:
    """Reject dead-flat or spiking markets (ATR relative to its 50-bar mean)."""
    row = df.iloc[-1]
    ar = row.get("atr_ratio")
    if isnan(ar):
        return False
    lo = cfg_get(context, "vol_min_ratio", 0.35)
    hi = cfg_get(context, "vol_max_ratio", 2.8)
    return lo <= float(ar) <= hi


def build_candidate(df: pd.DataFrame, context: dict, name: str, direction: str,
                    score: float, reasons: list, experimental: bool = False) -> Candidate:
    row = df.iloc[-1]
    return Candidate(
        pair=str(context.get("pair", "?")),
        strategy=name,
        direction=direction,
        score=round(clamp(score, 0, 99), 1),
        ref_price=float(row["close"]),
        candle_ts=row["time"],
        reasons=reasons,
        experimental=experimental,
    )
