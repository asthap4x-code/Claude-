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
    """Breakout/retest and failed-breakout votes at confirmed swing levels."""
    last = df.iloc[-1]
    levels = [last.get(k) for k in ("sh1", "sh2", "sl1", "sl2")]
    levels = [float(x) for x in levels if not isnan(x)]
    if not levels:
        return 0.0, ""
    win = df.iloc[-(lookback + 6):-1]
    closes = win["close"].to_numpy(dtype=float)
    tol, buf = 0.3 * atr, 0.1 * atr
    c, o, h, lo = (float(last["close"]), float(last["open"]),
                   float(last["high"]), float(last["low"]))
    best, why = 0.0, ""
    for L in levels:
        # breakdown of L (was support) then bearish retest from below -> PUT
        below = np.where(closes < L - buf)[0]
        if below.size:
            k = below[0]
            if (closes[:k] > L).any() and (closes[k:] < L + buf).all() \
                    and h >= L - tol and c < L - 0.05 * atr and c < o:
                return -1.0, f"retest of broken level {L:.5g} from below, bearish rejection"
        # breakout of L (was resistance) then bullish retest from above -> CALL
        above = np.where(closes > L + buf)[0]
        if above.size:
            k = above[0]
            if (closes[:k] < L).any() and (closes[k:] > L - buf).all() \
                    and lo <= L + tol and c > L + 0.05 * atr and c > o:
                return 1.0, f"retest of broken level {L:.5g} from above, bullish support"
        # failed breakout (trap) in the last 3 closed candles before the last one
        recent = closes[-3:]
        if (recent > L + buf).any() and c < L and c < o:
            best, why = -0.8, f"failed breakout above {L:.5g}"
        elif (recent < L - buf).any() and c > L and c > o:
            best, why = 0.8, f"failed breakdown below {L:.5g}"
    return best, why


def zone_vote(df, atr: float):
    """Supply/demand zone reaction. Zone = base candle before an impulse candle
    (body >= 1.5 ATR); valid until a later close breaks it; ATR-widened."""
    n = len(df)
    o, c = df["open"].to_numpy(float), df["close"].to_numpy(float)
    h, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    a = df["atr"].to_numpy(float)
    pad = 0.1 * atr
    for i in range(n - 2, max(n - 60, 2), -1):
        if np.isnan(a[i - 1]) or abs(c[i] - o[i]) < 1.5 * a[i - 1]:
            continue
        b = i - 1
        if c[i] > o[i]:  # demand
            zlo, zhi = lo[b] - pad, max(o[b], c[b]) + pad
            if (c[i + 1:] < zlo).any():
                continue
            if lo[-1] <= zhi and c[-1] > zlo and c[-1] > o[-1] and i < n - 1:
                return 1.0, f"reaction from demand zone {zlo:.5g}-{zhi:.5g}"
        else:  # supply
            zlo, zhi = min(o[b], c[b]) - pad, h[b] + pad
            if (c[i + 1:] > zhi).any():
                continue
            if h[-1] >= zlo and c[-1] < zhi and c[-1] < o[-1] and i < n - 1:
                return -1.0, f"reaction from supply zone {zlo:.5g}-{zhi:.5g}"
    return 0.0, ""


def pattern_vote(prev, last):
    po, pc, o, c = float(prev["open"]), float(prev["close"]), float(last["open"]), float(last["close"])
    rng = float(last["rng"])
    if rng <= 0:
        return 0.0, ""
    body = abs(c - o)
    uw, lw = float(last["upper_wick"]), float(last["lower_wick"])
    if pc < po and c > o and o <= pc and c >= po:
        return 1.0, "bullish engulfing"
    if pc > po and c < o and o >= pc and c <= po:
        return -1.0, "bearish engulfing"
    if body > 0 and lw >= 2 * body and uw <= 0.3 * rng:
        return 0.8, "hammer / bullish pin"
    if body > 0 and uw >= 2 * body and lw <= 0.3 * rng:
        return -0.8, "shooting star / bearish pin"
    if float(last["body_ratio"]) >= 0.7:
        return 0.5 * sign(c - o), "momentum candle"
    return 0.0, ""


class AiStrategy(Strategy):
    name = "AI"

    def evaluate(self, candles, context) -> Optional[Candidate]:
        df = candles
        if len(df) < 80:
            return None
        last, prev = df.iloc[-1], df.iloc[-2]
        atr = last["atr"]
        if isnan(atr) or atr <= 0 or not volatility_ok(df, context):
            return None
        atr = float(atr)
        votes, why = {}, {}

        t = 0.0
        if not isnan(last["trend"]):
            t += 0.6 * int(last["trend"])
        if last["ema10"] > last["ema20"] > last["ema50"]:
            t += 0.4
        elif last["ema10"] < last["ema20"] < last["ema50"]:
            t -= 0.4
        votes["trend"], why["trend"] = t, "trend structure + EMA stack"

        votes["ema"] = 0.5 * sign(last["close"] - last["ema20"]) + 0.5 * sign(last["ema10"] - prev["ema10"])
        why["ema"] = "price vs EMA20 and EMA10 slope"

        k, d, pk, pd_ = last["stoch_k"], last["stoch_d"], prev["stoch_k"], prev["stoch_d"]
        sv = 0.0
        if not any(isnan(x) for x in (k, d, pk, pd_)):
            if k > d and pk <= pd_ and k < 40:
                sv = 1.0
            elif k < d and pk >= pd_ and k > 60:
                sv = -1.0
            elif k < 20:
                sv = 0.5
            elif k > 80:
                sv = -0.5
        votes["stoch"], why["stoch"] = sv, "stochastic cross / extreme"

        r = last["rsi"]
        votes["rsi"] = 0.0 if isnan(r) else (1.0 if r < 35 else -1.0 if r > 65 else 0.0)
        why["rsi"] = "RSI extreme"

        h0, h1 = last["macd_hist"], prev["macd_hist"]
        mv = 0.0
        if not isnan(h0) and not isnan(h1):
            mv = 0.5 * sign(h0) + (0.5 if (h0 > h1) else -0.5 if (h0 < h1) else 0.0)
        votes["macd"], why["macd"] = mv, "MACD histogram"

        votes["sr"], why["sr"] = retest_vote(df, atr)
        votes["zone"], why["zone"] = zone_vote(df, atr)

        lv, lw = 0.0, ""
        sh, sl = last["sh1"], last["sl1"]
        rng = float(last["rng"])
        if rng > 0:
            if not isnan(sh) and last["high"] > sh and last["close"] < sh and last["upper_wick"] >= 0.4 * rng:
                lv, lw = -1.0, "liquidity sweep above swing high"
            elif not isnan(sl) and last["low"] < sl and last["close"] > sl and last["lower_wick"] >= 0.4 * rng:
                lv, lw = 1.0, "liquidity sweep below swing low"
        votes["liquidity"], why["liquidity"] = lv, lw

        sv2, sw = 0.0, ""
        ps, pl = prev["sh1"], prev["sl1"]
        if not isnan(ps) and last["close"] > ps and prev["close"] <= ps:
            sv2, sw = 1.0, "CHoCH up" if (not isnan(prev["trend"]) and prev["trend"] < 0) else "BOS up"
        elif not isnan(pl) and last["close"] < pl and prev["close"] >= pl:
            sv2, sw = -1.0, "CHoCH down" if (not isnan(prev["trend"]) and prev["trend"] > 0) else "BOS down"
        votes["structure"], why["structure"] = sv2, sw

        votes["price_action"], why["price_action"] = pattern_vote(prev, last)

        total = sum(WEIGHTS[n] * float(v) for n, v in votes.items())
        if total == 0:
            return None
        s = 1 if total > 0 else -1
        agree = sum(1 for v in votes.values() if sign(v) == s)
        if abs(total) < cfg_get(context, "ai_min_total", 40) or agree < cfg_get(context, "ai_min_agree", 4):
            return None
        score = clamp(abs(total) * 1.25, 0, 99)
        top = sorted(((abs(WEIGHTS[n] * v), n) for n, v in votes.items() if sign(v) == s), reverse=True)[:4]
        reasons = [why[n] for _, n in top if why.get(n)]
        reasons.append(f"{agree}/10 components agree")
        return build_candidate(df, context, self.name, CALL if s > 0 else PUT, score, reasons)
