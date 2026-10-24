#!/usr/bin/env python3
"""STM PRO BOT - signal-only Telegram bot (no trading, no account funds).

Single-file core: config, logging, localisation, indicators, data providers
(mock + PyQuotex adapter), scanner, signal engine, result engine, scheduler,
database and Telegram UI. Strategy rules live in their own modules
(ai_strategy.py, stm_strategy.py, random_strategy.py, black4x.py, opx4o.py).

Run:  python main.py             (bot)
      python main.py --check     (validate configuration and exit)
      python main.py --backtest  (chronological train/test backtest on mock data)
      python main.py --diagnose-provider   (inspect the PyQuotex install, no secrets printed)

This software gives no guarantee of accuracy or profit. Trading carries risk.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_strategy import AiStrategy
from black4x import Black4xStrategy
from opx4o import Opx4oStrategy
from random_strategy import RandomStrategy
from stm_strategy import StmPvip2Strategy, StmPvipStrategy
from strategy_base import CALL, PUT, Candidate, Strategy

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

log = logging.getLogger("stm")

# =============================================================== CONFIG ====
def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    return default if v == "" else v.lower() in ("1", "true", "yes", "on")


def _env_num(name: str, default: float, cast=float):
    v = _env(name, "")
    if v == "":
        return default
    try:
        return cast(v)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {v!r}")


class ConfigError(Exception):
    pass


@dataclass
class Config:
    bot_token: str = ""
    channel_id: str = ""
    channel_username: str = ""
    channel_invite_url: str = ""
    channel_name: str = "STM PRO Channel"
    admin_ids: tuple = ()
    admin_username: str = ""
    feedback_contact: str = ""
    referral_url: str = ""
    min_payout: float = 80.0
    timezone: str = "Asia/Dhaka"
    scan_interval_seconds: int = 60
    database_path: str = "stm_pro_bot.db"
    provider: str = "mock"                 # mock | quotex
    quotex_email: str = ""
    quotex_password: str = ""
    quotex_lang: str = "en"
    language: str = "both"                 # en | bn | both
    lead_seconds: float = 5.0
    candle_mode: str = "closed"            # closed | provisional
    candles_count: int = 150
    concurrency: int = 5
    rate_limit_per_sec: float = 8.0
    retries: int = 3
    cache_ttl: float = 20.0
    max_stale_candles: int = 0
    min_confidence: float = 50.0
    max_signals_per_cycle: int = 1
    enable_random: bool = False
    enabled_strategies: tuple = ("AI", "STM_PVIP", "STM_PVIP2", "BLACK4X", "OPX4O")
    mtg_enabled: bool = True
    loss_streak_limit: int = 2
    cooldown_hours: float = 6.0
    require_membership: bool = True
    membership_fail_open: bool = False
    show_interstitial: bool = True
    result_settle_delay: float = 3.0
    result_retry_minutes: int = 60
    summary_time: str = "23:55"
    admin_stop_command: str = "STMOP"
    pvip_run: int = 3
    pvip2_body_min: float = 0.55
    pvip2_body_max: float = 0.85
    pvip2_cap_enabled: bool = True
    pvip2_boundary_atr: float = 0.1
    doji_threshold: float = 0.10
    opx4o_min_range_atr: float = 0.3
    vol_min_ratio: float = 0.35
    vol_max_ratio: float = 2.8
    ai_min_total: float = 40.0
    ai_min_agree: int = 4
    swing_left: int = 3
    swing_right: int = 3
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        ids = [x for x in re.split(r"[,\s]+", _env("ADMIN_IDS")) if x]
        bad = [x for x in ids if not re.fullmatch(r"\d{3,15}", x)]
        if bad:
            raise ConfigError("ADMIN_IDS must be comma separated numeric Telegram user IDs")
        strategies = [s.strip().upper() for s in _env(
            "ENABLED_STRATEGIES", "AI,STM_PVIP,STM_PVIP2,BLACK4X,OPX4O").split(",") if s.strip()]
        return cls(
            bot_token=_env("BOT_TOKEN"), channel_id=_env("CHANNEL_ID"),
            channel_username=_env("CHANNEL_USERNAME").lstrip("@"),
            channel_invite_url=_env("CHANNEL_INVITE_URL"),
            channel_name=_env("CHANNEL_NAME", "STM PRO Channel"),
            admin_ids=tuple(int(x) for x in ids), admin_username=_env("ADMIN_USERNAME").lstrip("@"),
            feedback_contact=_env("FEEDBACK_CONTACT"), referral_url=_env("REFERRAL_URL"),
            min_payout=_env_num("MIN_PAYOUT", 80.0), timezone=_env("TIMEZONE", "Asia/Dhaka"),
            scan_interval_seconds=_env_num("SCAN_INTERVAL_SECONDS", 60, int),
            database_path=_env("DATABASE_PATH", "stm_pro_bot.db"),
            provider=_env("PROVIDER", "mock").lower(), quotex_email=_env("QUOTEX_EMAIL"),
            quotex_password=_env("QUOTEX_PASSWORD"), quotex_lang=_env("QUOTEX_LANG", "en"),
            language=_env("LANGUAGE", "both").lower(), lead_seconds=_env_num("SIGNAL_LEAD_SECONDS", 5.0),
            candle_mode=_env("CANDLE_MODE", "closed").lower(),
            concurrency=_env_num("CONCURRENCY", 5, int),
            rate_limit_per_sec=_env_num("RATE_LIMIT_PER_SEC", 8.0),
            min_confidence=_env_num("MIN_CONFIDENCE", 50.0),
            max_signals_per_cycle=_env_num("MAX_SIGNALS_PER_CYCLE", 1, int),
            enable_random=_env_bool("ENABLE_RANDOM", False), enabled_strategies=tuple(strategies),
            mtg_enabled=_env_bool("MTG_ENABLED", True),
            loss_streak_limit=_env_num("LOSS_STREAK_LIMIT", 2, int),
            cooldown_hours=_env_num("COOLDOWN_HOURS", 6.0),
            require_membership=_env_bool("REQUIRE_MEMBERSHIP", True),
            membership_fail_open=_env_bool("MEMBERSHIP_FAIL_OPEN", False),
            show_interstitial=_env_bool("SHOW_INTERSTITIAL", True),
            summary_time=_env("SUMMARY_TIME", "23:55"),
            admin_stop_command=_env("ADMIN_STOP_COMMAND", "STMOP").lstrip("/"),
            pvip_run=_env_num("PVIP_RUN", 3, int), pvip2_body_min=_env_num("PVIP2_BODY_MIN", 0.55),
            pvip2_body_max=_env_num("PVIP2_BODY_MAX", 0.85),
            pvip2_cap_enabled=_env_bool("PVIP2_CAP_ENABLED", True),
            doji_threshold=_env_num("DOJI_THRESHOLD", 0.10),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self, need_telegram: bool = True) -> list:
        errs = []
        if need_telegram and not re.fullmatch(r"\d{6,12}:[A-Za-z0-9_-]{30,}", self.bot_token):
            errs.append("BOT_TOKEN missing or malformed")
        if need_telegram and not self.admin_ids:
            errs.append("ADMIN_IDS must contain at least one numeric Telegram user ID")
        if self.require_membership and need_telegram and not (self.channel_id or self.channel_username):
            errs.append("CHANNEL_ID (or CHANNEL_USERNAME) is required when REQUIRE_MEMBERSHIP=true")
        try:
            ZoneInfo(self.timezone)
        except Exception:
            errs.append(f"TIMEZONE {self.timezone!r} is not a valid IANA zone")
        if self.provider not in ("mock", "quotex"):
            errs.append("PROVIDER must be mock or quotex")
        if self.candle_mode not in ("closed", "provisional"):
            errs.append("CANDLE_MODE must be closed or provisional")
        if not (0 < self.min_payout <= 100):
            errs.append("MIN_PAYOUT must be between 0 and 100")
        if self.scan_interval_seconds < 60 or self.scan_interval_seconds % 60:
            errs.append("SCAN_INTERVAL_SECONDS must be a multiple of 60 (signals use 1-minute expiry)")
        if not (3 <= self.pvip_run <= 10):
            errs.append("PVIP_RUN must be between 3 and 10")
        if self.concurrency < 1:
            errs.append("CONCURRENCY must be >= 1")
        if self.language not in ("en", "bn", "both"):
            errs.append("LANGUAGE must be en, bn or both")
        if not re.fullmatch(r"\d{1,2}:\d{2}", self.summary_time):
            errs.append("SUMMARY_TIME must look like 23:55")
        return errs

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def secrets(self) -> list:
        return [s for s in (self.bot_token, self.quotex_password, self.quotex_email) if s]


class SecretFilter(logging.Filter):
    """Replaces known secrets in log lines so they can never be printed."""
    def __init__(self, secrets_fn: Callable[[], list]):
        super().__init__()
        self.secrets_fn = secrets_fn

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            for s in self.secrets_fn():
                if s and s in msg:
                    msg = msg.replace(s, "***")
            record.msg, record.args = msg, ()
        except Exception:
            pass
        return True


def setup_logging(cfg: Config) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    h.addFilter(SecretFilter(lambda: cfg.secrets))
    root.handlers[:] = [h]
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs URLs containing the bot token


# =========================================================== LOCALISATION ==
STRINGS = {
    "welcome": {
        "en": "Welcome to STM PRO BOT!\nChannel: {channel}\nAdmin: {admin}\nThis is a signal-only bot. It never trades for you.",
        "bn": "STM PRO BOT-এ স্বাগতম!\nচ্যানেল: {channel}\nঅ্যাডমিন: {admin}\nএটি শুধু সিগন্যাল বট। আপনার হয়ে ট্রেড করে না।"},
    "join_first": {
        "en": "Please join our channel first, then tap Verify.\nNote: this only checks channel membership. It cannot prove you registered with any referral link.",
        "bn": "প্রথমে চ্যানেলে জয়েন করুন, তারপর Verify চাপুন।\nনোট: এটি শুধু চ্যানেল মেম্বারশিপ চেক করে। রেফারেল লিংকে রেজিস্ট্রেশন হয়েছে কিনা প্রমাণ করতে পারে না।"},
    "not_member": {"en": "Not a member yet. Join the channel, then tap Verify again.",
                   "bn": "এখনও মেম্বার নন। চ্যানেলে জয়েন করে আবার Verify চাপুন।"},
    "cannot_verify": {"en": "Membership check is unavailable right now. Please try again later.",
                      "bn": "এই মুহূর্তে মেম্বারশিপ চেক করা যাচ্ছে না। পরে আবার চেষ্টা করুন।"},
    "risk1": {"en": "Risk notice 1/3: trading can lose money.", "bn": "ঝুঁকি নোটিস ১/৩: ট্রেডিংয়ে টাকা হারাতে পারেন।"},
    "risk2": {"en": "Risk notice 2/3: no signal is guaranteed. Past results do not predict future ones.",
              "bn": "ঝুঁকি নোটিস ২/৩: কোনো সিগন্যালের নিশ্চয়তা নেই।"},
    "risk3": {"en": "Risk notice 3/3: only use money you can afford to lose. MTG never guarantees recovery.",
              "bn": "ঝুঁকি নোটিস ৩/৩: যে টাকা হারানো সহ্য করতে পারেন শুধু তাই ব্যবহার করুন। MTG লস রিকভারির নিশ্চয়তা দেয় না।"},
    "verified": {"en": "Verified. Tap Get Signal.", "bn": "ভেরিফাই সম্পন্ন। Get Signal চাপুন।"},
    "pick_cats": {"en": "Choose market categories:", "bn": "মার্কেট ক্যাটাগরি বাছাই করুন:"},
    "pick_pairs": {"en": "Choose pairs (page {p}/{n}):", "bn": "পেয়ার বাছাই করুন (পেজ {p}/{n}):"},
    "pick_strats": {"en": "Choose strategies:", "bn": "স্ট্র্যাটেজি বাছাই করুন:"},
    "started": {"en": "Signals ON for {n} pair(s). Signals arrive ~{lead}s before each entry minute.",
                "bn": "{n}টি পেয়ারের জন্য সিগন্যাল চালু। প্রতিটি এন্ট্রির প্রায় {lead} সেকেন্ড আগে আসবে।"},
    "stopped_user": {"en": "Your signals are stopped.", "bn": "আপনার সিগন্যাল বন্ধ করা হয়েছে।"},
    "engine_stopped": {"en": "Signals stopped by admin. Open trades will still be reported.",
                       "bn": "অ্যাডমিন সিগন্যাল বন্ধ করেছেন। চলমান ট্রেডের রেজাল্ট আসবে।"},
    "need_verify": {"en": "Please /start and verify first.", "bn": "আগে /start দিয়ে ভেরিফাই করুন।"},
    "disclaimer": {"en": "Risk is possible; no accuracy or profit is guaranteed.",
                   "bn": "ঝুঁকি আছে; কোনো একুরেসি বা লাভের নিশ্চয়তা নেই।"},
    "latency": {"en": "Network latency and broker execution can make your real entry differ from the scheduled time.",
                "bn": "নেটওয়ার্ক ও ব্রোকারের কারণে প্রকৃত এন্ট্রি সময় আলাদা হতে পারে।"},
}


def tr(key: str, cfg: Config, **kw) -> str:
    d = STRINGS[key]
    parts = [d["en"]] if cfg.language == "en" else [d["bn"]] if cfg.language == "bn" else [d["en"], d["bn"]]
    return "\n".join(p.format(**kw) for p in parts)


# ================================================================== TIME ===
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def next_entry(now: datetime, lead: float, interval_seconds: int = 60) -> datetime:
    """Next entry-minute boundary E (a multiple of `interval_seconds`) with E - lead > now."""
    e = floor_minute(now + timedelta(seconds=lead)) + timedelta(seconds=60)
    step = max(60, interval_seconds)
    while int(e.timestamp()) % step:
        e += timedelta(seconds=60)
    return e


def local_str(ts: float, cfg: Config, with_date: bool = False) -> str:
    dt = datetime.fromtimestamp(ts, timezone.utc).astimezone(cfg.tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S %Z" if with_date else "%H:%M:%S %Z")


# ============================================================ INDICATORS ===
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(dn == 0) & up.notna()] = 100.0
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
    ll, hh = df["low"].rolling(k).min(), df["high"].rolling(k).max()
    raw = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    kk = raw.rolling(smooth).mean()
    return kk, kk.rolling(d).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    signal = line.ewm(span=sig, adjust=False, min_periods=sig).mean()
    return line, signal, line - signal


def swing_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Confirmed swing highs/lows and the trend they imply - strictly causal.

    A swing high at bar i needs high[i] to be the unique maximum of bars
    [i-left, i+right]. It becomes KNOWN at bar i+right (when its last confirming
    bar closes), so row j only ever uses swings with i+right <= j.
    Columns: sh1/sh2 = latest/previous confirmed swing high, sl1/sl2 likewise for lows,
    trend = +1 (higher highs AND higher lows), -1 (lower both), 0 otherwise.
    """
    n = len(df)
    h, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    highs, lows = [], []  # (confirm_index, price)
    for i in range(left, n - right):
        wh = h[i - left:i + right + 1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1:
            highs.append((i + right, h[i]))
        wl = lo[i - left:i + right + 1]
        if lo[i] == wl.min() and (wl == lo[i]).sum() == 1:
            lows.append((i + right, lo[i]))
    sh1 = np.full(n, np.nan); sh2 = sh1.copy(); sl1 = sh1.copy(); sl2 = sh1.copy()
    trend = np.zeros(n)
    ph = pl = 0
    for j in range(n):
        while ph < len(highs) and highs[ph][0] <= j:
            ph += 1
        while pl < len(lows) and lows[pl][0] <= j:
            pl += 1
        if ph >= 1:
            sh1[j] = highs[ph - 1][1]
        if ph >= 2:
            sh2[j] = highs[ph - 2][1]
        if pl >= 1:
            sl1[j] = lows[pl - 1][1]
        if pl >= 2:
            sl2[j] = lows[pl - 2][1]
        if ph >= 2 and pl >= 2:
            if sh1[j] > sh2[j] and sl1[j] > sl2[j]:
                trend[j] = 1
            elif sh1[j] < sh2[j] and sl1[j] < sl2[j]:
                trend[j] = -1
    return pd.DataFrame({"sh1": sh1, "sh2": sh2, "sl1": sl1, "sl2": sl2, "trend": trend}, index=df.index)


def prepare(raw: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Add every indicator / candle feature column. Causal: row j depends only on rows <= j."""
    df = raw.reset_index(drop=True).copy()
    c = df["close"]
    for n in (10, 20, 50):
        df[f"ema{n}"] = ema(c, n)
    df["sma20"], df["sma50"] = sma(c, 20), sma(c, 50)
    df["rsi"], df["atr"] = rsi(c), atr(df)
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(50, min_periods=20).mean()
    df["stoch_k"], df["stoch_d"] = stochastic(df)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(c)
    df["body"] = (df["close"] - df["open"]).abs()
    df["rng"] = df["high"] - df["low"]
    # body_ratio = body / (high - low)  (documented definition of "body %")
    df["body_ratio"] = np.where(df["rng"] > 0, df["body"] / df["rng"].where(df["rng"] > 0, 1), 0.0)
    df["color"] = np.sign(df["close"] - df["open"]).astype(int)
    thr = getattr(cfg, "doji_threshold", 0.10)
    df["is_doji"] = (df["rng"] <= 0) | (df["body_ratio"] <= thr)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    sw = swing_structure(df, getattr(cfg, "swing_left", 3), getattr(cfg, "swing_right", 3))
    return pd.concat([df, sw], axis=1)


def validate_candles(df: pd.DataFrame, expected_last_open: pd.Timestamp, min_rows: int = 80,
                     contiguous_tail: int = 60, max_stale: int = 0):
    """OHLC sanity, timezone, gaps and staleness. Returns (ok, reason)."""
    if df is None or len(df) < min_rows:
        return False, f"not enough candles ({0 if df is None else len(df)} < {min_rows})"
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        return False, "missing OHLC columns"
    if df["time"].dt.tz is None:
        return False, "timestamps are not timezone-aware"
    vals = df[["open", "high", "low", "close"]]
    if vals.isna().any().any() or not np.isfinite(vals.to_numpy(float)).all():
        return False, "NaN/inf in OHLC"
    if (vals <= 0).any().any():
        return False, "non-positive price"
    if (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12).any() or \
            (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12).any():
        return False, "inconsistent OHLC (high/low do not bound open/close)"
    if not df["time"].is_monotonic_increasing or df["time"].duplicated().any():
        return False, "timestamps not strictly increasing"
    tail = df["time"].iloc[-contiguous_tail:].diff().dropna()
    if (tail != pd.Timedelta(seconds=60)).any():
        return False, "gap in the most recent candles"
    lag = (expected_last_open - df["time"].iloc[-1]).total_seconds() / 60
    if lag > max_stale or lag < 0:
        return False, f"stale/future data (last candle {df['time'].iloc[-1]}, expected {expected_last_open})"
    return True, "ok"


# =========================================================== PROVIDERS =====
class ProviderError(Exception):
    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal


@dataclass
class Asset:
    pair: str          # id used to fetch candles
    display: str
    category: str      # FOREX | CURRENCY | CRYPTO | COMMODITIES
    is_otc: bool
    payout: Optional[float] = None


CATEGORIES = ["FOREX", "CURRENCY", "CRYPTO", "COMMODITIES"]
_CRYPTO = ("BTC", "ETH", "LTC", "XRP", "BNB", "SOL", "ADA", "DOGE", "DOT", "AVAX", "MATIC", "TRX", "LINK")
_COMMOD = ("XAU", "XAG", "GOLD", "SILVER", "BRENT", "CRUDE", "OIL", "WTI", "XPT", "XPD", "NATGAS", "COPPER")


def classify_asset(name: str) -> tuple:
    """Heuristic (documented): OTC currency pairs -> CURRENCY, live FX -> FOREX,
    coins -> CRYPTO, metals/energy -> COMMODITIES. Returns (category, is_otc)."""
    up = name.upper().replace("/", "").replace(" ", "")
    otc = "OTC" in up
    base = up.replace("_OTC", "").replace("(OTC)", "").replace("OTC", "")
    if any(base.startswith(x) or x in base[:6] for x in _CRYPTO):
        return "CRYPTO", otc
    if any(x in base for x in _COMMOD):
        return "COMMODITIES", otc
    return ("CURRENCY" if otc else "FOREX"), otc


class DataProvider(ABC):
    name = "base"

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def reconnect(self) -> None:
        await self.close()
        await self.connect()

    @abstractmethod
    async def list_assets(self) -> list: ...
    @abstractmethod
    async def get_candles(self, pair: str, count: int, as_of: datetime, include_forming: bool) -> pd.DataFrame: ...
    @abstractmethod
    async def get_payout(self, pair: str) -> Optional[float]: ...

    async def get_candle_at(self, pair: str, open_ts: datetime, now: datetime) -> Optional[pd.Series]:
        """The CLOSED 1-minute candle whose open time == open_ts, else None."""
        if now < open_ts + timedelta(seconds=60):
            return None
        n = int((now - open_ts).total_seconds() // 60) + 3
        df = await self.get_candles(pair, min(n, 190), now, False)
        hit = df[df["time"] == pd.Timestamp(open_ts)]
        return None if hit.empty else hit.iloc[0]


class MockProvider(DataProvider):
    """Deterministic synthetic market (continuous, consistent between calls).
    For software testing only - prices are NOT real."""
    name = "mock"
    PAIRS = {
        "EURUSD": 1.0850, "GBPUSD": 1.2700, "USDJPY": 150.20, "AUDUSD": 0.6600, "USDCAD": 1.3600,
        "EURUSD_otc": 1.0850, "GBPUSD_otc": 1.2700, "AUDCAD_otc": 0.9000,
        "BTCUSD": 65000.0, "ETHUSD": 3200.0, "XAUUSD": 2350.0, "UKBRENT": 85.0,
    }

    def _seed(self, pair: str) -> int:
        return int(hashlib.sha256(pair.encode()).hexdigest()[:8], 16)

    def _price(self, pair: str, m: int) -> float:
        base, s = self.PAIRS[pair], self._seed(pair)
        vol = 0.0004
        ph1, ph2 = (s % 628) / 100, ((s >> 8) % 628) / 100
        trend = math.sin(m / 45 + ph1) * 8 + math.sin(m / 11 + ph2) * 4
        noise = random.Random(s * 1000003 + m).gauss(0, 1)
        return base * (1 + vol * (trend + noise))

    def _candle(self, pair: str, m: int) -> dict:
        o, c = self._price(pair, m), self._price(pair, m + 1)
        r = random.Random(self._seed(pair) * 7919 + m)
        base = self.PAIRS[pair] * 0.0002
        hi = max(o, c) + abs(r.gauss(0, 1)) * base
        lo = min(o, c) - abs(r.gauss(0, 1)) * base
        return {"time": pd.Timestamp(m * 60, unit="s", tz="UTC"), "open": o, "high": hi, "low": lo, "close": c}

    async def list_assets(self) -> list:
        out = []
        for p in self.PAIRS:
            cat, otc = classify_asset(p)
            out.append(Asset(p, p.replace("_otc", " (OTC)"), cat, otc, await self.get_payout(p)))
        return out

    async def get_candles(self, pair, count, as_of, include_forming):
        if pair not in self.PAIRS:
            raise ProviderError(f"unknown mock pair {pair}", fatal=True)
        last = int(as_of.timestamp() // 60) - (0 if include_forming else 1)
        return pd.DataFrame([self._candle(pair, m) for m in range(last - count + 1, last + 1)])

    async def get_payout(self, pair):
        if pair not in self.PAIRS:
            return None
        hour = int(time.time() // 3600)
        return float(70 + (self._seed(pair) + hour) % 23)


class QuotexProvider(DataProvider):
    """Adapter for the UNOFFICIAL pyquotex package (pip install from GitHub).

    Verified from its public README only: `from pyquotex.stable_api import Quotex`,
    `await client.connect()` and `client.get_candles(...)` exist. Payout/asset helper
    names vary between versions, so they are discovered at runtime with hasattr and
    the adapter FAILS SAFE (no payout -> no signal). It has NOT been tested live.
    Run `python main.py --diagnose-provider` to see what your install offers.
    """
    name = "quotex"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = None
        self._ids: dict = {}

    @staticmethod
    def _unwrap(res):
        return res[0] if isinstance(res, (tuple, list)) and res else bool(res)

    async def connect(self):
        try:
            from pyquotex.stable_api import Quotex
        except Exception as e:
            raise ProviderError(f"pyquotex is not installed or incompatible: {e}", fatal=True)
        email, pw = self.cfg.quotex_email, self.cfg.quotex_password
        if (not email or not pw) and sys.stdin.isatty():
            email = email or input("Quotex email: ").strip()
            pw = pw or getpass.getpass("Quotex password (hidden): ")
            self.cfg.quotex_email, self.cfg.quotex_password = email, pw
        if not email or not pw:
            raise ProviderError("QUOTEX_EMAIL / QUOTEX_PASSWORD not set (use deployment secrets)", fatal=True)
        try:
            self.client = Quotex(email=email, password=pw, lang=self.cfg.quotex_lang)
        except TypeError as e:
            raise ProviderError(f"unexpected Quotex() signature: {e}", fatal=True)
        for need in ("connect", "get_candles"):
            if not hasattr(self.client, need):
                raise ProviderError(f"pyquotex missing method {need}() - incompatible version", fatal=True)
        ok = self._unwrap(await self.client.connect())
        if not ok:
            raise ProviderError("pyquotex connect() reported failure")
        log.info("Quotex connected")

    async def close(self):
        try:
            if self.client is not None and hasattr(self.client, "close"):
                res = self.client.close()
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            log.warning("Quotex close error: %s", e)
        self.client = None

    def _need(self):
        if self.client is None:
            raise ProviderError("provider not connected")
        return self.client

    async def _payment_map(self) -> dict:
        cl = self._need()
        fn = getattr(cl, "get_payment", None) or getattr(cl, "get_payment_data", None)
        if fn is None:
            raise ProviderError("pyquotex has no payout helper (get_payment) - cannot verify payouts", fatal=True)
        res = fn()
        if asyncio.iscoroutine(res):
            res = await res
        if not isinstance(res, dict):
            raise ProviderError("unexpected payout structure from pyquotex")
        return res

    @staticmethod
    def _payout_from(info) -> Optional[float]:
        if not isinstance(info, dict):
            return None
        cands = []
        prof = info.get("profit")
        if isinstance(prof, dict):
            cands += [prof.get("1M"), prof.get("1m")]
        cands += [info.get("turbo_payment"), info.get("payment"), info.get("payout")]
        for v in cands:
            try:
                f = float(v)
                if 0 < f <= 100:
                    return f
            except (TypeError, ValueError):
                continue
        return None

    async def list_assets(self):
        out = []
        for name, info in (await self._payment_map()).items():
            if isinstance(info, dict) and info.get("open") is False:
                continue
            pay = self._payout_from(info)
            cat, otc = classify_asset(str(name))
            pid = str(name).replace("/", "").replace(" (OTC)", "_otc").replace(" ", "")
            self._ids[pid] = str(name)
            out.append(Asset(pid, str(name), cat, otc, pay))
        return out

    async def get_payout(self, pair):
        info = (await self._payment_map()).get(self._ids.get(pair, pair))
        return self._payout_from(info)

    async def get_candles(self, pair, count, as_of, include_forming):
        cl = self._need()
        asset = pair
        if hasattr(cl, "get_available_asset"):
            try:
                res = await cl.get_available_asset(pair, force_open=False)
                if isinstance(res, (tuple, list)) and res and res[0]:
                    asset = res[0]
            except Exception as e:
                log.debug("get_available_asset failed for %s: %s", pair, e)
        raw = await cl.get_candles(asset, as_of.timestamp(), count * 60, 60)
        if not raw:
            raise ProviderError(f"no candles returned for {pair}")
        try:
            df = pd.DataFrame(raw)[["time", "open", "high", "low", "close"]].copy()
        except KeyError:
            raise ProviderError("candle format incompatible (need time/open/high/low/close)", fatal=True)
        df["time"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        if not include_forming:
            df = df[df["time"] + pd.Timedelta(seconds=60) <= pd.Timestamp(as_of)]
        return df.tail(count).reset_index(drop=True)


def make_provider(cfg: Config) -> DataProvider:
    return MockProvider() if cfg.provider == "mock" else QuotexProvider(cfg)


async def diagnose_provider(cfg: Config) -> None:
    print("pyquotex diagnostics (no secrets printed)")
    try:
        import pyquotex.stable_api as sa
    except Exception as e:
        print(f"  pyquotex import FAILED: {e}")
        return
    names = [n for n in dir(sa.Quotex) if not n.startswith("_")]
    print("  public methods:", ", ".join(names))
    for need in ("connect", "get_candles", "get_payment", "get_available_asset", "close"):
        print(f"  {need}: {'yes' if need in names else 'MISSING'}")


# ============================================================== SCANNER ====
class RateLimiter:
    def __init__(self, per_sec: float):
        self.interval = 1.0 / max(per_sec, 0.1)
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


async def with_retries(fn: Callable[[], Awaitable], retries: int, label: str,
                       on_fail: Optional[Callable[[], Awaitable]] = None, base: float = 0.5):
    """Exponential backoff with jitter; fatal ProviderErrors are not retried."""
    last = None
    for attempt in range(retries + 1):
        try:
            return await fn()
        except ProviderError as e:
            last = e
            if e.fatal:
                raise
        except (asyncio.TimeoutError, OSError, ConnectionError) as e:
            last = e
        if attempt < retries:
            delay = base * (2 ** attempt) + random.random() * 0.2
            log.warning("%s failed (%s); retry %d/%d in %.1fs", label, last, attempt + 1, retries, delay)
            await asyncio.sleep(delay)
            if on_fail:
                try:
                    await on_fail()
                except Exception as e2:
                    log.warning("reconnect failed: %s", e2)
    raise ProviderError(f"{label} failed after {retries + 1} attempts: {last}")


class MarketScanner:
    def __init__(self, cfg: Config, provider: DataProvider):
        self.cfg, self.provider = cfg, provider
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.rl = RateLimiter(cfg.rate_limit_per_sec)
        self._cache: dict = {}

    async def _call(self, label, fn):
        async with self.sem:
            await self.rl.wait()
            return await with_retries(fn, self.cfg.retries, label, self.provider.reconnect)

    async def candles(self, pair: str, as_of: datetime, include_forming: bool) -> pd.DataFrame:
        key = (pair, int(as_of.timestamp() // 10), include_forming)
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < self.cfg.cache_ttl:
            return hit[1]
        df = await self._call(f"candles {pair}",
                              lambda: self.provider.get_candles(pair, self.cfg.candles_count, as_of, include_forming))
        if len(self._cache) > 500:
            self._cache.clear()
        self._cache[key] = (time.monotonic(), df)
        return df

    async def payout(self, pair: str) -> Optional[float]:
        return await self._call(f"payout {pair}", lambda: self.provider.get_payout(pair))

    async def candle_at(self, pair: str, open_ts: datetime, now: datetime):
        return await self._call(f"candle_at {pair}", lambda: self.provider.get_candle_at(pair, open_ts, now))

    async def assets(self) -> list:
        return await self._call("assets", self.provider.list_assets)


# ============================================================== DATABASE ===
SCHEMA = """
CREATE TABLE IF NOT EXISTS signals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, display TEXT, category TEXT, is_otc INTEGER,
  strategy TEXT, direction TEXT, payout REAL, confidence REAL, ref_price REAL,
  signal_ts REAL, entry_ts INTEGER, expiry_ts INTEGER, candle_ts INTEGER,
  status TEXT DEFAULT 'OPEN', open_price REAL, close_price REAL, first_result TEXT,
  mtg_used INTEGER DEFAULT 0, mtg_open REAL, mtg_close REAL, mtg_result TEXT, final_result TEXT,
  attempts INTEGER DEFAULT 0, pending_notified INTEGER DEFAULT 0, resolved_ts REAL,
  experimental INTEGER DEFAULT 0, reasons TEXT,
  UNIQUE(pair, entry_ts), UNIQUE(pair, strategy, candle_ts));
CREATE TABLE IF NOT EXISTS users(
  user_id INTEGER PRIMARY KEY, verified INTEGER DEFAULT 0, active INTEGER DEFAULT 0,
  cats TEXT DEFAULT '[]', pairs TEXT DEFAULT '[]', strategies TEXT DEFAULT '[]', page INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS deliveries(signal_id INTEGER, user_id INTEGER, PRIMARY KEY(signal_id, user_id));
CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS admin_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, admin_id INTEGER, action TEXT, detail TEXT);
"""
RETRYABLE = ("OPEN", "AWAIT_MTG", "UNAVAILABLE")


class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    def _x(self, sql: str, p: tuple = ()):
        with self._lock:
            cur = self.conn.execute(sql, p)
            self.conn.commit()
            return cur

    def _q(self, sql: str, p: tuple = ()) -> list:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, p).fetchall()]

    # -- key/value state -------------------------------------------------
    def kv_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        r = self._q("SELECT value FROM kv WHERE key=?", (key,))
        return r[0]["value"] if r else default

    def kv_set(self, key: str, value: Any) -> None:
        self._x("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))

    # -- users -----------------------------------------------------------
    def get_user(self, uid: int) -> dict:
        r = self._q("SELECT * FROM users WHERE user_id=?", (uid,))
        if not r:
            self._x("INSERT OR IGNORE INTO users(user_id) VALUES(?)", (uid,))
            r = self._q("SELECT * FROM users WHERE user_id=?", (uid,))
        u = r[0]
        for k in ("cats", "pairs", "strategies"):
            u[k] = json.loads(u[k] or "[]")
        return u

    def set_user(self, uid: int, **fields) -> None:
        self.get_user(uid)
        for k, v in fields.items():
            if k not in ("verified", "active", "cats", "pairs", "strategies", "page"):
                raise ValueError(k)
            self._x(f"UPDATE users SET {k}=? WHERE user_id=?",
                    (json.dumps(v) if isinstance(v, (list, tuple)) else v, uid))

    def active_users(self) -> list:
        return [self.get_user(r["user_id"]) for r in self._q("SELECT user_id FROM users WHERE active=1 AND verified=1")]

    # -- signals ---------------------------------------------------------
    def add_signal(self, s: dict) -> Optional[int]:
        cols = ("pair display category is_otc strategy direction payout confidence ref_price signal_ts "
                "entry_ts expiry_ts candle_ts experimental reasons").split()
        cur = self._x(f"INSERT OR IGNORE INTO signals({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                      tuple(s[c] for c in cols))
        return cur.lastrowid if cur.rowcount else None

    def get_signal(self, sid: int) -> Optional[dict]:
        r = self._q("SELECT * FROM signals WHERE id=?", (sid,))
        return r[0] if r else None

    def update_signal(self, sid: int, **f) -> None:
        if not f:
            return
        self._x(f"UPDATE signals SET {','.join(k + '=?' for k in f)} WHERE id=?", (*f.values(), sid))

    def pair_busy(self, pair: str, now_ts: Optional[float] = None, window_min: int = 60) -> bool:
        """True while a signal on this pair is unresolved (overlap prevention). UNAVAILABLE
        signals only block while they are still inside the retry window."""
        if self._q("SELECT 1 FROM signals WHERE pair=? AND status IN ('OPEN','AWAIT_MTG')", (pair,)):
            return True
        if now_ts is None:
            return False
        return bool(self._q("SELECT 1 FROM signals WHERE pair=? AND status='UNAVAILABLE' AND expiry_ts>?",
                            (pair, now_ts - window_min * 60)))

    def unresolved(self, now_ts: float, retry_minutes: int) -> list:
        return self._q("SELECT * FROM signals WHERE status IN ('OPEN','AWAIT_MTG') OR "
                       "(status='UNAVAILABLE' AND expiry_ts > ?) ORDER BY entry_ts",
                       (now_ts - retry_minutes * 60,))

    def add_delivery(self, sid: int, uid: int) -> None:
        self._x("INSERT OR IGNORE INTO deliveries VALUES(?,?)", (sid, uid))

    def deliveries(self, sid: int) -> list:
        return [r["user_id"] for r in self._q("SELECT user_id FROM deliveries WHERE signal_id=?", (sid,))]

    def signals_between(self, t0: float, t1: float, resolved_only: bool = False) -> list:
        q = "SELECT * FROM signals WHERE entry_ts>=? AND entry_ts<? "
        if resolved_only:
            q += "AND status='RESOLVED' "
        return self._q(q + "ORDER BY entry_ts", (t0, t1))

    def open_count(self) -> int:
        return len(self._q("SELECT 1 FROM signals WHERE status IN ('OPEN','AWAIT_MTG','UNAVAILABLE')"))

    # -- statistics ------------------------------------------------------
    def stats(self, t0: Optional[float] = None, t1: Optional[float] = None,
              pair: Optional[str] = None, strategy: Optional[str] = None) -> dict:
        q, p = "SELECT * FROM signals WHERE status='RESOLVED'", []
        if t0 is not None:
            q += " AND entry_ts>=?"; p.append(t0)
        if t1 is not None:
            q += " AND entry_ts<?"; p.append(t1)
        if pair:
            q += " AND pair=?"; p.append(pair)
        if strategy:
            q += " AND strategy=?"; p.append(strategy)
        return compute_stats(self._q(q, tuple(p)))

    # -- admin log -------------------------------------------------------
    def log_admin(self, admin_id: int, action: str, detail: str = "") -> None:
        self._x("INSERT INTO admin_log(ts,admin_id,action,detail) VALUES(?,?,?,?)",
                (time.time(), admin_id, action, detail))
        log.info("ADMIN %s %s %s", admin_id, action, detail)


def compute_stats(rows: list) -> dict:
    """Headline: WIN + MTG WIN count as wins, LOSS as loss, DRAW excluded.
    First-attempt and recovery performance are reported separately."""
    first_wins = sum(1 for r in rows if r["final_result"] == "WIN")
    mtg_wins = sum(1 for r in rows if r["final_result"] == "MTG WIN")
    losses = sum(1 for r in rows if r["final_result"] == "LOSS")
    draws = sum(1 for r in rows if r["final_result"] == "DRAW")
    first_losses = sum(1 for r in rows if r["first_result"] == "LOSS")
    wins = first_wins + mtg_wins
    total = wins + losses
    mtg_tried = sum(1 for r in rows if r["mtg_used"] and r["first_result"] == "LOSS")
    return {
        "wins": wins, "losses": losses, "draws": draws, "total": total,
        "first_wins": first_wins, "mtg_wins": mtg_wins, "first_losses": first_losses,
        "accuracy": round(100 * wins / total, 1) if total else None,
        "first_accuracy": round(100 * first_wins / (first_wins + first_losses), 1) if (first_wins + first_losses) else None,
        "recovery_rate": round(100 * mtg_wins / mtg_tried, 1) if mtg_tried else None,
    }


# ============================================================= CONTROL =====
class Control:
    """Persistent engine state: running/paused/stopped, loss streak, cooldown, auto-run."""
    def __init__(self, db: Database, cfg: Config):
        self.db, self.cfg = db, cfg

    @property
    def state(self) -> str:
        return self.db.kv_get("engine_state", "running")

    def set_state(self, s: str) -> None:
        assert s in ("running", "paused", "stopped")
        self.db.kv_set("engine_state", s)

    @property
    def loss_streak(self) -> int:
        return int(self.db.kv_get("loss_streak", "0"))

    @property
    def cooldown_until(self) -> float:
        return float(self.db.kv_get("cooldown_until", "0"))

    @property
    def auto(self) -> tuple:
        return int(self.db.kv_get("auto_sent", "0")), int(self.db.kv_get("auto_target", "0"))

    def in_cooldown(self, now_ts: float) -> bool:
        return self.cooldown_until > now_ts

    def check_cooldown_end(self, now_ts: float) -> bool:
        """True exactly once when a cooldown has just ended (streak is reset)."""
        cu = self.cooldown_until
        if cu and cu <= now_ts:
            self.db.kv_set("cooldown_until", 0)
            self.db.kv_set("loss_streak", 0)
            return True
        return False

    def can_signal(self, now_ts: float) -> tuple:
        if self.state != "running":
            return False, f"engine {self.state}"
        if self.in_cooldown(now_ts):
            return False, "cooldown after loss streak"
        return True, "ok"

    def register_result(self, final: str, now_ts: float) -> bool:
        """Update loss streak from a FINAL result. True if a cooldown just started."""
        if final in ("WIN", "MTG WIN"):
            self.db.kv_set("loss_streak", 0)
            return False
        if final == "LOSS":
            streak = self.loss_streak + 1
            self.db.kv_set("loss_streak", streak)
            if streak >= self.cfg.loss_streak_limit:
                self.db.kv_set("cooldown_until", now_ts + self.cfg.cooldown_hours * 3600)
                return True
        return False

    def clear_cooldown(self) -> None:
        self.db.kv_set("cooldown_until", 0)
        self.db.kv_set("loss_streak", 0)

    def auto_start(self, n: int) -> None:
        self.db.kv_set("auto_target", n)
        self.db.kv_set("auto_sent", 0)
        self.set_state("running")

    def auto_register(self) -> tuple:
        """Count one SENT signal. Returns (sent, target); target 0 means no auto-run."""
        sent, target = self.auto
        if target <= 0:
            return 0, 0
        sent += 1
        self.db.kv_set("auto_sent", sent)
        if sent >= target:
            self.db.kv_set("auto_target", 0)
            self.set_state("stopped")
        return sent, target

    def auto_remaining(self) -> Optional[int]:
        sent, target = self.auto
        return None if target <= 0 else max(0, target - sent)


# ======================================================== SIGNAL ENGINE ====
STRAT_MENU = [("AI", "AI STRATEGY", ["AI"]),
              ("STM", "STM STRATEGY", ["STM_PVIP", "STM_PVIP2"]),
              ("RANDOM", "RANDOM STRATEGY (EXPERIMENTAL)", ["RANDOM"]),
              ("BLACK4X", "BLACK4X", ["BLACK4X"]),
              ("OPX4O", "OPX4O", ["OPX4O"])]


def build_strategies(cfg: Config, seed: Optional[int] = None) -> dict:
    allm = {"AI": AiStrategy(), "STM_PVIP": StmPvipStrategy(), "STM_PVIP2": StmPvip2Strategy(),
            "BLACK4X": Black4xStrategy(), "OPX4O": Opx4oStrategy(), "RANDOM": RandomStrategy(seed)}
    out = {k: v for k, v in allm.items() if k in cfg.enabled_strategies and k != "RANDOM"}
    if cfg.enable_random:
        out["RANDOM"] = allm["RANDOM"]
    return out


def expand_strategies(keys: list, strategies: dict) -> set:
    names = set()
    for key, _label, members in STRAT_MENU:
        if key in keys:
            names.update(m for m in members if m in strategies)
    return names


def settle(direction: str, open_price: float, close_price: float) -> str:
    """Documented convention: entry = OPEN of the entry-minute candle, exit = its CLOSE.
    CALL wins if close > open, PUT wins if close < open, equal -> DRAW (refund, excluded
    from win/loss counts). The broker's own settlement (price at the exact second) can
    differ slightly; this is the provider-data convention, not an official settlement."""
    if close_price == open_price:
        return "DRAW"
    win = close_price > open_price if direction == CALL else close_price < open_price
    return "WIN" if win else "LOSS"


def classify_final(first: str, mtg: Optional[str]) -> str:
    """First attempt result + optional MTG 1 result -> final label.
    WIN -> WIN, DRAW -> DRAW, LOSS -> LOSS (no recovery) / MTG WIN / LOSS (MTG lost or drew)."""
    if first in ("WIN", "DRAW"):
        return first
    if mtg is None:
        return "LOSS"
    return "MTG WIN" if mtg == "WIN" else "LOSS"


class SignalEngine:
    def __init__(self, cfg: Config, db: Database, scanner: MarketScanner, strategies: dict):
        self.cfg, self.db, self.scanner, self.strategies = cfg, db, scanner, strategies

    def expected_last_open(self, entry: datetime) -> datetime:
        return entry - timedelta(seconds=60 if self.cfg.candle_mode == "provisional" else 120)

    def analyze(self, raw: pd.DataFrame, pair: str, names: set, entry: datetime) -> tuple:
        ok, why = validate_candles(raw, pd.Timestamp(self.expected_last_open(entry)),
                                   max_stale=self.cfg.max_stale_candles)
        if not ok:
            return [], why
        df = prepare(raw, self.cfg)
        ctx = {"pair": pair, "cfg": self.cfg}
        out = []
        for n in sorted(names):
            try:
                c = self.strategies[n].evaluate(df, ctx)
            except Exception:
                log.exception("strategy %s crashed on %s", n, pair)
                continue
            if c is not None and (c.experimental or c.score >= self.cfg.min_confidence):
                out.append(c)
        return out, "ok"

    async def _pair(self, pair: str, names: set, entry: datetime, now: datetime, asset: Asset):
        if self.db.pair_busy(pair, now.timestamp(), self.cfg.result_retry_minutes):
            return None
        payout = await self.scanner.payout(pair)           # live, verified payout
        if payout is None or payout < self.cfg.min_payout:
            return None
        raw = await self.scanner.candles(pair, now, self.cfg.candle_mode == "provisional")
        cands, why = await asyncio.to_thread(self.analyze, raw, pair, names, entry)
        if not cands:
            if why != "ok":
                log.debug("skip %s: %s", pair, why)
            return None
        best = max(cands, key=lambda c: c.score)
        return {"pair": pair, "display": asset.display, "category": asset.category,
                "is_otc": int(asset.is_otc), "strategy": best.strategy, "direction": best.direction,
                "payout": payout, "confidence": best.score, "ref_price": best.ref_price,
                "signal_ts": time.time(), "entry_ts": int(entry.timestamp()),
                "expiry_ts": int(entry.timestamp()) + 60, "candle_ts": int(best.candle_ts.timestamp()),
                "experimental": int(best.experimental), "reasons": json.dumps(best.reasons)}

    async def run_cycle(self, entry: datetime, now: datetime, selections: dict, assets: dict) -> list:
        jobs = [self._pair(p, names, entry, now, assets[p])
                for p, names in selections.items() if p in assets and names]
        res = await asyncio.gather(*jobs, return_exceptions=True)
        drafts = []
        for r in res:
            if isinstance(r, Exception):
                log.warning("pair scan failed: %s", r)
            elif r:
                drafts.append(r)
        drafts.sort(key=lambda d: d["confidence"], reverse=True)
        return drafts


# ======================================================== RESULT ENGINE ====
class ResultEngine:
    def __init__(self, cfg: Config, db: Database, scanner: MarketScanner):
        self.cfg, self.db, self.scanner = cfg, db, scanner

    async def _candle(self, pair: str, open_ts: int, now: datetime):
        try:
            return await self.scanner.candle_at(pair, datetime.fromtimestamp(open_ts, timezone.utc), now)
        except ProviderError as e:
            log.warning("result fetch %s: %s", pair, e)
            return None

    async def process(self, now: datetime) -> list:
        """Resolve what can be resolved. Returns [(kind, signal_id)], kind RESULT|PENDING."""
        events, ts = [], now.timestamp()
        for s in self.db.unresolved(ts, self.cfg.result_retry_minutes):
            sid = s["id"]
            stage2 = s["first_result"] is not None
            entry = s["entry_ts"] + (60 if stage2 else 0)
            if ts < entry + 60 + self.cfg.result_settle_delay:
                continue
            if s["status"] == "UNAVAILABLE" and ts - (s["resolved_ts"] or 0) < 60:
                continue                                   # retry at most once a minute
            candle = await self._candle(s["pair"], entry, now)
            if candle is None:
                upd = {"attempts": s["attempts"] + 1, "resolved_ts": ts}
                if ts > entry + 60 + 120:
                    if not s["pending_notified"]:
                        events.append(("PENDING", sid))
                    upd.update(status="UNAVAILABLE", pending_notified=1)
                self.db.update_signal(sid, **upd)
                continue
            o, c = float(candle["open"]), float(candle["close"])
            r = settle(s["direction"], o, c)
            if not stage2:
                if r == "LOSS" and self.cfg.mtg_enabled:
                    self.db.update_signal(sid, open_price=o, close_price=c, first_result="LOSS",
                                          mtg_used=1, status="AWAIT_MTG", pending_notified=0)
                    continue
                self.db.update_signal(sid, open_price=o, close_price=c, first_result=r,
                                      final_result=classify_final(r, None), status="RESOLVED", resolved_ts=ts)
            else:
                self.db.update_signal(sid, mtg_open=o, mtg_close=c, mtg_result=r,
                                      final_result=classify_final("LOSS", r), status="RESOLVED", resolved_ts=ts)
            events.append(("RESULT", sid))
        return events


# ============================================================ FORMATTING ===
def confidence_bar(score: float, width: int = 10) -> str:
    n = max(0, min(width, round(score / 100 * width)))
    return "█" * n + "░" * (width - n)


def fnum(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.6g}" if abs(x) < 1000 else f"{x:.2f}"


def format_signal(s: dict, cfg: Config) -> str:
    lines = ["🤖 STM PRO BOT", "━━━━━━━━━━━━━━━━━━━━━",
             f"💱 Pair: {s['display']}", f"⏱️ Trade at: {local_str(s['entry_ts'], cfg)}",
             f"🔮 Direction: {s['direction']}", f"💰 Payout: {s['payout']:.0f}%",
             "━━━━━━━━━━━━━━━━━━━━━", f"🎯 SIGNAL CONFIDENCE: {s['confidence']:.0f}%",
             confidence_bar(s["confidence"]), "━━━━━━━━━━━━━━━━━━━━━", "⏳ Expiry: 1 minute"]
    if s.get("experimental"):
        lines.append("🧪 EXPERIMENTAL random test signal - no edge, not a prediction.")
    lines.append("⚠️ Risk is possible; no accuracy or profit is guaranteed.")
    return "\n".join(lines)


def format_result(s: dict, cfg: Config, day: dict, pair_rec: dict) -> str:
    final = s["final_result"] if s["status"] == "RESOLVED" else "PENDING"
    lines = ["📊 STM PRO BOT RESULT", "━━━━━━━━━━━━━━━━━━━━━",
             f"💱 Pair: {s['display']}", f"⏱️ Entry time: {local_str(s['entry_ts'], cfg)}",
             f"🔮 Direction: {s['direction']}",
             f"📈 Opening price: {fnum(s['open_price'])}", f"📉 Closing price: {fnum(s['close_price'])}"]
    if s["mtg_used"] and s["mtg_open"] is not None:
        lines.append(f"🔁 MTG 1 candle: {fnum(s['mtg_open'])} → {fnum(s['mtg_close'])} ({s['mtg_result']})")
    elif s["mtg_used"]:
        lines.append("🔁 MTG 1: waiting for the recovery candle")
    lines.append(f"🏁 Result: {final}")
    if s["mtg_used"] and s["first_result"] == "LOSS":
        lines.append("↩️ First attempt: LOSS (kept in detailed stats)")
    acc = "—" if day["accuracy"] is None else f"{day['accuracy']}"
    lines += ["━━━━━━━━━━━━━━━━━━━━━", f"🔥 Wins: {day['wins']} | Losses: {day['losses']}",
              f"💎 Pair record: {pair_rec['wins']}W-{pair_rec['losses']}L",
              f"📊 Verified accuracy: {acc}% (today, MTG wins included)"]
    if cfg.feedback_contact:
        lines.append(f"📩 Feedback: {cfg.feedback_contact}")
    return "\n".join(lines)


def format_stats(st: dict, title: str) -> str:
    def p(v):
        return "—" if v is None else f"{v}%"
    return (f"📈 {title}\nResolved: {st['total']} (draws excluded: {st['draws']})\n"
            f"Headline wins (incl. MTG): {st['wins']} | Losses: {st['losses']} | Accuracy: {p(st['accuracy'])}\n"
            f"First-attempt wins: {st['first_wins']} | First-attempt accuracy: {p(st['first_accuracy'])}\n"
            f"MTG 1 wins: {st['mtg_wins']} | Recovery rate: {p(st['recovery_rate'])}\n"
            f"Original first-attempt losses: {st['first_losses']}")


def chunk_text(text: str, limit: int = 3800) -> list:
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    return out + ([cur.rstrip()] if cur.strip() else [])


def format_daily_summary(rows: list, cfg: Config, day_label: str) -> str:
    groups = [("OTC markets", [r for r in rows if r["is_otc"]]),
              ("Real markets", [r for r in rows if not r["is_otc"]])]
    out = [f"🗓️ STM PRO BOT - Daily summary {day_label}", "(chronological, actual results)"]
    for title, grp in groups:
        if not grp:
            continue                      # only show groups that really had signals
        st = compute_stats([r for r in grp if r["status"] == "RESOLVED"])
        out += ["", f"━━ {title} ━━"]
        for r in grp:
            res = r["final_result"] if r["status"] == "RESOLVED" else "PENDING/UNAVAILABLE"
            out.append(f"{local_str(r['entry_ts'], cfg).split(' ')[0]} {r['display']} {r['direction']} "
                       f"[{r['strategy']}] → {res}")
        acc = "—" if st["accuracy"] is None else f"{st['accuracy']}%"
        out.append(f"Wins {st['wins']} | Losses {st['losses']} | Win rate {acc} (MTG wins included; "
                   f"first-attempt {st['first_accuracy'] if st['first_accuracy'] is not None else '—'}%)")
    if len(out) == 2:
        out.append("No signals today.")
    out.append("\n⚠️ Win rate is computed only from persisted, resolved signals. No profit is guaranteed.")
    return "\n".join(out)


# ============================================================== NOTIFIERS ==
class Notifier:
    """Delivery interface (Telegram in production, a fake in tests)."""
    async def send(self, user_id: int, text: str, markup=None) -> None: ...
    async def send_many(self, user_ids: list, text: str) -> None:
        for u in user_ids:
            await self.send(u, text)
    async def notify_admins(self, text: str) -> None: ...
    async def post_channel(self, text: str) -> None: ...


class TelegramNotifier(Notifier):
    def __init__(self, bot, cfg: Config, db: Database):
        self.bot, self.cfg, self.db = bot, cfg, db
        self.sem = asyncio.Semaphore(15)

    async def send(self, user_id, text, markup=None):
        async with self.sem:
            for attempt in range(2):
                try:
                    await self.bot.send_message(chat_id=user_id, text=text, reply_markup=markup)
                    break
                except Exception as e:
                    name = type(e).__name__
                    if name == "RetryAfter" and attempt == 0:
                        await asyncio.sleep(float(getattr(e, "retry_after", 1)) + 0.5)
                        continue
                    if name == "Forbidden":
                        self.db.set_user(user_id, active=0)       # user blocked the bot
                    log.warning("send to %s failed: %s", user_id, name)
                    break
            await asyncio.sleep(0.04)

    async def notify_admins(self, text):
        for a in self.cfg.admin_ids:
            await self.send(a, text)

    async def post_channel(self, text):
        target = self.cfg.channel_id or f"@{self.cfg.channel_username}"
        for part in chunk_text(text):
            try:
                await self.bot.send_message(chat_id=target, text=part)
            except Exception as e:
                log.warning("channel post failed: %s", type(e).__name__)


# ================================================================ BOT APP ==
def day_bounds(cfg: Config, now: datetime) -> tuple:
    local = now.astimezone(cfg.tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return local.timestamp(), (local + timedelta(days=1)).timestamp()


class BotApp:
    def __init__(self, cfg: Config, db: Database, provider: DataProvider, notifier: Optional[Notifier] = None):
        self.cfg, self.db, self.provider = cfg, db, provider
        self.notifier = notifier or Notifier()
        self.scanner = MarketScanner(cfg, provider)
        self.strategies = build_strategies(cfg)
        self.engine = SignalEngine(cfg, db, self.scanner, self.strategies)
        self.results = ResultEngine(cfg, db, self.scanner)
        self.control = Control(db, cfg)
        self.assets: dict = {}
        self.assets_ts = 0.0
        self.last_cycle: dict = {}
        self.stop_event: Optional[asyncio.Event] = None
        self.tasks: list = []

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        self.stop_event = asyncio.Event()
        await self.provider.connect()
        await self.refresh_assets()
        for coro in (self._scheduler_loop, self._results_loop, self._summary_loop):
            self.tasks.append(asyncio.create_task(coro(), name=coro.__name__))
        log.info("STM PRO BOT started: provider=%s assets=%d mode=%s", self.provider.name,
                 len(self.assets), self.cfg.candle_mode)

    async def shutdown(self) -> None:
        if self.stop_event:
            self.stop_event.set()
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        try:
            await self.provider.close()
        finally:
            self.db.close()
        log.info("shutdown complete (open signals are preserved in the database)")

    async def refresh_assets(self) -> None:
        lst = await self.scanner.assets()
        self.assets = {a.pair: a for a in lst}
        self.assets_ts = time.time()
        if not self.assets:
            log.error("provider returned no assets - signals cannot be generated")

    # -- selections ------------------------------------------------------
    def user_names(self, user: dict) -> set:
        return expand_strategies(user["strategies"], self.strategies)

    def selections(self, users: list) -> dict:
        sel: dict = {}
        for u in users:
            names = self.user_names(u)
            for p in u["pairs"]:
                sel.setdefault(p, set()).update(names)
        return sel

    # -- one scan cycle --------------------------------------------------
    async def cycle(self, entry: datetime, now: Optional[datetime] = None) -> list:
        now = now or utcnow()
        if self.control.check_cooldown_end(now.timestamp()):
            await self.notifier.notify_admins("✅ Cooldown finished. Signals resume automatically.")
        ok, why = self.control.can_signal(now.timestamp())
        self.last_cycle = {"entry": entry.timestamp(), "ok": ok, "why": why, "sent": 0}
        if not ok:
            return []
        users = self.db.active_users()
        sel = self.selections(users)
        if not sel:
            return []
        timeout = max(1.0, self.cfg.lead_seconds - 0.8)
        try:
            drafts = await asyncio.wait_for(self.engine.run_cycle(entry, now, sel, self.assets), timeout)
        except asyncio.TimeoutError:
            log.warning("scan exceeded %.1fs - skipping this minute (never send late signals)", timeout)
            self.last_cycle["why"] = "scan timeout"
            return []
        limit = self.cfg.max_signals_per_cycle
        rem = self.control.auto_remaining()
        if rem is not None:
            limit = min(limit, rem)
        sent = await self.publish(drafts[:limit], entry, users)
        self.last_cycle["sent"] = len(sent)
        return sent

    async def publish(self, drafts: list, entry: datetime, users: list) -> list:
        out = []
        for d in drafts:
            if (entry - utcnow()).total_seconds() < 0.5:
                log.warning("too late to publish %s - dropped", d["pair"])
                continue
            d = dict(d, signal_ts=time.time())
            sid = self.db.add_signal(d)
            if sid is None:
                continue                                          # duplicate (pair/entry or pair/strategy/candle)
            row = self.db.get_signal(sid)
            text = format_signal(row, self.cfg)
            for u in users:
                strat_key_names = self.user_names(u)
                if d["pair"] in u["pairs"] and d["strategy"] in strat_key_names:
                    self.db.add_delivery(sid, u["user_id"])
                    await self.notifier.send(u["user_id"], text)
            out.append(row)
            sent, target = self.control.auto_register()
            if target:
                await self.notifier.notify_admins(f"🔁 Auto run: {sent}/{target} signals sent"
                                                  + (" - finished, signals stopped." if sent >= target else ""))
        return out

    # -- results ---------------------------------------------------------
    async def results_tick(self, now: Optional[datetime] = None) -> None:
        now = now or utcnow()
        events = await self.results.process(now)
        t0, t1 = day_bounds(self.cfg, now)
        for kind, sid in events:
            sig = self.db.get_signal(sid)
            day = self.db.stats(t0, t1)
            pr = self.db.stats(t0, t1, pair=sig["pair"])
            text = format_result(sig, self.cfg, day, pr)
            await self.notifier.send_many(self.db.deliveries(sid), text)
            if kind == "RESULT" and self.control.register_result(sig["final_result"], now.timestamp()):
                msg = (f"⏸️ {self.cfg.loss_streak_limit} losing signals in a row. Signals are paused for "
                       f"{self.cfg.cooldown_hours:g} hours (cooldown). Please take a break and review your risk.")
                await self.notifier.send_many([u["user_id"] for u in self.db.active_users()], msg)
                await self.notifier.notify_admins(msg)

    # -- loops -----------------------------------------------------------
    async def _sleep(self, seconds: float) -> bool:
        """Sleep; True if shutdown was requested."""
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=max(0.0, seconds))
            return True
        except asyncio.TimeoutError:
            return False

    async def _scheduler_loop(self):
        while True:
            entry = next_entry(utcnow(), self.cfg.lead_seconds, self.cfg.scan_interval_seconds)
            fire = entry - timedelta(seconds=self.cfg.lead_seconds)
            if await self._sleep((fire - utcnow()).total_seconds()):
                return
            try:
                await self.cycle(entry)
            except Exception:
                log.exception("scan cycle failed")

    async def _results_loop(self):
        while True:
            if await self._sleep(5):
                return
            try:
                if time.time() - self.assets_ts > 600:
                    await self.refresh_assets()
                await self.results_tick()
            except Exception:
                log.exception("results loop error")

    async def _summary_loop(self):
        while True:
            if await self._sleep(20):
                return
            try:
                local = utcnow().astimezone(self.cfg.tz)
                if local.strftime("%H:%M") == self.cfg.summary_time and \
                        self.db.kv_get("summary_date") != local.strftime("%Y-%m-%d"):
                    self.db.kv_set("summary_date", local.strftime("%Y-%m-%d"))
                    await self.post_summary(utcnow())
            except Exception:
                log.exception("summary error")

    async def post_summary(self, now: datetime) -> None:
        t0, t1 = day_bounds(self.cfg, now)
        rows = self.db.signals_between(t0, t1)
        await self.notifier.post_channel(
            format_daily_summary(rows, self.cfg, now.astimezone(self.cfg.tz).strftime("%Y-%m-%d")))


# ============================================================ TELEGRAM UI ==
def is_admin(cfg: Config, uid: int) -> bool:
    return uid in cfg.admin_ids          # numeric ID only; command names are never credentials


async def check_membership(bot, cfg: Config, uid: int) -> Optional[bool]:
    """True/False, or None when the check cannot be made (bot lacks rights, API error)."""
    if is_admin(cfg, uid):
        return True
    target = cfg.channel_id or (f"@{cfg.channel_username}" if cfg.channel_username else "")
    if not target:
        return None
    try:
        m = await bot.get_chat_member(chat_id=target, user_id=uid)
    except Exception as e:
        log.warning("getChatMember failed (%s) - is the bot an admin of the channel?", type(e).__name__)
        return None
    status = getattr(m.status, "value", m.status)
    if status in ("creator", "administrator", "member"):
        return True
    if status == "restricted":
        return bool(getattr(m, "is_member", False))
    return False


CB_RE = re.compile(r"^(verify|getsig|cats_next|pairs_next|go|stop|back_cats|back_pairs"
                   r"|cat:(FOREX|CURRENCY|CRYPTO|COMMODITIES|ALL)|pg:\d{1,3}|pr:(\d{1,3}|ALL)"
                   r"|st:(AI|STM|RANDOM|BLACK4X|OPX4O|ALL)|auto:(10|20|25|30|40))$")
PER_PAGE = 8


def toggle(lst: list, item) -> list:
    return [x for x in lst if x != item] if item in lst else lst + [item]


def available_categories(assets: dict) -> list:
    present = {a.category for a in assets.values()}
    return [c for c in CATEGORIES if c in present]      # only what the provider really supports


def eligible_assets(assets: dict, cats: list) -> list:
    return sorted((a for a in assets.values() if a.category in cats), key=lambda a: (a.category, a.display))


def kb_categories(selected: list, available: list) -> list:
    rows = [[(("✅ " if c in selected else "▫️ ") + c, f"cat:{c}")] for c in available]
    all_sel = bool(available) and all(c in selected for c in available)
    rows.append([(("✅ " if all_sel else "▫️ ") + "SELECT ALL", "cat:ALL")])
    rows.append([("Next ➡️", "cats_next")])
    return rows


def kb_pairs(elig: list, selected: list, page: int) -> list:
    pages = max(1, math.ceil(len(elig) / PER_PAGE))
    page = max(0, min(page, pages - 1))
    rows = []
    for i in range(page * PER_PAGE, min(len(elig), (page + 1) * PER_PAGE)):
        a = elig[i]
        rows.append([(("✅ " if a.pair in selected else "▫️ ") + a.display, f"pr:{i}")])
    all_sel = bool(elig) and all(a.pair in selected for a in elig)
    rows.append([(("✅ " if all_sel else "▫️ ") + "SELECT ALL", "pr:ALL")])
    nav = []
    if page > 0:
        nav.append(("⬅️ Prev", f"pg:{page - 1}"))
    if page < pages - 1:
        nav.append(("Next page ➡️", f"pg:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([("⬅️ Back", "back_cats"), ("Continue ➡️", "pairs_next")])
    return rows


def kb_strategies(selected: list, strategies: dict) -> list:
    rows, usable = [], []
    for key, label, members in STRAT_MENU:
        if any(m in strategies for m in members):
            usable.append(key)
            rows.append([(("✅ " if key in selected else "▫️ ") + label, f"st:{key}")])
    all_sel = bool(usable) and all(k in selected for k in usable)
    rows.append([(("✅ " if all_sel else "▫️ ") + "SELECT ALL", "st:ALL")])
    rows.append([("⬅️ Back", "back_pairs"), ("▶️ Start signals", "go")])
    return rows


class TelegramUI:
    def __init__(self, app: BotApp):
        self.app, self.cfg, self.db = app, app.cfg, app.db

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def markup(rows: list):
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        out = []
        for row in rows:
            out.append([InlineKeyboardButton(t, url=d[1]) if isinstance(d, tuple) else
                        InlineKeyboardButton(t, callback_data=d) for t, d in row])
        return InlineKeyboardMarkup(out)

    def join_url(self) -> str:
        if self.cfg.channel_invite_url:
            return self.cfg.channel_invite_url
        return f"https://t.me/{self.cfg.channel_username}" if self.cfg.channel_username else ""

    async def _edit(self, q, text: str, rows: Optional[list] = None):
        try:
            await q.edit_message_text(text=text, reply_markup=self.markup(rows) if rows else None)
        except Exception as e:
            if "not modified" not in str(e).lower():
                log.warning("edit failed: %s", e)

    async def _admin_only(self, update) -> bool:
        uid = update.effective_user.id
        if is_admin(self.cfg, uid):
            return True
        self.db.log_admin(uid, "DENIED", (update.effective_message.text or "")[:40] if update.effective_message else "")
        if update.effective_message:
            await update.effective_message.reply_text("Unknown command.")   # reveal nothing
        return False

    # -- user flow -------------------------------------------------------
    async def cmd_start(self, update, context):
        uid = update.effective_user.id
        user = self.db.get_user(uid)
        if is_admin(self.cfg, uid) and not user["verified"]:
            self.db.set_user(uid, verified=1)
            user["verified"] = 1
        admin = f"@{self.cfg.admin_username}" if self.cfg.admin_username else "-"
        text = "🤖 STM PRO BOT\n" + tr("welcome", self.cfg, channel=self.cfg.channel_name, admin=admin)
        if user["verified"] or not self.cfg.require_membership:
            if not user["verified"]:
                self.db.set_user(uid, verified=1)
            rows = [[("📡 Get Signal", "getsig")]]
        else:
            text += "\n\n" + tr("join_first", self.cfg)
            rows = []
            if self.join_url():
                rows.append([("📢 Join Channel", ("url", self.join_url()))])
            rows.append([("✅ I Joined - Verify", "verify")])
        if self.cfg.referral_url:
            rows.append([("🔗 Open Quotex", ("url", self.cfg.referral_url))])
        await update.effective_message.reply_text(text, reply_markup=self.markup(rows))

    async def on_callback(self, update, context):
        q = update.callback_query
        data, uid = q.data or "", q.from_user.id
        if not CB_RE.match(data):
            await q.answer("Invalid action")
            return
        if data.startswith("auto:"):
            return await self._auto_pick(q, uid, int(data.split(":")[1]))
        user = self.db.get_user(uid)
        if data == "verify":
            return await self._verify(q, context, uid)
        if not user["verified"]:
            await q.answer(tr("need_verify", self.cfg), show_alert=True)
            return
        await q.answer()
        assets = self.app.assets
        if data == "getsig" or data == "back_cats":
            await self._edit(q, tr("pick_cats", self.cfg),
                             kb_categories(user["cats"], available_categories(assets)))
        elif data.startswith("cat:"):
            avail, key = available_categories(assets), data[4:]
            if key == "ALL":
                cats = [] if all(c in user["cats"] for c in avail) else list(avail)
            elif key in avail:
                cats = toggle(user["cats"], key)
            else:
                return
            elig = {a.pair for a in eligible_assets(assets, cats)}
            self.db.set_user(uid, cats=cats, pairs=[p for p in user["pairs"] if p in elig])
            await self._edit(q, tr("pick_cats", self.cfg), kb_categories(cats, avail))
        elif data == "cats_next":
            if not user["cats"]:
                return await q.answer("Select at least one category", show_alert=True)
            self.db.set_user(uid, page=0)
            await self._show_pairs(q, user, 0)
        elif data.startswith("pg:"):
            self.db.set_user(uid, page=int(data[3:]))
            await self._show_pairs(q, user, int(data[3:]))
        elif data.startswith("pr:"):
            elig = eligible_assets(assets, user["cats"])
            key = data[3:]
            if key == "ALL":
                pairs = [] if all(a.pair in user["pairs"] for a in elig) else [a.pair for a in elig]
            else:
                i = int(key)
                if not 0 <= i < len(elig):
                    return
                pairs = toggle(user["pairs"], elig[i].pair)
            self.db.set_user(uid, pairs=pairs)
            user["pairs"] = pairs
            await self._show_pairs(q, user, user["page"])
        elif data == "pairs_next" or data == "back_pairs":
            if data == "pairs_next" and not user["pairs"]:
                return await q.answer("Select at least one pair", show_alert=True)
            await self._edit(q, tr("pick_strats", self.cfg), kb_strategies(user["strategies"], self.app.strategies))
        elif data.startswith("st:"):
            usable = [k for k, _l, m in STRAT_MENU if any(x in self.app.strategies for x in m)]
            key = data[3:]
            if key == "ALL":
                sel = [] if all(k in user["strategies"] for k in usable) else usable
            elif key in usable:
                sel = toggle(user["strategies"], key)
            else:
                return await q.answer("That strategy is disabled", show_alert=True)
            self.db.set_user(uid, strategies=sel)
            await self._edit(q, tr("pick_strats", self.cfg), kb_strategies(sel, self.app.strategies))
        elif data == "go":
            await self._go(q, context, uid, user)
        elif data == "stop":
            self.db.set_user(uid, active=0)
            await self._edit(q, tr("stopped_user", self.cfg), [[("📡 Get Signal", "getsig")]])

    async def _show_pairs(self, q, user, page: int):
        elig = eligible_assets(self.app.assets, user["cats"])
        pages = max(1, math.ceil(len(elig) / PER_PAGE))
        page = max(0, min(page, pages - 1))
        await self._edit(q, tr("pick_pairs", self.cfg, p=page + 1, n=pages), kb_pairs(elig, user["pairs"], page))

    async def _verify(self, q, context, uid: int):
        res = await check_membership(context.bot, self.cfg, uid)
        if res is None and self.cfg.membership_fail_open:
            res = True
        if res is None:
            return await q.answer(tr("cannot_verify", self.cfg), show_alert=True)
        if res is False:
            return await q.answer(tr("not_member", self.cfg), show_alert=True)
        await q.answer()
        self.db.set_user(uid, verified=1)
        if self.cfg.show_interstitial:
            for key in ("risk1", "risk2", "risk3"):               # 3 edits only, no flooding
                await self._edit(q, tr(key, self.cfg))
                await asyncio.sleep(1.5)
        await self._edit(q, tr("verified", self.cfg), [[("📡 Get Signal", "getsig")]])

    async def _go(self, q, context, uid: int, user: dict):
        if not user["pairs"] or not user["strategies"]:
            return await q.answer("Select pairs and strategies first", show_alert=True)
        if self.cfg.require_membership:
            res = await check_membership(context.bot, self.cfg, uid)
            if res is False or (res is None and not self.cfg.membership_fail_open):
                self.db.set_user(uid, verified=0, active=0)
                return await q.answer(tr("not_member", self.cfg), show_alert=True)
        self.db.set_user(uid, active=1)
        text = tr("started", self.cfg, n=len(user["pairs"]), lead=int(self.cfg.lead_seconds))
        text += f"\nMin payout: {self.cfg.min_payout:g}% | Timezone: {self.cfg.timezone}\n" + tr("latency", self.cfg)
        text += "\n⚠️ " + tr("disclaimer", self.cfg)
        await self._edit(q, text, [[("⏹ Stop my signals", "stop")]])

    # -- stats / help ----------------------------------------------------
    async def cmd_stats(self, update, context):
        uid = update.effective_user.id
        if not self.db.get_user(uid)["verified"]:
            return await update.effective_message.reply_text(tr("need_verify", self.cfg))
        t0, t1 = day_bounds(self.cfg, utcnow())
        msg = format_stats(self.db.stats(t0, t1), "Today") + "\n\n" + format_stats(self.db.stats(), "All time")
        await update.effective_message.reply_text(msg + "\n⚠️ " + tr("disclaimer", self.cfg).split("\n")[0])

    async def cmd_help(self, update, context):
        txt = ("/start - welcome & verification\n/stats - verified statistics\n/help - this message\n\n"
               "Signals are for 1-minute expiry, sent about "
               f"{int(self.cfg.lead_seconds)}s before the entry minute. This bot never trades for you.")
        if is_admin(self.cfg, update.effective_user.id):
            txt += (f"\n\nAdmin: /{self.cfg.admin_stop_command} stop signals, /STMUTO auto-run, /STMON start, "
                    "/pause, /resume, /status")
        await update.effective_message.reply_text(txt)

    # -- admin -----------------------------------------------------------
    async def cmd_stop(self, update, context):
        if not await self._admin_only(update):
            return
        uid = update.effective_user.id
        self.app.control.set_state("stopped")
        self.db.log_admin(uid, "STOP")
        users = [u["user_id"] for u in self.db.active_users()]
        await self.app.notifier.send_many(users, "⛔ " + tr("engine_stopped", self.cfg))
        await update.effective_message.reply_text(
            f"Signals stopped. {self.db.open_count()} open signal(s) will still be resolved and reported.")

    async def cmd_auto(self, update, context):
        if not await self._admin_only(update):
            return
        rows = [[(str(n), f"auto:{n}") for n in (10, 20, 25)], [(str(n), f"auto:{n}") for n in (30, 40)]]
        await update.effective_message.reply_text(
            "How many signals? Only signals actually SENT count; skipped candidates do NOT count. "
            "The run stops after exactly N sent signals.", reply_markup=self.markup(rows))

    async def _auto_pick(self, q, uid: int, n: int):
        if not is_admin(self.cfg, uid):
            self.db.log_admin(uid, "DENIED", f"auto:{n}")
            return await q.answer("Not allowed", show_alert=True)
        await q.answer()
        self.app.control.auto_start(n)
        self.db.log_admin(uid, "AUTO", str(n))
        extra = " (cooldown active - starts after it ends)" if self.app.control.in_cooldown(time.time()) else ""
        await self._edit(q, f"🔁 Auto run started: 0/{n} signals{extra}. Progress is sent after each signal.")

    async def _resume(self, update, action: str, new_state: str):
        if not await self._admin_only(update):
            return
        uid = update.effective_user.id
        force = bool(update.effective_message.text and update.effective_message.text.lower().endswith("force"))
        if new_state == "running" and self.app.control.in_cooldown(time.time()):
            if not force:
                return await update.effective_message.reply_text(
                    "Cooldown is active after a loss streak. Send the command with 'force' to override.")
            self.app.control.clear_cooldown()
        self.app.control.set_state(new_state)
        self.db.log_admin(uid, action, "force" if force else "")
        await update.effective_message.reply_text(f"State: {new_state}")

    async def cmd_on(self, update, context):
        await self._resume(update, "ON", "running")

    async def cmd_resume(self, update, context):
        await self._resume(update, "RESUME", "running")

    async def cmd_pause(self, update, context):
        await self._resume(update, "PAUSE", "paused")

    async def cmd_status(self, update, context):
        if not await self._admin_only(update):
            return
        c, now = self.app.control, time.time()
        t0, t1 = day_bounds(self.cfg, utcnow())
        sent, target = c.auto
        cd = f"until {local_str(c.cooldown_until, self.cfg)}" if c.in_cooldown(now) else "no"
        txt = (f"State: {c.state}\nCooldown: {cd} | Loss streak: {c.loss_streak}/{self.cfg.loss_streak_limit}\n"
               f"Auto run: {sent}/{target}\nProvider: {self.app.provider.name} | Candle mode: {self.cfg.candle_mode}\n"
               f"Assets: {len(self.app.assets)} | Active users: {len(self.db.active_users())} | "
               f"Open signals: {self.db.open_count()}\nLast cycle: {self.app.last_cycle or '-'}\n\n"
               + format_stats(self.db.stats(t0, t1), "Today"))
        await update.effective_message.reply_text(txt)

    # -- wiring ----------------------------------------------------------
    def build(self):
        from telegram.ext import Application, CallbackQueryHandler, CommandHandler, filters
        priv = filters.ChatType.PRIVATE

        async def post_init(application):
            self.app.notifier = TelegramNotifier(application.bot, self.cfg, self.db)
            try:
                await self.app.start()
            except ProviderError as e:
                log.error("PROVIDER STARTUP FAILED: %s", e)
                log.error("Fix the provider configuration, or set PROVIDER=mock to test without Quotex.")
                raise SystemExit(2)

        async def post_shutdown(application):
            await self.app.shutdown()

        application = (Application.builder().token(self.cfg.bot_token)
                       .post_init(post_init).post_shutdown(post_shutdown).build())
        add = application.add_handler
        add(CommandHandler("start", self.cmd_start, filters=priv))
        add(CommandHandler("help", self.cmd_help, filters=priv))
        add(CommandHandler("stats", self.cmd_stats, filters=priv))
        add(CommandHandler(self.cfg.admin_stop_command.lower(), self.cmd_stop, filters=priv))
        add(CommandHandler("stmuto", self.cmd_auto, filters=priv))
        add(CommandHandler("stmon", self.cmd_on, filters=priv))
        add(CommandHandler("pause", self.cmd_pause, filters=priv))
        add(CommandHandler("resume", self.cmd_resume, filters=priv))
        add(CommandHandler("status", self.cmd_status, filters=priv))
        add(CallbackQueryHandler(self.on_callback))
        return application


# ============================================================== BACKTEST ===
def settle_next(direction: str, nxt) -> str:
    return settle(direction, float(nxt["open"]), float(nxt["close"]))


def run_backtest(cfg: Config, pair: str = "EURUSD", n: int = 3000, train_frac: float = 0.7) -> dict:
    """Chronological split. Signals use rows <= i only; the outcome is candle i+1
    (open -> close). Indicators are causal, so prepare() once is equivalent to
    preparing each prefix (verified by a unit test). Thresholds are NOT tuned on test data."""
    prov = MockProvider()
    end = floor_minute(utcnow())
    raw = asyncio.run(prov.get_candles(pair, n, end, False))
    df = prepare(raw, cfg)
    strategies = build_strategies(cfg)
    split = int(len(df) * train_frac)
    ctx = {"pair": pair, "cfg": cfg}
    report = {}
    for name, strat in strategies.items():
        rec = {"train": {"W": 0, "L": 0, "D": 0}, "test": {"W": 0, "L": 0, "D": 0}}
        buckets: dict = {}
        for i in range(100, len(df) - 1):
            cand = strat.evaluate(df.iloc[:i + 1], ctx)
            if cand is None or (not cand.experimental and cand.score < cfg.min_confidence):
                continue
            r = settle_next(cand.direction, df.iloc[i + 1])
            part = "train" if i < split else "test"
            rec[part]["W" if r == "WIN" else "L" if r == "LOSS" else "D"] += 1
            if part == "test" and r != "DRAW":
                b = int(cand.score // 10) * 10
                w, t = buckets.get(b, (0, 0))
                buckets[b] = (w + (r == "WIN"), t + 1)
        report[name] = {"counts": rec, "buckets": buckets}
    return report


def print_backtest(rep: dict) -> None:
    def pct(c):
        t = c["W"] + c["L"]
        return f"{100 * c['W'] / t:5.1f}% of {t:4d}" if t else "   n/a"
    print("BACKTEST on SYNTHETIC mock data (chronological 70/30 split, no look-ahead).")
    print("Mock prices are not a real market: these numbers say nothing about live results.\n")
    print(f"{'strategy':<11}{'train win rate':>22}{'test win rate':>22}")
    for name, r in rep.items():
        print(f"{name:<11}{pct(r['counts']['train']):>22}{pct(r['counts']['test']):>22}")
    print("\nTest-set score buckets (score range -> realised win rate):")
    for name, r in rep.items():
        for b in sorted(r["buckets"]):
            w, t = r["buckets"][b]
            print(f"  {name:<10} {b:>2}-{b + 9:<2}: {100 * w / t:5.1f}% ({t} signals)")


# ================================================================== CLI ====
def run_bot(cfg: Config) -> None:
    try:
        import telegram  # noqa: F401
    except ImportError:
        sys.exit("python-telegram-bot is not installed. Run: pip install -r requirements.txt")
    db = Database(cfg.database_path)
    provider = make_provider(cfg)
    app = BotApp(cfg, db, provider)
    ui = TelegramUI(app)
    log.info("starting (provider=%s, tz=%s, min payout=%s%%)", cfg.provider, cfg.timezone, cfg.min_payout)
    ui.build().run_polling(allowed_updates=["message", "callback_query"], drop_pending_updates=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="STM PRO BOT (signal-only)")
    ap.add_argument("--check", action="store_true", help="validate configuration and exit")
    ap.add_argument("--backtest", action="store_true", help="run the mock-data backtest")
    ap.add_argument("--diagnose-provider", action="store_true", help="inspect the pyquotex install")
    args = ap.parse_args(argv)
    try:
        cfg = Config.from_env()
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}")
        return 2
    setup_logging(cfg)
    if args.backtest:
        print_backtest(run_backtest(cfg))
        return 0
    if args.diagnose_provider:
        asyncio.run(diagnose_provider(cfg))
        return 0
    errs = cfg.validate(need_telegram=True)
    if errs:
        print("CONFIGURATION PROBLEMS (values are never printed):")
        for e in errs:
            print(" -", e)
        print("Copy .env.example to .env and fill it in.")
        return 2
    if args.check:
        print("Configuration OK.")
        return 0
    run_bot(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
