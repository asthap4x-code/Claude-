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
