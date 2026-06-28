import asyncio
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_URL   = "https://api.binance.com/api/v3/klines"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
DB_PATH           = os.getenv("DB_PATH", "bot.db")

# Ngắn hạn: scalp trong ngày — 15m timing, 1H momentum, 4H xu hướng chính
SHORT_TERM_TIMEFRAMES = {
    "15M": ("15m", 200),
    "1H":  ("1h",  150),
    "4H":  ("4h",  100),
}

# Dài hạn: swing/position — 4H entry, 1D xu hướng, 1W big picture
LONG_TERM_TIMEFRAMES = {
    "4H": ("4h",  150),
    "1D": ("1d",  150),
    "1W": ("1w",  100),
}

# Sau bao nhiêu giờ thì tự động check WIN/LOSS
RESULT_CHECK_HOURS = {
    "short": 12,
    "long":  24,
}

PREDICTION_HISTORY_COUNT = 5


# ─── DB ───────────────────────────────────────────────────────────────────────

def init_prediction_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT NOT NULL,
                mode                TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                check_after_hours   INTEGER NOT NULL,
                next_check_at       TEXT,
                direction           TEXT NOT NULL,
                entry_low           REAL,
                entry_high          REAL,
                sl                  REAL,
                tp1                 REAL,
                tp2                 REAL,
                market_snapshot     TEXT,
                reasoning_summary   TEXT,
                full_response       TEXT,
                result              TEXT NOT NULL DEFAULT 'PENDING',
                result_price        REAL,
                result_reason       TEXT,
                result_checked_at   TEXT
            )
        """)
        # Migration: thêm cột mới cho DB cũ nếu chưa có
        for col, definition in [
            ("reasoning_summary", "TEXT"),
            ("full_response",     "TEXT"),
            ("result_reason",     "TEXT"),
            ("market_snapshot",   "TEXT"),
            ("next_check_at",     "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        rows = conn.execute(
            "SELECT id, created_at, check_after_hours FROM predictions "
            "WHERE result = 'PENDING' AND (next_check_at IS NULL OR next_check_at = '')"
        ).fetchall()
        for pid, created_at, check_hours in rows:
            try:
                created = parse_utc_datetime(created_at)
                if created is None:
                    continue
                next_check_at = (created + timedelta(hours=int(check_hours))).isoformat()
                conn.execute(
                    "UPDATE predictions SET next_check_at=? WHERE id=?",
                    (next_check_at, pid),
                )
            except Exception:
                pass
        conn.commit()


def save_prediction(
    symbol: str,
    mode: str,
    direction: str,
    entry_low: float | None,
    entry_high: float | None,
    sl: float | None,
    tp1: float | None,
    tp2: float | None,
    market_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
) -> int:
    created_dt = datetime.now(timezone.utc)
    now = created_dt.isoformat()
    check_hours = RESULT_CHECK_HOURS.get(mode, 24)
    next_check_at = (created_dt + timedelta(hours=check_hours)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (symbol, mode, created_at, check_after_hours, next_check_at, direction,
                 entry_low, entry_high, sl, tp1, tp2,
                 market_snapshot, reasoning_summary, full_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, mode, now, check_hours, next_check_at, direction,
             entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, reasoning_summary, full_response),
        )
        conn.commit()
        return cursor.lastrowid


def get_pending_predictions() -> list[dict]:
    """
    Chỉ lấy prediction PENDING đã đến hạn check.
    Không quét toàn bộ PENDING rồi tự tính trong Python nữa.
    """
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, symbol, mode, created_at, check_after_hours, next_check_at, "
            "direction, entry_low, entry_high, sl, tp1, tp2 "
            "FROM predictions "
            "WHERE result = 'PENDING' AND next_check_at IS NOT NULL AND next_check_at <= ?",
            (now,),
        ).fetchall()

    return [
        {
            "id": row[0],
            "symbol": row[1],
            "mode": row[2],
            "created_at": row[3],
            "check_after_hours": row[4],
            "next_check_at": row[5],
            "direction": row[6],
            "entry_low": row[7],
            "entry_high": row[8],
            "sl": row[9],
            "tp1": row[10],
            "tp2": row[11],
        }
        for row in rows
    ]


def update_prediction_result(
    pid: int,
    result: str,
    result_price: float,
    result_reason: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE predictions SET result=?, result_price=?, result_reason=?, result_checked_at=? WHERE id=?",
            (result, result_price, result_reason, now, pid),
        )
        conn.commit()


def get_recent_predictions(symbol: str, mode: str, limit: int = PREDICTION_HISTORY_COUNT) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT created_at, direction, entry_low, entry_high, sl, tp1, tp2,
                   reasoning_summary, full_response, result, result_price, result_reason,
                   market_snapshot
            FROM predictions
            WHERE symbol=? AND mode=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, mode, limit),
        ).fetchall()

    return [
        {
            "created_at":        row[0],
            "direction":         row[1],
            "entry_low":         row[2],
            "entry_high":        row[3],
            "sl":                row[4],
            "tp1":               row[5],
            "tp2":               row[6],
            "reasoning_summary": row[7],
            "full_response":     row[8],
            "result":            row[9],
            "result_price":      row[10],
            "result_reason":     row[11],
            "market_snapshot":   row[12],
        }
        for row in rows
    ]


# ─── Auto WIN/LOSS check ──────────────────────────────────────────────────────

def get_current_price_raw(symbol: str) -> float | None:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=30,
        )
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception:
        return None


def evaluate_prediction(pred: dict, current_price: float) -> str:
    direction, sl, tp1 = pred["direction"], pred["sl"], pred["tp1"]
    if not sl or not tp1:
        return "UNKNOWN"
    if direction == "LONG":
        if current_price >= tp1:   return "WIN"
        if current_price <= sl:    return "LOSS"
    elif direction == "SHORT":
        if current_price <= tp1:   return "WIN"
        if current_price >= sl:    return "LOSS"
    return "PENDING_STILL"


def parse_utc_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_binance_klines_since(
    symbol: str,
    interval: str,
    start: datetime,
    limit: int = 1000,
) -> pd.DataFrame | None:
    try:
        r = requests.get(
            BINANCE_API_URL,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": int(start.timestamp() * 1000),
                "limit": limit,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        print(f"Historical Binance error {symbol} {interval}: {exc}")
        return None


def evaluate_prediction_with_candles(pred: dict, candles: pd.DataFrame | None) -> tuple[str, float | None, str]:
    """
    Chấm prediction theo đúng flow lệnh chờ:
    1) Giá phải chạm vùng Entry trước.
    2) Sau khi Entry được khớp, mới bắt đầu tính TP1/SL.
    3) Nếu hết hạn mà chưa chạm Entry: NOT_FILLED.
    """
    direction = pred["direction"]
    entry_low = pred.get("entry_low")
    entry_high = pred.get("entry_high")
    sl = pred.get("sl")
    tp1 = pred.get("tp1")

    if not sl or not tp1:
        return "UNKNOWN", None, "Missing SL or TP1, cannot evaluate outcome."
    if candles is None or candles.empty:
        return "UNKNOWN", None, "No candle data after prediction."

    created = parse_utc_datetime(pred["created_at"])
    if created is not None:
        candles = candles[candles["close_time"] >= pd.Timestamp(created)]
    if candles.empty:
        return "UNKNOWN", None, "No closed candle after prediction."

    # Nếu không parse được Entry thì fallback về cách cũ: xem như đã vào lệnh ngay.
    require_entry_fill = entry_low is not None and entry_high is not None
    entry_filled = not require_entry_fill
    entry_time = None

    for _, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        closed_at = str(row["close_time"])[:16]

        if not entry_filled:
            touched_entry = low <= float(entry_high) and high >= float(entry_low)
            if not touched_entry:
                continue
            entry_filled = True
            entry_time = closed_at

            # Trong cùng một nến vừa chạm Entry mà cũng chạm TP/SL thì không biết thứ tự.
            if direction == "LONG":
                same_candle_tp = high >= tp1
                same_candle_sl = low <= sl
            elif direction == "SHORT":
                same_candle_tp = low <= tp1
                same_candle_sl = high >= sl
            else:
                same_candle_tp = same_candle_sl = False

            if same_candle_tp or same_candle_sl:
                return (
                    "AMBIGUOUS",
                    close,
                    f"Entry zone and TP1/SL were touched in the same 15m candle at {closed_at}; order is unknown.",
                )
            continue

        if direction == "LONG":
            hit_tp = high >= tp1
            hit_sl = low <= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 and SL both touched in the same 15m candle at {closed_at}."
            if hit_tp:
                return "WIN", tp1, f"Entry filled at {entry_time or 'prediction time'}; TP1 touched before SL at {closed_at}."
            if hit_sl:
                return "LOSS", sl, f"Entry filled at {entry_time or 'prediction time'}; SL touched before TP1 at {closed_at}."

        if direction == "SHORT":
            hit_tp = low <= tp1
            hit_sl = high >= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 and SL both touched in the same 15m candle at {closed_at}."
            if hit_tp:
                return "WIN", tp1, f"Entry filled at {entry_time or 'prediction time'}; TP1 touched before SL at {closed_at}."
            if hit_sl:
                return "LOSS", sl, f"Entry filled at {entry_time or 'prediction time'}; SL touched before TP1 at {closed_at}."

    last_close = float(candles.iloc[-1]["close"])
    if require_entry_fill and not entry_filled:
        return "NOT_FILLED", last_close, "Entry zone was not touched before expiry."
    return "PENDING_STILL", last_close, "Entry was filled, but TP1 or SL was not touched before expiry."


async def auto_check_pending_predictions() -> list[str]:
    init_prediction_db()
    due      = get_pending_predictions()
    messages = []
    for pred in due:
        created = parse_utc_datetime(pred["created_at"])
        if created is None:
            continue

        candles = await asyncio.to_thread(
            get_binance_klines_since,
            pred["symbol"],
            "15m",
            created,
        )
        result, outcome_price, reason = evaluate_prediction_with_candles(pred, candles)

        price = outcome_price
        if price is None:
            price = await asyncio.to_thread(get_current_price_raw, pred["symbol"])
        if price is None:
            continue
        if result == "PENDING_STILL":
            result = "EXPIRED"
        update_prediction_result(pred["id"], result, price, reason)
        emoji = {"WIN": "✅", "LOSS": "❌", "NOT_FILLED": "🚫", "AMBIGUOUS": "⚠️"}.get(result, "⏱")
        messages.append(
            f"{emoji} [{pred['symbol']} {pred['mode'].upper()}] "
            f"{pred['direction']} từ {pred['created_at'][:10]} → {result} "
            f"(giá check: {price:,.4f})"
        )
    return messages


# ─── Binance + Indicators ─────────────────────────────────────────────────────

def get_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    try:
        r = requests.get(
            BINANCE_API_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "count",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        print(f"Lỗi Binance {symbol} {interval}: {exc}")
        return None


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    delta = data.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = pd.Series(np.nan, index=data.index, dtype="float64")
    avg_loss = pd.Series(np.nan, index=data.index, dtype="float64")
    if len(data) <= period:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    avg_gain.iloc[period] = gain.iloc[1: period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1: period + 1].mean()
    for i in range(period + 1, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    return rsi


def calculate_macd(data: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = calculate_ema(data, fast)
    ema_slow    = calculate_ema(data, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < 60:
        return None
    r = df.copy()
    r["ema_7"],  r["ema_25"], r["ema_50"] = (
        calculate_ema(r["close"], 7),
        calculate_ema(r["close"], 25),
        calculate_ema(r["close"], 50),
    )
    r["rsi_6"],  r["rsi_14"] = calculate_rsi(r["close"], 6), calculate_rsi(r["close"], 14)
    r["macd_line"], r["macd_signal"], r["macd_hist"] = calculate_macd(r["close"])
    r["vol_ma20"]  = r["volume"].rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
    r["atr_14"]    = calculate_atr(r, 14)
    return r.dropna().reset_index(drop=True)


# ─── Format helpers ───────────────────────────────────────────────────────────

def fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 100:
        return f"{v:,.{decimals}f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    return f"{v:,.8f}"


def summarize_price_action_features(df: pd.DataFrame, lookback: int = 30) -> str:
    """Tóm tắt hành vi giá từ nhiều nến, để Claude không phải đọc quá nhiều nến thô."""
    if df is None or df.empty:
        return "Thống kê hành vi giá: Không đủ dữ liệu."

    window = df.tail(lookback).copy()
    if window.empty:
        return "Thống kê hành vi giá: Không đủ dữ liệu."

    close_first = float(window.iloc[0]["close"])
    close_last = float(window.iloc[-1]["close"])
    high = float(window["high"].max())
    low = float(window["low"].min())
    range_value = high - low
    change = close_last - close_first
    change_pct = (change / close_first * 100) if close_first else 0

    body = (window["close"] - window["open"]).abs()
    candle_range = (window["high"] - window["low"]).replace(0, np.nan)
    body_ratio = (body / candle_range).replace([np.inf, -np.inf], np.nan)

    bullish = int((window["close"] > window["open"]).sum())
    bearish = int((window["close"] < window["open"]).sum())

    # Đếm chuỗi nến cùng màu ở cuối window
    last_dir = 0
    streak = 0
    for _, row in window.iloc[::-1].iterrows():
        cur_dir = 1 if row["close"] > row["open"] else (-1 if row["close"] < row["open"] else 0)
        if last_dir == 0:
            last_dir = cur_dir
            streak = 1 if cur_dir != 0 else 0
        elif cur_dir == last_dir and cur_dir != 0:
            streak += 1
        else:
            break
    streak_label = "xanh" if last_dir > 0 else ("đỏ" if last_dir < 0 else "trung tính")

    largest_bull = window[window["close"] > window["open"]].copy()
    largest_bear = window[window["close"] < window["open"]].copy()
    largest_bull_move = float((largest_bull["close"] - largest_bull["open"]).max()) if not largest_bull.empty else 0.0
    largest_bear_move = float((largest_bear["open"] - largest_bear["close"]).max()) if not largest_bear.empty else 0.0

    vol_avg = float(window["vol_ratio"].mean()) if "vol_ratio" in window.columns else np.nan
    close_pos = ((close_last - low) / range_value * 100) if range_value else 50

    return (
        f"Thống kê {len(window)} nến: đổi {fmt(change)} ({change_pct:.2f}%), "
        f"biên độ {fmt(range_value)}, xanh/đỏ {bullish}/{bearish}, "
        f"chuỗi cuối {streak} nến {streak_label}, "
        f"nến xanh lớn nhất {fmt(largest_bull_move)}, nến đỏ lớn nhất {fmt(largest_bear_move)}, "
        f"volume TB {fmt(vol_avg, 2)}x, vị trí đóng cửa trong biên độ {close_pos:.0f}%."
    )


def summarize_timeframe(label: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return f"\nKHUNG {label}: Không đủ dữ liệu.\n"

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    ema7, ema25, ema50 = last["ema_7"], last["ema_25"], last["ema_50"]

    if ema7 > ema25 > ema50:
        ema_align = "TĂNG (EMA7>EMA25>EMA50)"
    elif ema7 < ema25 < ema50:
        ema_align = "GIẢM (EMA7<EMA25<EMA50)"
    else:
        ema_align = "TRUNG TÍNH (đan xen)"

    macd_dir = "TĂNG" if last["macd_hist"] > 0 else "GIẢM"
    macd_cross = ""
    if prev["macd_hist"] < 0 <= last["macd_hist"]:
        macd_cross = " — VỪA CROSS BULLISH"
    elif prev["macd_hist"] > 0 >= last["macd_hist"]:
        macd_cross = " — VỪA CROSS BEARISH"

    vol_lbl = "CAO" if last["vol_ratio"] > 1.5 else ("THẤP" if last["vol_ratio"] < 0.7 else "BÌNH THƯỜNG")

    window    = df.tail(50)
    key_high  = window["high"].max()
    key_low   = window["low"].min()

    candles = "\n".join(
        f"  {str(row['timestamp'])[:16]} O:{fmt(row['open'])} H:{fmt(row['high'])} "
        f"L:{fmt(row['low'])} C:{fmt(row['close'])} "
        f"RSI14:{fmt(row['rsi_14'],1)} Vol:{fmt(row['vol_ratio'],2)}x"
        for _, row in df.tail(10).iterrows()
    )

    return "\n".join([
        f"\nKHUNG {label}:",
        f"  Giá: {fmt(last['close'])} | Nến trước: {fmt(prev['close'])}",
        f"  EMA7={fmt(ema7)} EMA25={fmt(ema25)} EMA50={fmt(ema50)} → {ema_align}",
        f"  RSI(6)={fmt(last['rsi_6'],1)} RSI(14)={fmt(last['rsi_14'],1)}",
        f"  MACD={fmt(last['macd_line'],4)} Signal={fmt(last['macd_signal'],4)} Hist={fmt(last['macd_hist'],4)} → {macd_dir}{macd_cross}",
        f"  ATR(14)={fmt(last['atr_14'])}",
        f"  Volume={fmt(last['vol_ratio'],2)}x → {vol_lbl}",
        f"  High/Low 50 nến: {fmt(key_high)} / {fmt(key_low)}",
        f"  {summarize_price_action_features(df, 30)}",
        f"  10 nến gần nhất:",
        candles,
    ])


def estimate_liquidity_sweep_zones(
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
) -> str:
    """
    Ước lượng vùng quét thanh khoản/stop-loss từ high/low nến.
    Đây không phải dữ liệu liquidation heatmap thật.

    Thiết kế:
    - SCALP: vùng chính lấy từ 15M 50 nến, tham chiếu thêm 1H 24 nến.
      ATR an toàn lấy từ cả 15M và 1H để tránh SL/TP quá sát.
    - SWING: vùng chính lấy từ 4H 50 nến, tham chiếu thêm 1D 20 nến.
      ATR an toàn lấy từ cả 4H và 1D.
    """
    if mode == "short":
        main_label = "15M"
        confirm_label = "1H"
        main_window_size = 50      # 50 nến 15M ≈ 12.5 giờ
        confirm_window_size = 24   # 24 nến 1H ≈ 1 ngày
        min_stop_desc = "max(1.5 x ATR 15M, 0.6 x ATR 1H)"
    else:
        main_label = "4H"
        confirm_label = "1D"
        main_window_size = 50      # 50 nến 4H ≈ 8.3 ngày
        confirm_window_size = 20   # 20 nến 1D ≈ 20 ngày
        min_stop_desc = "max(1.5 x ATR 4H, 0.35 x ATR 1D)"

    def get_window_stats(label: str, window_size: int) -> dict | None:
        df = timeframe_data.get(label)
        if df is None or df.empty:
            return None
        window = df.tail(window_size)
        if window.empty:
            return None
        atr = None
        if "atr_14" in df.columns:
            try:
                raw_atr = float(df.iloc[-1]["atr_14"])
                if not np.isnan(raw_atr):
                    atr = raw_atr
            except Exception:
                atr = None
        return {
            "label": label,
            "window_size": len(window),
            "low": float(window["low"].min()),
            "high": float(window["high"].max()),
            "atr": atr,
        }

    main = get_window_stats(main_label, main_window_size)
    confirm = get_window_stats(confirm_label, confirm_window_size)

    if main is None:
        for label, df in timeframe_data.items():
            if df is None:
                continue
            main = get_window_stats(label, min(50, len(df)))
            if main is not None:
                break

    if main is None:
        return "VÙNG QUÉT THANH KHOẢN ƯỚC LƯỢNG: Không đủ dữ liệu."

    main_atr = main["atr"]
    confirm_atr = confirm["atr"] if confirm else None

    main_range = main["high"] - main["low"]
    main_buffer = (main_atr * 0.25) if main_atr else (main_range * 0.03)

    main_long_low = main["low"] - main_buffer
    main_long_high = main["low"]
    main_short_low = main["high"]
    main_short_high = main["high"] + main_buffer

    long_low = main_long_low
    long_high = main_long_high
    short_low = main_short_low
    short_high = main_short_high

    confirm_line = ""
    if confirm:
        confirm_range = confirm["high"] - confirm["low"]
        confirm_buffer = (confirm_atr * 0.20) if confirm_atr else (confirm_range * 0.02)
        confirm_long_low = confirm["low"] - confirm_buffer
        confirm_long_high = confirm["low"]
        confirm_short_low = confirm["high"]
        confirm_short_high = confirm["high"] + confirm_buffer

        long_low = min(long_low, confirm_long_low)
        long_high = max(long_high, confirm_long_high)
        short_low = min(short_low, confirm_short_low)
        short_high = max(short_high, confirm_short_high)
        confirm_line = (
            f"Tham chiếu {confirm['label']} {confirm['window_size']} nến: "
            f"Long {fmt(confirm_long_low)}–{fmt(confirm_long_high)} | "
            f"Short {fmt(confirm_short_low)}–{fmt(confirm_short_high)}"
        )

    stop_candidates = []
    if mode == "short":
        if main_atr:
            stop_candidates.append(1.5 * main_atr)
        if confirm_atr:
            stop_candidates.append(0.6 * confirm_atr)
    else:
        if main_atr:
            stop_candidates.append(1.5 * main_atr)
        if confirm_atr:
            stop_candidates.append(0.35 * confirm_atr)

    min_stop_distance = max(stop_candidates) if stop_candidates else None

    atr_parts = []
    if main_atr is not None:
        atr_parts.append(f"ATR {main['label']}: {fmt(main_atr)}")
    if confirm_atr is not None and confirm:
        atr_parts.append(f"ATR {confirm['label']}: {fmt(confirm_atr)}")
    atr_line = " | ".join(atr_parts) if atr_parts else "ATR tham chiếu: N/A"
    min_stop_line = (
        f"Khoảng SL tối thiểu: {fmt(min_stop_distance)} ({min_stop_desc})"
        if min_stop_distance is not None else
        f"Khoảng SL tối thiểu: N/A ({min_stop_desc})"
    )

    lines = [
        "VÙNG QUÉT THANH KHOẢN ƯỚC LƯỢNG:",
        f"Vùng chính {main['label']} {main['window_size']} nến: "
        f"Long {fmt(main_long_low)}–{fmt(main_long_high)} | "
        f"Short {fmt(main_short_low)}–{fmt(main_short_high)}",
    ]
    if confirm_line:
        lines.append(confirm_line)
    lines.extend([
        f"Vùng quét Long tổng hợp: {fmt(long_low)}–{fmt(long_high)}",
        f"Vùng quét Short tổng hợp: {fmt(short_low)}–{fmt(short_high)}",
        atr_line,
        min_stop_line,
        "Lưu ý: đây là vùng quét thanh khoản/stop-loss ước lượng từ high/low nến, không phải dữ liệu thanh lý thật.",
    ])
    return "\n".join(lines)

# ─── Fear & Greed ─────────────────────────────────────────────────────────────

def build_market_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_str: str,
) -> str:
    lines = [current_price_str, fear_greed_info]
    for label, df in timeframe_data.items():
        if df is None or df.empty:
            lines.append(f"{label}: no data")
            continue

        last = df.iloc[-1]
        ema_align = "mixed"
        if last["ema_7"] > last["ema_25"] > last["ema_50"]:
            ema_align = "bullish"
        elif last["ema_7"] < last["ema_25"] < last["ema_50"]:
            ema_align = "bearish"

        lines.append(
            f"{label}: close={fmt(last['close'])}, EMA={ema_align} "
            f"(7={fmt(last['ema_7'])},25={fmt(last['ema_25'])},50={fmt(last['ema_50'])}), "
            f"RSI14={fmt(last['rsi_14'], 1)}, MACD_hist={fmt(last['macd_hist'], 4)}, "
            f"vol={fmt(last['vol_ratio'], 2)}x"
        )

    return " | ".join(lines)


def get_fear_greed_index() -> str:
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=30)
        r.raise_for_status()
        payload = r.json()
        if payload.get("metadata", {}).get("error"):
            raise ValueError(payload["metadata"]["error"])
        value = int(payload["data"][0]["value"])
        label = payload["data"][0].get("value_classification", "")
        return f"Chỉ số Sợ hãi & Tham lam: {value}/100 ({label})"
    except Exception as exc:
        print(f"Fear/Greed error: {exc}")
        return "Chỉ số Sợ hãi & Tham lam: không có dữ liệu"


def get_current_price_str(symbol: str) -> tuple[str, float | None]:
    price = get_current_price_raw(symbol)
    if price is None:
        return "Giá hiện tại: không có dữ liệu", None
    return f"Giá hiện tại: {fmt(price)} USDT", price


# ─── History formatter ────────────────────────

def format_prediction_history(history: list[dict]) -> str:
    if not history:
        return "No previous analysis for this symbol/mode."

    lines = [f"RECENT LEARNING SUMMARY ({len(history)} latest analyses):"]
    finished = [p for p in history if p["result"] in ("WIN", "LOSS")]
    if finished:
        wins = sum(1 for p in finished if p["result"] == "WIN")
        win_rate = wins / len(finished) * 100
        lines.append(f"- Closed results: {wins}/{len(finished)} WIN, win rate {win_rate:.0f}%.")

        long_finished = [p for p in finished if p["direction"] == "LONG"]
        short_finished = [p for p in finished if p["direction"] == "SHORT"]
        if long_finished:
            long_wins = sum(1 for p in long_finished if p["result"] == "WIN")
            lines.append(f"- LONG: {long_wins}/{len(long_finished)} WIN.")
        if short_finished:
            short_wins = sum(1 for p in short_finished if p["result"] == "WIN")
            lines.append(f"- SHORT: {short_wins}/{len(short_finished)} WIN.")

    losses = [p for p in finished if p["result"] == "LOSS"]
    if losses:
        loss_dirs = [p["direction"] for p in losses]
        if loss_dirs.count("LONG") > loss_dirs.count("SHORT"):
            lines.append("- Repeated issue: recent LONG calls have more losses. Require stronger bullish confirmation.")
        elif loss_dirs.count("SHORT") > loss_dirs.count("LONG"):
            lines.append("- Repeated issue: recent SHORT calls have more losses. Require stronger bearish confirmation.")

    for i, p in enumerate(history, 1):
        entry = f"{fmt(p['entry_low'])}-{fmt(p['entry_high'])}" if p["entry_low"] and p["entry_high"] else "N/A"
        checked = f"checked price {fmt(p['result_price'])}" if p["result_price"] else "not checked"
        reason = p.get("result_reason") or "Outcome not checked yet."
        decision_reason = p.get("reasoning_summary") or "No decision reasoning summary."
        snapshot = p.get("market_snapshot") or "No market snapshot."
        lines.append(
            f"- #{i} {p['created_at'][:16]} {p['direction']} {p['result']} ({checked}); "
            f"Entry {entry}, SL {fmt(p['sl'])}, TP1 {fmt(p['tp1'])}, TP2 {fmt(p['tp2'])}. "
            f"Decision why: {decision_reason} Outcome: {reason} Market then: {snapshot}"
        )

    lines.append("Use this summary as learning context; do not copy old full responses.")
    return "\n".join(lines)


def build_user_prompt(
    symbol: str,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_str: str,
    history: list[dict],
) -> str:
    mode_label = "SCALP (ngắn hạn)" if mode == "short" else "SWING (dài hạn)"
    focus      = (
        "Dùng 15M để timing entry, 1H để xác nhận momentum, 4H để xác định xu hướng chính."
        if mode == "short" else
        "Dùng 4H để timing entry, 1D để xác nhận xu hướng, 1W để xác định big picture."
    )

    history_block    = format_prediction_history(history)
    tf_blocks        = "".join(summarize_timeframe(lbl, df) for lbl, df in timeframe_data.items())
    liquidity_block  = estimate_liquidity_sweep_zones(mode, timeframe_data)

    return f"""YÊU CẦU PHÂN TÍCH {mode_label} CHO {symbol}

{current_price_str}
{fear_greed_info}
Phương pháp: {focus}

═══════════════════════════════
{history_block}
═══════════════════════════════
{tf_blocks}
═══════════════════════════════
{liquidity_block}
═══════════════════════════════

Yêu cầu:
1. Đọc kỹ RECENT LEARNING SUMMARY, đặc biệt Decision why, Outcome và Market then, nhưng chỉ dùng nội bộ để điều chỉnh quyết định.
2. Không hiển thị RECENT LEARNING SUMMARY, lịch sử phân tích, hoặc mục "Nhìn lại lịch sử" trong câu trả lời cho user.
3. Không copy phân tích cũ. Chỉ dùng summary để tránh lặp lại lỗi.
4. QUYẾT ĐỊNH cuối cùng bắt buộc là LONG hoặc SHORT, không được chỉ CHỜ.
5. Phải dùng mục VÙNG QUÉT THANH KHOẢN ƯỚC LƯỢNG để chọn Entry, SL, TP1, TP2.
6. Toàn bộ câu trả lời phải là tiếng Việt tự nhiên, không chen từ tiếng Anh ngoài các ký hiệu được phép trong system prompt.
"""


# ─── Tóm tắt reasoning bằng call Haiku thứ 2 (rất ngắn, rẻ) ─────────────────

def summarize_reasoning(full_response: str) -> str:
    """
    Gọi Haiku lần 2 để tóm tắt lý do ra quyết định thành ~50 từ.
    Chi phí cực thấp (~50 input tokens + ~60 output tokens).
    Lưu vào DB để lần sau model học được pattern thực sự.
    """
    if not ANTHROPIC_API_KEY:
        return ""
    try:
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=120,
            temperature=0,
            messages=[{
                "role": "user",
                "content": (
                    "Tóm tắt trong 1-2 câu (tối đa 60 từ) lý do kỹ thuật chính "
                    "dẫn đến quyết định LONG/SHORT trong phân tích sau. "
                    "Chỉ nêu các chỉ báo cụ thể (EMA, RSI, MACD, volume) và mức giá. "
                    "Không giải thích, không lời mở đầu.\n\n"
                    + full_response[:2000]  # chỉ cần phần đầu là đủ
                ),
            }],
            timeout=60,
        )
        return "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    except Exception as exc:
        print(f"Lỗi summarize_reasoning: {exc}")
        return ""


# ─── Parse prediction từ output ──────────────────────────────────────────────

def parse_prediction_from_output(output: str) -> dict:
    def find_price(patterns: list[str], text: str | None = None) -> float | None:
        haystack = output if text is None else text
        for pat in patterns:
            m = re.search(pat, haystack, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).replace(",", ""))
                except Exception:
                    pass
        return None

    # Direction: ưu tiên dòng QUYẾT ĐỊNH, fallback emoji
    direction = "WAIT"
    m = re.search(r"QUYẾT ĐỊNH[:\s]+(LONG|SHORT)", output, re.IGNORECASE)
    if m:
        direction = m.group(1).upper()
    elif re.search(r"📈\s*LONG", output):
        direction = "LONG"
    elif re.search(r"📉\s*SHORT", output):
        direction = "SHORT"

    selected_output = output
    if direction in ("LONG", "SHORT"):
        section_match = re.search(
            rf"(?m)^\s*(?:📈|📉)?\s*{direction}\s*[—\-]",
            output,
            re.IGNORECASE,
        )
        if section_match:
            selected_output = output[section_match.start():]
            other_direction = "SHORT" if direction == "LONG" else "LONG"
            next_match = re.search(
                rf"(?m)^\s*(?:📈|📉)?\s*{other_direction}\s*[—\-]",
                selected_output[1:],
                re.IGNORECASE,
            )
            risk_match = re.search(r"\n\s*(?:⚠️|📊|Lưu ý|Rủi ro)", selected_output[1:], re.IGNORECASE)
            cut_points = [
                match.start() + 1
                for match in [next_match, risk_match]
                if match is not None
            ]
            if cut_points:
                selected_output = selected_output[:min(cut_points)]

    # Entry — có thể là range "95,000–95,500" hoặc đơn "95,000"
    entry_low = entry_high = None
    em = re.search(r"Entry[:\s]+([0-9,\.]+)(?:\s*[–\-]\s*([0-9,\.]+))?", selected_output, re.IGNORECASE)
    if em:
        try:
            entry_low  = float(em.group(1).replace(",", ""))
            entry_high = float(em.group(2).replace(",", "")) if em.group(2) else entry_low
        except Exception:
            pass

    sl  = find_price([r"SL[:\s]+([0-9,\.]+)"], selected_output)
    tp1 = find_price([r"TP1[:\s]+([0-9,\.]+)"], selected_output)
    tp2 = find_price([r"TP2[:\s]+([0-9,\.]+)"], selected_output)

    return {
        "direction":  direction,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    for p in [Path("analyze_system_prompt.txt"), Path("analysis_system_prompt.txt")]:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Không tìm thấy analyze_system_prompt.txt")


def call_claude_analysis(symbol: str, mode: str) -> str:

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env.")

    init_prediction_db()

    binance_symbol = f"{symbol.upper()}USDT"
    configs        = SHORT_TERM_TIMEFRAMES if mode == "short" else LONG_TERM_TIMEFRAMES

    timeframe_data: dict[str, pd.DataFrame | None] = {}
    for label, (interval, limit) in configs.items():
        timeframe_data[label] = add_indicators(get_binance_klines(binance_symbol, interval, limit))

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        raise RuntimeError(f"Could not fetch Binance data for {binance_symbol}.")

    system_prompt                    = load_system_prompt()
    fear_greed_info                  = get_fear_greed_index()
    current_price_str, current_price = get_current_price_str(binance_symbol)
    market_snapshot                  = build_market_snapshot(
        timeframe_data,
        fear_greed_info,
        current_price_str,
    )
    history                          = get_recent_predictions(binance_symbol, mode)
    user_prompt                      = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        history=history,
    )

    client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=300,
    )
    output = "".join(b.text for b in response.content if hasattr(b, "text"))

    # Parse prediction
    pred = parse_prediction_from_output(output)

    if pred["direction"] in ("LONG", "SHORT"):
        # Gọi Haiku lần 2 tóm tắt reasoning (~$0.0001 mỗi lần — cực rẻ)
        reasoning_summary = summarize_reasoning(output)

        save_prediction(
            symbol=binance_symbol,
            mode=mode,
            direction=pred["direction"],
            entry_low=pred["entry_low"],
            entry_high=pred["entry_high"],
            sl=pred["sl"],
            tp1=pred["tp1"],
            tp2=pred["tp2"],
            market_snapshot=market_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
        )

    return output


async def analyze_symbol(symbol: str, mode: str) -> str:
    return await asyncio.to_thread(call_claude_analysis, symbol, mode)
