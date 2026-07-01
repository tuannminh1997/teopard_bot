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

# V4 lifecycle
# short = SCALP, long = SWING
ENTRY_WAIT_HOURS = {
    "short": 12,      # Scalp: chờ khớp Entry tối đa 12h
    "long": 24,       # Swing: chờ khớp Entry tối đa 24h
}

TRADE_MAX_HOLD_HOURS = {
    "short": 72,      # Scalp: sau khi khớp Entry, theo dõi tối đa 72h
    "long": 24 * 7,   # Swing: sau khi khớp Entry, theo dõi tối đa 7 ngày
}

CHECK_INTERVAL_HOURS = {
    "short": 1,       # Scalp: check mỗi 1h
    "long": 12,       # Swing: check mỗi 12h
}

RESULT_CHECK_INTERVAL = {
    "short": "15m",   # Scalp: chấm kết quả bằng nến 15 phút
    "long": "1h",     # Swing: chấm kết quả bằng nến 1 giờ
}


def get_result_check_interval(mode: str) -> str:
    return RESULT_CHECK_INTERVAL.get(mode, "15m")

PREDICTION_HISTORY_COUNT = 5
VN_TZ = timezone(timedelta(hours=7))


# ─── DB ───────────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def init_prediction_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER,
                chat_id             INTEGER,
                symbol              TEXT NOT NULL,
                mode                TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                check_after_hours   INTEGER NOT NULL DEFAULT 12,
                entry_wait_hours    INTEGER NOT NULL DEFAULT 12,
                max_hold_hours      INTEGER NOT NULL DEFAULT 72,
                next_check_at       TEXT,
                direction           TEXT NOT NULL,
                entry_low           REAL,
                entry_high          REAL,
                sl                  REAL,
                tp1                 REAL,
                tp2                 REAL,
                entry_status        TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
                entry_filled_at     TEXT,
                entry_price         REAL,
                trade_closed_at     TEXT,
                rr_result           REAL,
                hold_hours          REAL,
                market_snapshot     TEXT,
                feature_snapshot    TEXT,
                reasoning_summary   TEXT,
                full_response       TEXT,
                result              TEXT NOT NULL DEFAULT 'PENDING_ENTRY',
                result_price        REAL,
                result_reason       TEXT,
                result_checked_at   TEXT
            )
        """)
        for col, definition in [
            ("user_id", "INTEGER"),
            ("chat_id", "INTEGER"),
            ("check_after_hours", "INTEGER NOT NULL DEFAULT 12"),
            ("entry_wait_hours", "INTEGER NOT NULL DEFAULT 12"),
            ("max_hold_hours", "INTEGER NOT NULL DEFAULT 72"),
            ("next_check_at", "TEXT"),
            ("entry_status", "TEXT NOT NULL DEFAULT 'PENDING_ENTRY'"),
            ("entry_filled_at", "TEXT"),
            ("entry_price", "REAL"),
            ("trade_closed_at", "TEXT"),
            ("rr_result", "REAL"),
            ("hold_hours", "REAL"),
            ("reasoning_summary", "TEXT"),
            ("full_response", "TEXT"),
            ("result_reason", "TEXT"),
            ("market_snapshot", "TEXT"),
            ("feature_snapshot", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        # Migrate old PENDING rows to lifecycle naming.
        try:
            conn.execute("UPDATE predictions SET result='PENDING_ENTRY' WHERE result='PENDING'")
            conn.execute("UPDATE predictions SET entry_status='PENDING_ENTRY' WHERE entry_status IS NULL OR entry_status='' ")
        except sqlite3.OperationalError:
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
    feature_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
    user_id: int | None = None,
    chat_id: int | None = None,
) -> int:
    now = utc_now()
    entry_wait = ENTRY_WAIT_HOURS.get(mode, 24)
    max_hold = TRADE_MAX_HOLD_HOURS.get(mode, 72)
    next_check = now + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, chat_id, symbol, mode, created_at, check_after_hours, entry_wait_hours, max_hold_hours,
                 next_check_at, direction, entry_low, entry_high, sl, tp1, tp2,
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_ENTRY', ?, ?, ?, ?, 'PENDING_ENTRY')
            """,
            (user_id, chat_id, symbol, mode, iso(now), entry_wait, entry_wait, max_hold,
             iso(next_check), direction, entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response),
        )
        conn.commit()
        return cursor.lastrowid


def save_rejected_prediction(
    symbol: str,
    mode: str,
    direction: str | None,
    entry_low: float | None,
    entry_high: float | None,
    sl: float | None,
    tp1: float | None,
    tp2: float | None,
    market_snapshot: str | None,
    feature_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
    validation_errors: list[str],
    user_id: int | None = None,
    chat_id: int | None = None,
) -> int:
    """
    Lưu các phân tích bị Python validator từ chối để Claude học tránh lặp lại lỗi.

    Những dòng này KHÔNG được auto-check vì result='REJECTED_PLAN' không nằm trong
    query get_due_predictions(). Mục đích chỉ là learning/history/debug, không tính
    như WIN/LOSS.
    """
    now = utc_now()
    reason = " ; ".join(validation_errors[:8]) if validation_errors else "Plan bị từ chối bởi Python validator."
    safe_direction = (direction or "REJECTED").upper()
    if safe_direction not in ("LONG", "SHORT"):
        safe_direction = "REJECTED"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, chat_id, symbol, mode, created_at, check_after_hours, entry_wait_hours, max_hold_hours,
                 next_check_at, direction, entry_low, entry_high, sl, tp1, tp2,
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response,
                 result, result_reason, result_checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?,
                    'REJECTED_PLAN', ?, ?, ?, ?, 'REJECTED_PLAN', ?, ?)
            """,
            (user_id, chat_id, symbol, mode, iso(now),
             ENTRY_WAIT_HOURS.get(mode, 24), ENTRY_WAIT_HOURS.get(mode, 24), TRADE_MAX_HOLD_HOURS.get(mode, 72),
             safe_direction, entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response, reason, iso(now)),
        )
        conn.commit()
        return cursor.lastrowid


def _row_to_pred(row) -> dict:
    keys = [
        "id", "user_id", "chat_id", "symbol", "mode", "created_at",
        "entry_wait_hours", "max_hold_hours", "next_check_at", "direction",
        "entry_low", "entry_high", "sl", "tp1", "tp2", "entry_status",
        "entry_filled_at", "entry_price", "result"
    ]
    return dict(zip(keys, row))


def get_due_predictions() -> list[dict]:
    now_s = iso(utc_now())
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, chat_id, symbol, mode, created_at,
                   entry_wait_hours, max_hold_hours, next_check_at, direction,
                   entry_low, entry_high, sl, tp1, tp2, entry_status,
                   entry_filled_at, entry_price, result
            FROM predictions
            WHERE result IN ('PENDING_ENTRY', 'ENTRY_FILLED')
              AND (next_check_at IS NULL OR next_check_at <= ?)
            ORDER BY id ASC
            LIMIT 100
            """,
            (now_s,),
        ).fetchall()
    return [_row_to_pred(row) for row in rows]


def schedule_next_check(pid: int, mode: str) -> None:
    next_at = utc_now() + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE predictions SET next_check_at=?, result_checked_at=? WHERE id=?",
            (iso(next_at), iso(utc_now()), pid),
        )
        conn.commit()


def mark_entry_filled(pid: int, entry_price: float, filled_at: datetime, mode: str) -> None:
    next_at = utc_now() + timedelta(hours=CHECK_INTERVAL_HOURS.get(mode, 1))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET result='ENTRY_FILLED', entry_status='ENTRY_FILLED', entry_price=?,
                entry_filled_at=?, next_check_at=?, result_checked_at=?
            WHERE id=?
            """,
            (entry_price, iso(filled_at), iso(next_at), iso(utc_now()), pid),
        )
        conn.commit()


def _calc_rr(direction: str, entry_price: float | None, sl: float | None, outcome_price: float | None, result: str) -> float | None:
    if entry_price is None or sl is None or outcome_price is None:
        return None
    risk = abs(entry_price - sl)
    if risk <= 0:
        return None
    if result == "LOSS":
        return -1.0
    if direction == "LONG":
        return (outcome_price - entry_price) / risk
    if direction == "SHORT":
        return (entry_price - outcome_price) / risk
    return None


def update_prediction_result(
    pid: int,
    result: str,
    result_price: float,
    result_reason: str | None = None,
    trade_closed_at: datetime | None = None,
    entry_price: float | None = None,
    direction: str | None = None,
    sl: float | None = None,
    entry_filled_at: datetime | None = None,
) -> None:
    now = utc_now()
    closed = trade_closed_at or now
    hold_hours = None
    if entry_filled_at is not None:
        hold_hours = max(0.0, (closed - entry_filled_at).total_seconds() / 3600)
    rr_result = _calc_rr(direction or "", entry_price, sl, result_price, result)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE predictions
            SET result=?, result_price=?, result_reason=?, result_checked_at=?,
                trade_closed_at=?, hold_hours=?, rr_result=?, next_check_at=NULL
            WHERE id=?
            """,
            (result, result_price, result_reason, iso(now), iso(closed), hold_hours, rr_result, pid),
        )
        conn.commit()


def get_recent_predictions(
    symbol: str,
    mode: str,
    user_id: int | None = None,
    limit: int = PREDICTION_HISTORY_COUNT,
) -> list[dict]:
    """
    Lấy lịch sử dùng cho Claude học lại.

    Quy tắc privacy/per-user learning:
    - Khi phân tích cho user nào, Claude chỉ nhận lịch sử của chính user đó.
    - Không dùng lịch sử global của user khác để tránh nhiễu chiến lược và tránh lộ dữ liệu.
    - Nếu user_id=None (ví dụ gọi legacy/manual), không đưa lịch sử học lại.
    """
    if user_id is None:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT created_at, direction, entry_low, entry_high, sl, tp1, tp2,
                   reasoning_summary, full_response, result, result_price, result_reason,
                   market_snapshot, feature_snapshot
            FROM predictions
            WHERE symbol=? AND mode=? AND user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, mode, user_id, limit),
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
            "feature_snapshot":  row[13],
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


def parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_vn_datetime(value: str | datetime | None) -> str:
    if not value:
        return "-"
    dt = value if isinstance(value, datetime) else parse_utc_datetime(value)
    if dt is None:
        return "-"
    local = dt.astimezone(VN_TZ)
    return local.strftime("%H:%M ngày %d/%m/%Y")


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
        print(f"Historical Binance error {symbol} {interval}: {exc}", flush=True)
        return None


def _entry_touched(direction: str, entry_low: float | None, entry_high: float | None, high: float, low: float) -> bool:
    if entry_low is None or entry_high is None:
        return False
    return low <= entry_high and high >= entry_low


def _entry_price(direction: str, entry_low: float | None, entry_high: float | None) -> float | None:
    if entry_low is None or entry_high is None:
        return None
    return (entry_low + entry_high) / 2


def _tp_sl_result(pred: dict, candles: pd.DataFrame) -> tuple[str, float | None, str, datetime | None]:
    direction, sl, tp1 = pred["direction"], pred["sl"], pred["tp1"]
    candle_label = "15M" if pred.get("mode") == "short" else "1H"
    if not sl or not tp1:
        return "UNKNOWN", None, "Thiếu SL hoặc TP1 nên không thể chấm kết quả.", None
    for _, row in candles.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        closed_at_ts = row["close_time"]
        closed_at = closed_at_ts.to_pydatetime() if hasattr(closed_at_ts, "to_pydatetime") else None
        text_time = str(row["close_time"])[:16]

        if direction == "LONG":
            hit_tp = high >= tp1
            hit_sl = low <= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 và SL cùng bị chạm trong một nến {candle_label} lúc {text_time}.", closed_at
            if hit_tp:
                return "WIN", tp1, f"TP1 chạm trước SL lúc {text_time}.", closed_at
            if hit_sl:
                return "LOSS", sl, f"SL chạm trước TP1 lúc {text_time}.", closed_at
        elif direction == "SHORT":
            hit_tp = low <= tp1
            hit_sl = high >= sl
            if hit_tp and hit_sl:
                return "AMBIGUOUS", close, f"TP1 và SL cùng bị chạm trong một nến {candle_label} lúc {text_time}.", closed_at
            if hit_tp:
                return "WIN", tp1, f"TP1 chạm trước SL lúc {text_time}.", closed_at
            if hit_sl:
                return "LOSS", sl, f"SL chạm trước TP1 lúc {text_time}.", closed_at
    return "RUNNING", float(candles.iloc[-1]["close"]), "Đã khớp Entry nhưng chưa chạm TP1 hoặc SL.", None


def evaluate_prediction_lifecycle(pred: dict, candles: pd.DataFrame | None) -> dict:
    now = utc_now()
    created = parse_utc_datetime(pred.get("created_at"))
    entry_filled_at = parse_utc_datetime(pred.get("entry_filled_at"))
    if created is None:
        return {"action": "skip", "reason": "Không đọc được thời gian tạo prediction."}
    if candles is None or candles.empty:
        return {"action": "reschedule", "reason": "Không có dữ liệu nến."}

    candles = candles[candles["close_time"] >= pd.Timestamp(entry_filled_at or created)]
    if candles.empty:
        return {"action": "reschedule", "reason": "Chưa có nến đóng sau thời điểm cần kiểm tra."}

    status = pred.get("result") or pred.get("entry_status") or "PENDING_ENTRY"

    if status == "PENDING_ENTRY":
        entry_deadline = created + timedelta(hours=int(pred.get("entry_wait_hours") or 24))
        for _, row in candles.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if _entry_touched(pred["direction"], pred.get("entry_low"), pred.get("entry_high"), high, low):
                filled_at_ts = row["close_time"]
                filled_at = filled_at_ts.to_pydatetime() if hasattr(filled_at_ts, "to_pydatetime") else now
                entry_price = _entry_price(pred["direction"], pred.get("entry_low"), pred.get("entry_high"))
                post = candles[candles["close_time"] >= row["close_time"]]
                result, price, reason, closed_at = _tp_sl_result({**pred, "entry_price": entry_price}, post)
                if result in ("WIN", "LOSS", "AMBIGUOUS"):
                    return {"action": "close", "result": result, "price": price, "reason": f"Entry khớp rồi {reason}", "closed_at": closed_at or filled_at, "entry_price": entry_price, "entry_filled_at": filled_at}
                return {"action": "fill", "price": entry_price, "filled_at": filled_at, "reason": f"Entry đã khớp lúc {str(row['close_time'])[:16]}."}
        if now >= entry_deadline:
            last_price = float(candles.iloc[-1]["close"])
            return {"action": "close", "result": "NOT_FILLED", "price": last_price, "reason": f"Hết thời gian chờ Entry {pred.get('entry_wait_hours')}h nhưng giá chưa chạm vùng Entry.", "closed_at": now}
        return {"action": "reschedule", "reason": "Chưa chạm Entry, tiếp tục chờ."}

    if status == "ENTRY_FILLED":
        if entry_filled_at is None:
            return {"action": "reschedule", "reason": "Thiếu entry_filled_at."}
        result, price, reason, closed_at = _tp_sl_result(pred, candles)
        if result in ("WIN", "LOSS", "AMBIGUOUS"):
            return {"action": "close", "result": result, "price": price, "reason": reason, "closed_at": closed_at or now, "entry_price": pred.get("entry_price"), "entry_filled_at": entry_filled_at}
        hold_deadline = entry_filled_at + timedelta(hours=int(pred.get("max_hold_hours") or 72))
        if now >= hold_deadline:
            return {"action": "close", "result": "EXPIRED", "price": price, "reason": f"Đã khớp Entry nhưng quá thời gian giữ lệnh {pred.get('max_hold_hours')}h mà chưa chạm TP1/SL.", "closed_at": now, "entry_price": pred.get("entry_price"), "entry_filled_at": entry_filled_at}
        return {"action": "reschedule", "reason": reason}

    return {"action": "skip", "reason": f"Trạng thái {status} không cần kiểm tra."}


def format_prediction_result_message(pred: dict, result: str, price: float | None, reason: str, entry_price: float | None = None, hold_hours: float | None = None) -> str:
    emoji = {"WIN": "✅", "LOSS": "❌", "NOT_FILLED": "⏱", "EXPIRED": "⌛", "AMBIGUOUS": "⚠️"}.get(result, "📌")
    mode_label = "SCALP" if pred.get("mode") == "short" else "SWING"
    lines = [
        "🎯 Kết quả tự động kiểm tra dự đoán",
        "",
        f"{emoji} {pred.get('symbol')} — {mode_label} — {result}",
        f"Thời gian phân tích: {format_vn_datetime(pred.get('created_at'))}",
        f"Hướng: {pred.get('direction')}",
        f"Entry: {fmt(pred.get('entry_low'))}–{fmt(pred.get('entry_high'))}",
        f"SL: {fmt(pred.get('sl'))}",
        f"TP1: {fmt(pred.get('tp1'))}",
        f"TP2: {fmt(pred.get('tp2'))}",
    ]
    if entry_price is not None:
        lines.append(f"Giá khớp Entry: {fmt(entry_price)}")
    if price is not None:
        lines.append(f"Giá check: {fmt(price)}")
    if hold_hours is not None:
        lines.append(f"Thời gian giữ lệnh: {hold_hours:.1f}h")
    lines.extend([
        f"Lý do: {reason}",
        f"ID: #{pred.get('id')}",
    ])
    return "\n".join(lines)


async def auto_check_pending_predictions() -> dict:
    """Check predictions đến hạn, chỉ cập nhật DB và trả về số liệu tóm tắt.

    Hàm này cố ý không tạo notification để gửi cho user/admin nữa.
    User muốn xem kết quả thì chủ động dùng /history, /stats hoặc /dashboard.
    """
    init_prediction_db()
    due = get_due_predictions()
    entry_filled_count = 0
    closed_count = 0
    rescheduled_count = 0
    skipped_count = 0

    print(f"[AUTO_CHECK] Due predictions: {len(due)} at {iso(utc_now())}", flush=True)

    for pred in due:
        start_dt = parse_utc_datetime(pred.get("entry_filled_at")) or parse_utc_datetime(pred.get("created_at"))
        if start_dt is None:
            skipped_count += 1
            continue
        result_interval = get_result_check_interval(pred.get("mode", "short"))
        candles = await asyncio.to_thread(get_binance_klines_since, pred["symbol"], result_interval, start_dt)
        decision = evaluate_prediction_lifecycle(pred, candles)
        action = decision.get("action")

        if action == "fill":
            mark_entry_filled(pred["id"], decision["price"], decision["filled_at"], pred["mode"])
            entry_filled_count += 1
            # Không gửi tin khi khớp Entry; chỉ log Railway và lưu DB.
            print(f"[AUTO_CHECK] #{pred['id']} ENTRY_FILLED {pred['symbol']} {decision.get('reason')}", flush=True)
            continue

        if action == "close":
            result = decision["result"]
            price = decision.get("price")
            if price is None:
                price = await asyncio.to_thread(get_current_price_raw, pred["symbol"])
            if price is None:
                schedule_next_check(pred["id"], pred["mode"])
                rescheduled_count += 1
                continue
            entry_price = decision.get("entry_price") or pred.get("entry_price")
            entry_filled_at = decision.get("entry_filled_at") or parse_utc_datetime(pred.get("entry_filled_at"))
            update_prediction_result(
                pred["id"], result, float(price), decision.get("reason"),
                trade_closed_at=decision.get("closed_at"), entry_price=entry_price,
                direction=pred.get("direction"), sl=pred.get("sl"), entry_filled_at=entry_filled_at,
            )
            closed_count += 1
            print(
                f"[AUTO_CHECK] #{pred['id']} CLOSED {pred['symbol']} {result} "
                f"price={price} reason={decision.get('reason')}",
                flush=True,
            )
            continue

        if action == "reschedule":
            schedule_next_check(pred["id"], pred["mode"])
            rescheduled_count += 1
            continue

        skipped_count += 1

    return {
        "due_count": len(due),
        "entry_filled_count": entry_filled_count,
        "closed_count": closed_count,
        "rescheduled_count": rescheduled_count,
        "skipped_count": skipped_count,
        # Giữ key cũ để code cũ không crash nếu còn tham chiếu, nhưng luôn để rỗng.
        "admin_messages": [],
        "user_messages": [],
    }


# ─── Stats / History helpers ─────────────────────────────────────────────────

def build_prediction_where(
    symbol: str | None = None,
    user_id: int | None = None,
    include_rejected: bool = False,
) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if symbol:
        normalized_symbol = symbol.upper() if symbol.upper().endswith("USDT") else f"{symbol.upper()}USDT"
        clauses.append("symbol=?")
        params.append(normalized_symbol)
    if user_id is not None:
        clauses.append("user_id=?")
        params.append(user_id)
    # REJECTED_PLAN là bản ghi học nội bộ, không hiển thị trong /history, /stats, /dashboard
    # để user/admin không nhầm nó là một tín hiệu thật. get_recent_predictions() vẫn đọc được
    # các bản ghi này để Claude học từ lỗi validator.
    if not include_rejected:
        clauses.append("result!='REJECTED_PLAN'")
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def format_scope_label(symbol: str | None = None, user_id: int | None = None) -> str:
    symbol_label = symbol.upper() if symbol else None
    if user_id is None:
        return f"{symbol_label}" if symbol_label else "Teopard"
    return f"của bạn - {symbol_label}" if symbol_label else "của bạn"


def format_stats(symbol: str | None = None, user_id: int | None = None) -> str:
    init_prediction_db()
    where, params = build_prediction_where(symbol=symbol, user_id=user_id)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT result, direction, mode, rr_result FROM predictions {where}",
            params,
        ).fetchall()
    if not rows:
        return "Chưa có lịch sử dự đoán."
    total = len(rows)
    counts = {}
    for result, *_ in rows:
        counts[result] = counts.get(result, 0) + 1
    closed = [r for r in rows if r[0] in ("WIN", "LOSS")]
    wins = sum(1 for r in closed if r[0] == "WIN")
    losses = sum(1 for r in closed if r[0] == "LOSS")
    win_rate = wins / len(closed) * 100 if closed else 0
    rr_values = [r[3] for r in rows if r[3] is not None]
    avg_rr = sum(rr_values) / len(rr_values) if rr_values else 0
    title = f"📊 Thống kê {format_scope_label(symbol, user_id)}"
    return "\n".join([
        title,
        f"Tổng prediction: {total}",
        f"WIN/LOSS: {wins}/{losses} | Win rate: {win_rate:.1f}%",
        f"PENDING_ENTRY: {counts.get('PENDING_ENTRY', 0)}",
        f"ENTRY_FILLED: {counts.get('ENTRY_FILLED', 0)}",
        f"NOT_FILLED: {counts.get('NOT_FILLED', 0)}",
        f"EXPIRED: {counts.get('EXPIRED', 0)}",
        f"AMBIGUOUS: {counts.get('AMBIGUOUS', 0)}",
        f"RR trung bình: {avg_rr:.2f}R" if rr_values else "RR trung bình: chưa có dữ liệu",
    ])


def format_history(symbol: str | None = None, limit: int = 10, user_id: int | None = None) -> str:
    init_prediction_db()
    where, params = build_prediction_where(symbol=symbol, user_id=user_id)
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, direction, entry_low, entry_high, sl, tp1, tp2,
                   result, result_price, created_at, result_reason
            FROM predictions
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    if not rows:
        return "Chưa có lịch sử dự đoán."

    # user_id=None chỉ được dùng cho admin, nên admin sẽ thấy lệnh thuộc user nào.
    is_admin_scope = user_id is None
    lines = [f"🧾 10 dự đoán gần nhất {format_scope_label(symbol, user_id)}"]
    for row in rows:
        pid, owner_user_id, owner_chat_id, sym, mode, direction, entry_low, entry_high, sl, tp1, tp2, result, result_price, created_at, result_reason = row
        mode_label = "SCALP" if mode == "short" else "SWING"
        created_label = format_vn_datetime(created_at) if created_at else "không rõ"
        owner_line = ""
        if is_admin_scope:
            owner_label = str(owner_user_id) if owner_user_id is not None else "không rõ"
            chat_label = str(owner_chat_id) if owner_chat_id is not None else "không rõ"
            owner_line = f"User ID: {owner_label} | Chat ID: {chat_label}\n"
        reason_line = ""
        if result == "REJECTED_PLAN" and result_reason:
            short_reason = str(result_reason)[:260] + ("..." if len(str(result_reason)) > 260 else "")
            reason_line = f"\nLý do không auto-check: {short_reason}"
        lines.append(
            f"#{pid} {sym} {mode_label} {direction} → {result}\n"
            f"{owner_line}"
            f"Thời gian phân tích: {created_label}\n"
            f"Entry {fmt(entry_low)}–{fmt(entry_high)} | SL {fmt(sl)} | TP1 {fmt(tp1)} | TP2 {fmt(tp2)}"
            + (f" | Giá check {fmt(result_price)}" if result_price else "")
            + reason_line
        )
    return "\n\n".join(lines)


def clear_prediction_history() -> int:
    init_prediction_db()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM predictions")
        count = int(cur.fetchone()[0])
        conn.execute("DELETE FROM predictions")
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        except sqlite3.Error:
            pass
        conn.commit()
    return count


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
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


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
    r["atr_14"] = calculate_atr(r, 14)
    r["atr_pct"] = (r["atr_14"] / r["close"]) * 100
    r["vol_ma20"]  = r["volume"].rolling(20).mean()
    r["vol_ratio"] = r["volume"] / r["vol_ma20"]
    return r.dropna().reset_index(drop=True)


# ─── Feature engineering: ATR / Structure / Fibonacci / Liquidity ────────────

def _safe_float(v, default: float | None = None) -> float | None:
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _last_close_from_data(timeframe_data: dict[str, pd.DataFrame | None]) -> float | None:
    for df in timeframe_data.values():
        if df is not None and not df.empty:
            return _safe_float(df.iloc[-1]["close"])
    return None


def _current_atr(df: pd.DataFrame | None) -> float | None:
    if df is None or df.empty or "atr_14" not in df.columns:
        return None
    return _safe_float(df.iloc[-1].get("atr_14"))


def _find_pivots(df: pd.DataFrame | None, side: str, lookback: int = 100, left: int = 2, right: int = 2) -> list[dict]:
    if df is None or df.empty:
        return []
    data = df.tail(lookback).reset_index(drop=True)
    col = "high" if side == "high" else "low"
    pivots: list[dict] = []
    if len(data) < left + right + 1:
        return pivots
    for i in range(left, len(data) - right):
        val = float(data.loc[i, col])
        window = data.loc[i - left:i + right, col]
        if side == "high" and val >= float(window.max()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"]})
        elif side == "low" and val <= float(window.min()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"]})
    return pivots


def _cluster_zone(prices: list[float], current_price: float, side: str, atr: float | None) -> tuple[float | None, float | None, int]:
    if not prices:
        return None, None, 0
    tol = max((atr or 0) * 0.25, current_price * 0.0012)
    buf = max((atr or 0) * 0.20, current_price * 0.0008)
    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []
    cur = [sorted_prices[0]]
    for price in sorted_prices[1:]:
        if abs(price - sum(cur) / len(cur)) <= tol:
            cur.append(price)
        else:
            clusters.append(cur)
            cur = [price]
    clusters.append(cur)

    if side == "low":
        candidates = [c for c in clusters if sum(c) / len(c) <= current_price]
        candidates.sort(key=lambda c: (len(c), sum(c) / len(c)), reverse=True)
        candidates.sort(key=lambda c: abs(current_price - sum(c) / len(c)))
    else:
        candidates = [c for c in clusters if sum(c) / len(c) >= current_price]
        candidates.sort(key=lambda c: (len(c), -sum(c) / len(c)), reverse=True)
        candidates.sort(key=lambda c: abs(sum(c) / len(c) - current_price))

    if not candidates:
        return None, None, 0
    best = candidates[0]
    low = min(best) - buf
    high = max(best) + buf
    return low, high, len(best)


def _fallback_zone(df: pd.DataFrame | None, side: str, current_price: float, atr: float | None, window: int = 80) -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None
    data = df.tail(window)
    buf = max((atr or 0) * 0.20, current_price * 0.0008)
    if side == "low":
        price = float(data["low"].min())
    else:
        price = float(data["high"].max())
    return price - buf, price + buf


def _liquidity_zones(df: pd.DataFrame | None, current_price: float, atr: float | None) -> dict:
    low_pivots = [p["price"] for p in _find_pivots(df, "low", 100)]
    high_pivots = [p["price"] for p in _find_pivots(df, "high", 100)]
    long_low, long_high, long_hits = _cluster_zone(low_pivots, current_price, "low", atr)
    short_low, short_high, short_hits = _cluster_zone(high_pivots, current_price, "high", atr)

    if long_low is None or long_high is None:
        long_low, long_high = _fallback_zone(df, "low", current_price, atr, 80)
        long_hits = 1 if long_low is not None else 0
    if short_low is None or short_high is None:
        short_low, short_high = _fallback_zone(df, "high", current_price, atr, 80)
        short_hits = 1 if short_low is not None else 0

    # Deep zones dùng cực trị rộng hơn để tránh SL/TP quá sát vùng gần.
    deep_long_low, deep_long_high = _fallback_zone(df, "low", current_price, atr, 150)
    deep_short_low, deep_short_high = _fallback_zone(df, "high", current_price, atr, 150)
    return {
        "long_near": (long_low, long_high, long_hits),
        "short_near": (short_low, short_high, short_hits),
        "long_deep": (deep_long_low, deep_long_high),
        "short_deep": (deep_short_low, deep_short_high),
    }


def _structure_info(df: pd.DataFrame | None, current_price: float | None) -> dict:
    if df is None or df.empty:
        return {}
    data_recent = df.tail(60)
    data_major = df.tail(120 if len(df) >= 120 else len(df))
    if data_recent.empty or data_major.empty:
        return {}
    recent_high = float(data_recent["high"].max())
    recent_low = float(data_recent["low"].min())
    major_high = float(data_major["high"].max())
    major_low = float(data_major["low"].min())
    swing_low, swing_high = recent_low, recent_high
    span = max(swing_high - swing_low, 0.0)
    fibs = {}
    if span > 0:
        fibs = {
            "0.382": swing_low + span * 0.382,
            "0.5": swing_low + span * 0.5,
            "0.618": swing_low + span * 0.618,
        }
    first_close = float(data_recent.iloc[0]["close"])
    last_close = float(data_recent.iloc[-1]["close"])
    if last_close > first_close * 1.003:
        trend = "TĂNG"
    elif last_close < first_close * 0.997:
        trend = "GIẢM"
    else:
        trend = "ĐI NGANG"
    pivot_highs = _find_pivots(df, "high", 80)
    pivot_lows = _find_pivots(df, "low", 80)
    recent_pivot_high = pivot_highs[-1]["price"] if pivot_highs else recent_high
    recent_pivot_low = pivot_lows[-1]["price"] if pivot_lows else recent_low
    return {
        "trend": trend,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "major_high": major_high,
        "major_low": major_low,
        "recent_pivot_high": recent_pivot_high,
        "recent_pivot_low": recent_pivot_low,
        "fib": fibs,
    }


def _consecutive_candles(df: pd.DataFrame | None) -> str:
    if df is None or len(df) < 2:
        return "Không đủ dữ liệu"
    count = 0
    last_dir = None
    for _, row in df.tail(12).iloc[::-1].iterrows():
        direction = "xanh" if float(row["close"]) > float(row["open"]) else "đỏ" if float(row["close"]) < float(row["open"]) else "doji"
        if last_dir is None:
            last_dir = direction
            count = 1
        elif direction == last_dir:
            count += 1
        else:
            break
    return f"{count} nến {last_dir} liên tiếp"


def _wick_body_info(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "Không đủ dữ liệu"
    row = df.iloc[-1]
    high, low, open_, close = map(float, [row["high"], row["low"], row["open"], row["close"]])
    rng = max(high - low, 1e-12)
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return f"thân nến {body / rng * 100:.0f}%, râu trên {upper / rng * 100:.0f}%, râu dưới {lower / rng * 100:.0f}%"


def _mode_labels(mode: str) -> tuple[str, str, str]:
    if mode == "short":
        return "15M", "1H", "4H"
    return "4H", "1D", "1W"


def _risk_floor(timeframe_data: dict[str, pd.DataFrame | None], mode: str, current_price: float) -> float:
    if mode == "short":
        atr_main = _current_atr(timeframe_data.get("15M")) or 0
        atr_confirm = _current_atr(timeframe_data.get("1H")) or 0
        return max(atr_main * 2.5, atr_confirm * 1.2, current_price * 0.006)
    atr_main = _current_atr(timeframe_data.get("4H")) or 0
    atr_confirm = _current_atr(timeframe_data.get("1D")) or 0
    return max(atr_main * 2.2, atr_confirm * 0.7, current_price * 0.025)


def _reference_plans(current_price: float, zones: dict, risk: float) -> dict:
    long_low, long_high, _ = zones.get("long_near", (None, None, 0))
    short_low, short_high, _ = zones.get("short_near", (None, None, 0))
    plans = {}
    if long_low is not None and long_high is not None:
        entry_mid = (long_low + long_high) / 2
        sl = min(entry_mid - risk, (zones.get("long_deep", (None, None))[0] or entry_mid - risk))
        real_risk = max(entry_mid - sl, risk)
        plans["LONG"] = {
            "entry_low": long_low, "entry_high": long_high, "sl": sl,
            "tp1": entry_mid + real_risk, "tp2": entry_mid + real_risk * 2,
            "risk": real_risk,
        }
    if short_low is not None and short_high is not None:
        entry_mid = (short_low + short_high) / 2
        sl = max(entry_mid + risk, (zones.get("short_deep", (None, None))[1] or entry_mid + risk))
        real_risk = max(sl - entry_mid, risk)
        plans["SHORT"] = {
            "entry_low": short_low, "entry_high": short_high, "sl": sl,
            "tp1": entry_mid - real_risk, "tp2": entry_mid - real_risk * 2,
            "risk": real_risk,
        }
    return plans


def build_feature_engineering_block(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    main_label, structure_label, big_label = _mode_labels(mode)
    price = current_price or _last_close_from_data(timeframe_data)
    if price is None:
        return "FEATURE_ENGINEERING: Không đủ dữ liệu để tính cấu trúc, Fibonacci, ATR và vùng quét. Không được tự bịa các phần này."

    main_df = timeframe_data.get(main_label)
    structure_df = timeframe_data.get(structure_label)
    if structure_df is None or structure_df.empty:
        structure_df = main_df
    atr_main = _current_atr(main_df)
    atr_structure = _current_atr(structure_df)
    zones = _liquidity_zones(structure_df, price, atr_structure or atr_main)
    structure = _structure_info(structure_df, price)
    risk = _risk_floor(timeframe_data, mode, price)

    long_near = zones.get("long_near", (None, None, 0))
    short_near = zones.get("short_near", (None, None, 0))
    long_deep = zones.get("long_deep", (None, None))
    short_deep = zones.get("short_deep", (None, None))
    fib = structure.get("fib", {})

    lines = [
        "FEATURE_ENGINEERING_DO_PYTHON_TINH_SAN:",
        f"- Mode: {'SCALP' if mode == 'short' else 'SWING'} | Khung vào lệnh: {main_label} | Khung cấu trúc: {structure_label} | Khung lớn: {big_label}",
        f"- ATR14 {main_label}: {fmt(atr_main)} | ATR14 {structure_label}: {fmt(atr_structure)} | Rủi ro tối thiểu đề xuất: {fmt(risk)} USDT",
        f"- Chuỗi nến {main_label}: {_consecutive_candles(main_df)} | Nến cuối: {_wick_body_info(main_df)}",
        f"- Cấu trúc {structure_label}: {structure.get('trend', 'N/A')}; swing gần {fmt(structure.get('recent_low'))}–{fmt(structure.get('recent_high'))}; swing lớn {fmt(structure.get('major_low'))}–{fmt(structure.get('major_high'))}",
        f"- Fibonacci {structure_label}: 0.382={fmt(fib.get('0.382'))}; 0.5={fmt(fib.get('0.5'))}; 0.618={fmt(fib.get('0.618'))}",
        f"- Vùng quét Long gần: {fmt(long_near[0])}–{fmt(long_near[1])} (cụm {long_near[2]} điểm); sâu: {fmt(long_deep[0])}–{fmt(long_deep[1])}",
        f"- Vùng quét Short gần: {fmt(short_near[0])}–{fmt(short_near[1])} (cụm {short_near[2]} điểm); sâu: {fmt(short_deep[0])}–{fmt(short_deep[1])}",
        "- Quy tắc rủi ro: Claude tự lập Entry/SL/TP, nhưng khoảng cách Entry–SL nên không nhỏ hơn rủi ro tối thiểu đề xuất; TP1 nên khoảng >= 0.8R, TP2 nên khoảng >= 1.4R.",
        "- Ghi chú: Vùng quét chỉ là ước lượng từ pivot/equal high/equal low và high/low nến, không phải dữ liệu thanh lý thật. Block này là bản đồ kỹ thuật, không phải lệnh giao dịch chốt sẵn.",
    ]
    return "\n".join(lines)


def build_feature_snapshot(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    """
    Snapshot kỹ thuật ngắn gọn để lưu vào DB và đưa vào history learning.

    Khác với feature_block đầy đủ cho lần phân tích hiện tại, snapshot này chỉ giữ
    các feature quan trọng nhất tại thời điểm ra lệnh để Claude học lại vì sao
    lệnh cũ WIN/LOSS trong bối cảnh nào.
    """
    main_label, structure_label, big_label = _mode_labels(mode)
    price = current_price or _last_close_from_data(timeframe_data)
    if price is None:
        return "Feature snapshot: không đủ dữ liệu."

    main_df = timeframe_data.get(main_label)
    structure_df = timeframe_data.get(structure_label)
    if structure_df is None or structure_df.empty:
        structure_df = main_df
    big_df = timeframe_data.get(big_label)

    atr_main = _current_atr(main_df)
    atr_structure = _current_atr(structure_df)
    zones = _liquidity_zones(structure_df, price, atr_structure or atr_main)
    structure = _structure_info(structure_df, price)
    risk = _risk_floor(timeframe_data, mode, price)
    fib = structure.get("fib", {})

    long_near = zones.get("long_near", (None, None, 0))
    short_near = zones.get("short_near", (None, None, 0))
    long_deep = zones.get("long_deep", (None, None))
    short_deep = zones.get("short_deep", (None, None))

    def compact_tf(label: str, df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return f"{label}: N/A"
        last = df.iloc[-1]
        if last["ema_7"] > last["ema_25"] > last["ema_50"]:
            ema = "EMA tăng"
        elif last["ema_7"] < last["ema_25"] < last["ema_50"]:
            ema = "EMA giảm"
        else:
            ema = "EMA đan xen"
        return (
            f"{label}: close {fmt(last['close'])}, {ema}, "
            f"RSI14 {fmt(last['rsi_14'], 1)}, MACD_hist {fmt(last['macd_hist'], 4)}, "
            f"ATR14 {fmt(last.get('atr_14'))}, vol {fmt(last['vol_ratio'], 2)}x"
        )

    parts = [
        f"Mode {'SCALP' if mode == 'short' else 'SWING'}; frame entry {main_label}, structure {structure_label}, big {big_label}",
        compact_tf(main_label, main_df),
        compact_tf(structure_label, structure_df),
        compact_tf(big_label, big_df),
        f"Cấu trúc {structure_label}: {structure.get('trend', 'N/A')}; swing gần {fmt(structure.get('recent_low'))}-{fmt(structure.get('recent_high'))}; swing lớn {fmt(structure.get('major_low'))}-{fmt(structure.get('major_high'))}",
        f"Fib {structure_label}: 0.382 {fmt(fib.get('0.382'))}, 0.5 {fmt(fib.get('0.5'))}, 0.618 {fmt(fib.get('0.618'))}",
        f"Liquidity Long gần {fmt(long_near[0])}-{fmt(long_near[1])} / sâu {fmt(long_deep[0])}-{fmt(long_deep[1])}; Short gần {fmt(short_near[0])}-{fmt(short_near[1])} / sâu {fmt(short_deep[0])}-{fmt(short_deep[1])}",
        f"ATR/risk: ATR {main_label} {fmt(atr_main)}, ATR {structure_label} {fmt(atr_structure)}, risk_floor {fmt(risk)}",
        f"Nến {main_label}: {_consecutive_candles(main_df)}; {_wick_body_info(main_df)}",
    ]
    return " | ".join(parts)


# ─── Format helpers ───────────────────────────────────────────────────────────

def fmt(v, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 100:
        return f"{v:,.{decimals}f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    return f"{v:,.8f}"


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
        f"  ATR14={fmt(last.get('atr_14'))} ({fmt(last.get('atr_pct'),2)}%)",
        f"  Volume={fmt(last['vol_ratio'],2)}x → {vol_lbl}",
        f"  Nến hiện tại: {_consecutive_candles(df)} | {_wick_body_info(df)}",
        f"  High/Low 50 nến: {fmt(key_high)} / {fmt(key_low)}",
        f"  10 nến gần nhất:",
        candles,
    ])


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
            f"ATR14={fmt(last.get('atr_14'))}, vol={fmt(last['vol_ratio'], 2)}x"
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

    lines = [f"USER-SPECIFIC RECENT LEARNING SUMMARY ({len(history)} latest analyses from this user only):"]
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
        feature_snapshot = p.get("feature_snapshot") or "No feature snapshot."
        lines.append(
            f"- #{i} {p['created_at'][:16]} {p['direction']} {p['result']} ({checked}); "
            f"Entry {entry}, SL {fmt(p['sl'])}, TP1 {fmt(p['tp1'])}, TP2 {fmt(p['tp2'])}. "
            f"Decision why: {decision_reason} Outcome: {reason} "
            f"Market then: {snapshot} Feature then: {feature_snapshot}"
        )

    lines.append("Use this user-specific summary as learning context; do not copy old full responses and do not assume global user behavior.")
    return "\n".join(lines)


def build_user_prompt(
    symbol: str,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_str: str,
    history: list[dict],
    feature_block: str | None = None,
) -> str:
    mode_label = "SCALP (ngắn hạn)" if mode == "short" else "SWING (dài hạn)"
    focus      = (
        "Dùng 15M để timing entry, 1H để xác nhận momentum, 4H để xác định xu hướng chính."
        if mode == "short" else
        "Dùng 4H để timing entry, 1D để xác nhận xu hướng, 1W để xác định big picture."
    )

    history_block = format_prediction_history(history)
    tf_blocks     = "".join(summarize_timeframe(lbl, df) for lbl, df in timeframe_data.items())
    feature_block = feature_block or build_feature_engineering_block(timeframe_data, mode, None)

    return f"""YÊU CẦU PHÂN TÍCH {mode_label} CHO {symbol}

{current_price_str}
{fear_greed_info}
Phương pháp: {focus}

═══════════════════════════════
{feature_block}
═══════════════════════════════
{history_block}
═══════════════════════════════
{tf_blocks}
═══════════════════════════════

Yêu cầu:
1. Python chỉ cung cấp dữ liệu cứng: EMA/RSI/MACD/ATR, cấu trúc, Fibonacci, vùng quét, rủi ro tối thiểu. Không có kế hoạch LONG/SHORT chốt sẵn.
2. Claude phải tự phân tích và tự lập Entry/SL/TP dựa trên dữ liệu cứng đó. Không được tự tạo thêm Fibonacci/vùng quét nếu block Python ghi N/A hoặc không đủ dữ liệu.
3. Entry/SL/TP phải hợp logic với hướng giao dịch và tôn trọng rủi ro tối thiểu đề xuất theo ATR/giá.
4. Đọc kỹ RECENT LEARNING SUMMARY, đặc biệt Decision why, Outcome và Market then, nhưng không hiện mục “Nhìn lại lịch sử” trong câu trả lời.
5. Không copy phân tích cũ. Chỉ dùng summary để tránh lặp lại lỗi.
6. QUYẾT ĐỊNH cuối cùng bắt buộc là LONG hoặc SHORT, không được chỉ CHỜ.
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


# ─── Hybrid AI validator ─────────────────────────────────────────────────────

def validate_prediction_plan(
    pred: dict,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    current_price: float | None,
) -> list[str]:
    """
    Python không chọn lệnh thay Claude. Hàm này chỉ đóng vai trò kiểm tra an toàn:
    - LONG/SHORT đúng chiều Entry/SL/TP.
    - Có đủ số để bot auto-check.
    - SL không quá sát Entry so với rủi ro tối thiểu Python đã tính.
    - TP1/TP2 có tỷ lệ lợi nhuận/rủi ro tối thiểu.
    """
    errors: list[str] = []
    direction = (pred.get("direction") or "").upper()
    entry_low = pred.get("entry_low")
    entry_high = pred.get("entry_high")
    sl = pred.get("sl")
    tp1 = pred.get("tp1")
    tp2 = pred.get("tp2")

    if direction not in ("LONG", "SHORT"):
        errors.append("QUYẾT ĐỊNH phải là LONG hoặc SHORT.")
        return errors

    required = {
        "Entry thấp": entry_low,
        "Entry cao": entry_high,
        "SL": sl,
        "TP1": tp1,
        "TP2": tp2,
    }
    for name, value in required.items():
        if value is None:
            errors.append(f"Thiếu {name} dạng số cụ thể.")
        elif not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0:
            errors.append(f"{name} không hợp lệ: {value}.")
    if errors:
        return errors

    entry_low = float(entry_low)
    entry_high = float(entry_high)
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
        pred["entry_low"], pred["entry_high"] = entry_low, entry_high

    entry_mid = (entry_low + entry_high) / 2
    sl = float(sl)
    tp1 = float(tp1)
    tp2 = float(tp2)

    if direction == "LONG":
        if not sl < entry_low:
            errors.append("LONG phải có SL thấp hơn vùng Entry.")
        if not tp1 > entry_high:
            errors.append("LONG phải có TP1 cao hơn vùng Entry.")
        if not tp2 > tp1:
            errors.append("LONG phải có TP2 cao hơn TP1.")
        risk = entry_mid - sl
        reward1 = tp1 - entry_mid
        reward2 = tp2 - entry_mid
    else:
        if not sl > entry_high:
            errors.append("SHORT phải có SL cao hơn vùng Entry.")
        if not tp1 < entry_low:
            errors.append("SHORT phải có TP1 thấp hơn vùng Entry.")
        if not tp2 < tp1:
            errors.append("SHORT phải có TP2 thấp hơn TP1.")
        risk = sl - entry_mid
        reward1 = entry_mid - tp1
        reward2 = entry_mid - tp2

    if risk <= 0:
        errors.append("Khoảng cách Entry–SL không hợp lệ.")
        return errors

    price = current_price or _last_close_from_data(timeframe_data) or entry_mid
    min_risk = _risk_floor(timeframe_data, mode, float(price))
    # Cho Claude linh hoạt một chút, nhưng không để SL quá sát như trước.
    if min_risk and risk < min_risk * 0.70:
        errors.append(
            f"SL quá sát Entry: rủi ro {fmt(risk)} nhỏ hơn 70% rủi ro tối thiểu đề xuất {fmt(min_risk)}."
        )
    if reward1 <= 0 or reward1 < risk * 0.80:
        errors.append(f"TP1 chưa đủ lợi nhuận so với rủi ro: reward1={fmt(reward1)}, risk={fmt(risk)}.")
    if reward2 <= 0 or reward2 < risk * 1.40:
        errors.append(f"TP2 chưa đủ lợi nhuận mở rộng: reward2={fmt(reward2)}, risk={fmt(risk)}.")

    # Chặn số bịa quá xa giá hiện tại, nhưng không quá chặt để Swing vẫn có không gian.
    max_distance_pct = 0.08 if mode == "short" else 0.20
    if price and abs(entry_mid - float(price)) / float(price) > max_distance_pct:
        errors.append(
            f"Entry quá xa giá hiện tại {fmt(price)} so với mode {'SCALP' if mode == 'short' else 'SWING'}."
        )

    return errors


def build_repair_prompt(bad_output: str, validation_errors: list[str]) -> str:
    errors = "\n".join(f"- {e}" for e in validation_errors)
    return f"""Phản hồi trước chưa đạt kiểm tra logic của Python validator.

Lỗi cần sửa:
{errors}

Yêu cầu sửa:
- Giữ nguyên dữ liệu thị trường trong prompt gốc.
- Không tự tạo Fibonacci/vùng quét/ATR ngoài block Python.
- Tự điều chỉnh lại QUYẾT ĐỊNH, Entry, SL, TP1, TP2 sao cho hợp logic.
- Trả lại TOÀN BỘ phân tích theo đúng format bắt buộc.
- Không giải thích rằng bạn đang sửa lỗi.

Phản hồi trước:
{bad_output}
"""


def request_claude_repair(system_prompt: str, user_prompt: str, bad_output: str, validation_errors: list[str]) -> str:
    """Gọi Claude sửa một lần nếu Entry/SL/TP không qua validator."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": bad_output},
            {"role": "user", "content": build_repair_prompt(bad_output, validation_errors)},
        ],
        timeout=300,
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def maybe_append_not_saved_warning(output: str, errors: list[str]) -> str:
    """
    Nếu Claude vẫn trả plan không hợp lệ sau 1 lần repair, KHÔNG show nguyên plan lỗi cho user.

    Lý do: user có thể hiểu nhầm đó là tín hiệu tham khảo và vào lệnh theo một kế hoạch
    mà Python validator đã đánh rớt. Bot chỉ trả thông báo ngắn, còn bản ghi lỗi được
    lưu nội bộ dưới result=REJECTED_PLAN để Claude học tránh lặp lại.
    """
    if not errors:
        return output
    return (
        "⚠️ Teopard chưa tìm thấy setup hợp lệ để tạo tín hiệu.\n\n"
        "Kế hoạch Entry / SL / TP của AI chưa đạt kiểm tra quản trị rủi ro, "
        "nên bot không hiển thị lệnh này và không đưa vào auto-check.\n\n"
        "Bạn có thể thử phân tích lại sau khi giá thay đổi, hoặc chọn khung Scalp/Swing khác."
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    for p in [Path("analyze_system_prompt.txt"), Path("analysis_system_prompt.txt")]:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Không tìm thấy analyze_system_prompt.txt")


def load_timeframe_data(binance_symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Sync helper: fetch Binance candles then calculate indicators."""
    return add_indicators(get_binance_klines(binance_symbol, interval, limit))


def request_claude_analysis(system_prompt: str, user_prompt: str) -> str:
    """Sync helper: Anthropic SDK is synchronous, so call it via asyncio.to_thread()."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=300,
    )
    return "".join(b.text for b in response.content if hasattr(b, "text"))


def call_claude_analysis(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> str:
    """
    Legacy synchronous entry point.

    Không gọi hàm này trực tiếp trong Telegram async handler, vì bên trong có
    requests.get(), Anthropic SDK sync và SQLite. Handler phải gọi analyze_symbol(),
    hàm đó sẽ đưa các phần blocking sang worker thread bằng asyncio.to_thread().
    """

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env.")

    init_prediction_db()

    binance_symbol = f"{symbol.upper()}USDT"
    configs        = SHORT_TERM_TIMEFRAMES if mode == "short" else LONG_TERM_TIMEFRAMES

    timeframe_data: dict[str, pd.DataFrame | None] = {}
    for label, (interval, limit) in configs.items():
        timeframe_data[label] = load_timeframe_data(binance_symbol, interval, limit)

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        raise RuntimeError(f"Could not fetch Binance data for {binance_symbol}.")

    system_prompt                    = load_system_prompt()
    fear_greed_info                  = get_fear_greed_index()
    current_price_str, current_price = get_current_price_str(binance_symbol)
    feature_block                    = build_feature_engineering_block(timeframe_data, mode, current_price)
    feature_snapshot                 = build_feature_snapshot(timeframe_data, mode, current_price)
    market_snapshot                  = build_market_snapshot(
        timeframe_data,
        fear_greed_info,
        current_price_str,
    )
    history                          = get_recent_predictions(binance_symbol, mode, user_id=user_id)
    user_prompt                      = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        history=history,
        feature_block=feature_block,
    )

    output = request_claude_analysis(system_prompt, user_prompt)

    pred = parse_prediction_from_output(output)
    validation_errors = validate_prediction_plan(pred, mode, timeframe_data, current_price)
    if validation_errors:
        repaired = request_claude_repair(system_prompt, user_prompt, output, validation_errors)
        repaired_pred = parse_prediction_from_output(repaired)
        repaired_errors = validate_prediction_plan(repaired_pred, mode, timeframe_data, current_price)
        output, pred, validation_errors = repaired, repaired_pred, repaired_errors

    if pred["direction"] in ("LONG", "SHORT") and not validation_errors:
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
            feature_snapshot=feature_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
    elif validation_errors:
        save_rejected_prediction(
            symbol=binance_symbol,
            mode=mode,
            direction=pred.get("direction"),
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary="Validator rejected plan: " + " ; ".join(validation_errors[:5]),
            full_response=output,
            validation_errors=validation_errors,
            user_id=user_id,
            chat_id=chat_id,
        )

    return maybe_append_not_saved_warning(output, validation_errors)


async def collect_timeframe_data(binance_symbol: str, mode: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch nhiều timeframe song song trong worker threads.

    Mục tiêu: không để requests.get() block event loop của Telegram bot, và cũng
    giảm thời gian chờ vì 15M/1H/4H hoặc 4H/1D/1W được tải song song.
    """
    configs = SHORT_TERM_TIMEFRAMES if mode == "short" else LONG_TERM_TIMEFRAMES
    tasks = {
        label: asyncio.to_thread(load_timeframe_data, binance_symbol, interval, limit)
        for label, (interval, limit) in configs.items()
    }
    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def analyze_symbol(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> str:
    """
    Async entry point used by Telegram handlers.

    Không gọi requests.get(), Anthropic SDK sync hoặc SQLite trực tiếp trên event loop.
    Các phần I/O blocking được chuyển sang worker thread bằng asyncio.to_thread().
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env.")

    await asyncio.to_thread(init_prediction_db)

    binance_symbol = f"{symbol.upper()}USDT"

    # Binance requests.get() chạy trong worker threads, các timeframe tải song song.
    timeframe_data = await collect_timeframe_data(binance_symbol, mode)

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        raise RuntimeError(f"Could not fetch Binance data for {binance_symbol}.")

    # Các nguồn dữ liệu sync khác cũng được wrap bằng to_thread.
    system_prompt_task = asyncio.to_thread(load_system_prompt)
    fear_greed_task = asyncio.to_thread(get_fear_greed_index)
    current_price_task = asyncio.to_thread(get_current_price_str, binance_symbol)
    history_task = asyncio.to_thread(get_recent_predictions, binance_symbol, mode, user_id)

    system_prompt, fear_greed_info, price_tuple, history = await asyncio.gather(
        system_prompt_task,
        fear_greed_task,
        current_price_task,
        history_task,
    )
    current_price_str, current_price = price_tuple

    feature_block = build_feature_engineering_block(timeframe_data, mode, current_price)
    feature_snapshot = build_feature_snapshot(timeframe_data, mode, current_price)
    market_snapshot = build_market_snapshot(
        timeframe_data,
        fear_greed_info,
        current_price_str,
    )
    user_prompt = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        history=history,
        feature_block=feature_block,
    )

    # Anthropic SDK đang sync, nên gọi trong worker thread để không block bot.
    output = await asyncio.to_thread(request_claude_analysis, system_prompt, user_prompt)

    pred = parse_prediction_from_output(output)
    validation_errors = validate_prediction_plan(pred, mode, timeframe_data, current_price)
    if validation_errors:
        repaired = await asyncio.to_thread(
            request_claude_repair,
            system_prompt,
            user_prompt,
            output,
            validation_errors,
        )
        repaired_pred = parse_prediction_from_output(repaired)
        repaired_errors = validate_prediction_plan(repaired_pred, mode, timeframe_data, current_price)
        output, pred, validation_errors = repaired, repaired_pred, repaired_errors

    if pred["direction"] in ("LONG", "SHORT") and not validation_errors:
        reasoning_summary = await asyncio.to_thread(summarize_reasoning, output)
        await asyncio.to_thread(
            save_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=pred["direction"],
            entry_low=pred["entry_low"],
            entry_high=pred["entry_high"],
            sl=pred["sl"],
            tp1=pred["tp1"],
            tp2=pred["tp2"],
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
    elif validation_errors:
        await asyncio.to_thread(
            save_rejected_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=pred.get("direction"),
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary="Validator rejected plan: " + " ; ".join(validation_errors[:5]),
            full_response=output,
            validation_errors=validation_errors,
            user_id=user_id,
            chat_id=chat_id,
        )

    return maybe_append_not_saved_warning(output, validation_errors)
