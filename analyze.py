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

# ─── AI provider config ───────────────────────────────────────────────────────
# Default vẫn là Anthropic/Claude để không làm vỡ config cũ.
# Muốn test GLM 5.2 qua OpenRouter trên Railway:
#   AI_PROVIDER=openrouter
#   OPENROUTER_API_KEY=<key>
#   OPENROUTER_MODEL=z-ai/glm-5.2
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic").strip().lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
# Claude Sonnet 5 mặc định high; Teopard default max để phân tích Entry/SL/TP sâu nhất khi dùng Anthropic.
ANTHROPIC_EFFORT  = os.getenv("ANTHROPIC_EFFORT", "max").strip()
ANTHROPIC_SUMMARY_EFFORT = os.getenv("ANTHROPIC_SUMMARY_EFFORT", "high").strip()

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", os.getenv("GLM_MODEL", "z-ai/glm-5.2"))
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Teopard Bot")

LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", os.getenv("CLAUDE_MAX_TOKENS", "8000")))
LLM_MAX_CONTINUATIONS = int(os.getenv("LLM_MAX_CONTINUATIONS", "2"))
# Call tóm tắt reasoning dùng token riêng và KHÔNG continuation để tránh GLM lặp length vì reasoning token.
LLM_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_OUTPUT_TOKENS", "600"))
# Mặc định tắt reasoning cho summary. Phân tích chính vẫn dùng OPENROUTER_REASONING_EFFORT nếu bạn set high/xhigh.
OPENROUTER_SUMMARY_REASONING_EFFORT = os.getenv("OPENROUTER_SUMMARY_REASONING_EFFORT", "off").strip()
# Giữ tên cũ để code cũ không crash nếu còn tham chiếu.
CLAUDE_MAX_TOKENS = LLM_MAX_OUTPUT_TOKENS

DB_PATH           = os.getenv("DB_PATH", "bot.db")

# Ngắn hạn: scalp trong ngày — 15m timing, 1H momentum, 4H xu hướng chính
SHORT_TERM_TIMEFRAMES = {
    "15M": ("15m", 200),
    "1H":  ("1h",  150),
    "4H":  ("4h",  180),
}

# Dài hạn: swing/position — 1H kiểm tra sweep, 4H entry, 1D xu hướng, 1W big picture
# 1H chỉ dùng làm dữ liệu xác nhận rút râu/quét thanh khoản; khung lập lệnh SWING vẫn là 4H/1D/1W.
LONG_TERM_TIMEFRAMES = {
    "1H": ("1h",  240),
    "4H": ("4h",  180),
    "1D": ("1d",  150),
    "1W": ("1w",  120),
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
VISIBLE_PREDICTION_RETENTION_LIMIT = 10
HIDDEN_LEARNING_RETENTION_LIMIT = 10
HIDDEN_LEARNING_RESULTS = ("REJECTED_PLAN", "NO_TRADE")
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

        # Index nhẹ cho history/stats/learning/auto-check khi DB lớn hơn.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id_id ON predictions(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_symbol_mode_id ON predictions(user_id, symbol, mode, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_next_check ON predictions(result, next_check_at)")
        conn.commit()


def prune_prediction_history(user_id: int | None) -> None:
    """Giữ DB gọn: mỗi user chỉ giữ 10 lệnh hiển thị gần nhất.

    - /history chỉ dùng nhóm lệnh hiển thị, nên nhóm này được giữ đúng 10 dòng mới nhất.
    - NO_TRADE/REJECTED_PLAN là bản ghi học ẩn, không hiện trong /history; vẫn giới hạn
      riêng để DB không phình theo thời gian.
    - Learning prompt vẫn chỉ lấy PREDICTION_HISTORY_COUNT = 5 dòng gần nhất theo user/symbol/mode.
    """
    if user_id is None:
        return

    hidden_a, hidden_b = HIDDEN_LEARNING_RESULTS
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM predictions
            WHERE user_id=?
              AND result NOT IN (?, ?)
              AND id NOT IN (
                  SELECT id
                  FROM predictions
                  WHERE user_id=?
                    AND result NOT IN (?, ?)
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, hidden_a, hidden_b, user_id, hidden_a, hidden_b, VISIBLE_PREDICTION_RETENTION_LIMIT),
        )
        conn.execute(
            """
            DELETE FROM predictions
            WHERE user_id=?
              AND result IN (?, ?)
              AND id NOT IN (
                  SELECT id
                  FROM predictions
                  WHERE user_id=?
                    AND result IN (?, ?)
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (user_id, hidden_a, hidden_b, user_id, hidden_a, hidden_b, HIDDEN_LEARNING_RETENTION_LIMIT),
        )
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
        prediction_id = cursor.lastrowid
        conn.commit()
    prune_prediction_history(user_id)
    return prediction_id


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
        prediction_id = cursor.lastrowid
        conn.commit()
    prune_prediction_history(user_id)
    return prediction_id


def save_no_trade_prediction(
    symbol: str,
    mode: str,
    market_snapshot: str | None,
    feature_snapshot: str | None,
    reasoning_summary: str | None,
    full_response: str | None,
    user_id: int | None = None,
    chat_id: int | None = None,
) -> int:
    """
    Lưu quyết định NO_TRADE để Claude học được lúc nào nên đứng ngoài.

    Bản ghi này KHÔNG được auto-check, KHÔNG hiện trong /history, /stats, /dashboard.
    Nó chỉ được dùng trong per-user learning context cho những lần phân tích sau.
    """
    now = utc_now()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (user_id, chat_id, symbol, mode, created_at, check_after_hours, entry_wait_hours, max_hold_hours,
                 next_check_at, direction, entry_low, entry_high, sl, tp1, tp2,
                 entry_status, market_snapshot, feature_snapshot, reasoning_summary, full_response,
                 result, result_reason, result_checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'NO_TRADE', NULL, NULL, NULL, NULL, NULL,
                    'NO_TRADE', ?, ?, ?, ?, 'NO_TRADE', ?, ?)
            """,
            (user_id, chat_id, symbol, mode, iso(now),
             ENTRY_WAIT_HOURS.get(mode, 24), ENTRY_WAIT_HOURS.get(mode, 24), TRADE_MAX_HOLD_HOURS.get(mode, 72),
             market_snapshot, feature_snapshot, reasoning_summary or "Claude chọn NO_TRADE.", full_response,
             "Claude chọn NO_TRADE vì chưa có setup đủ rõ hoặc risk/reward chưa đáng để tạo tín hiệu.", iso(now)),
        )
        prediction_id = cursor.lastrowid
        conn.commit()
    prune_prediction_history(user_id)
    return prediction_id


def _row_to_pred(row) -> dict:
    keys = [
        "id", "user_id", "chat_id", "symbol", "mode", "created_at",
        "entry_wait_hours", "max_hold_hours", "next_check_at", "direction",
        "entry_low", "entry_high", "sl", "tp1", "tp2", "entry_status",
        "entry_filled_at", "entry_price", "result"
    ]
    return dict(zip(keys, row))


def get_due_predictions(force: bool = False) -> list[dict]:
    """
    Lấy prediction đang mở để auto-check.

    - force=False: chỉ lấy prediction đến hạn theo next_check_at, dùng cho job định kỳ.
    - force=True: lấy toàn bộ PENDING_ENTRY/ENTRY_FILLED, dùng cho /checknow để ép kiểm tra ngay.
    """
    now_s = iso(utc_now())
    where_due = "" if force else "AND (next_check_at IS NULL OR next_check_at <= ?)"
    params = () if force else (now_s,)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, created_at,
                   entry_wait_hours, max_hold_hours, next_check_at, direction,
                   entry_low, entry_high, sl, tp1, tp2, entry_status,
                   entry_filled_at, entry_price, result
            FROM predictions
            WHERE result IN ('PENDING_ENTRY', 'ENTRY_FILLED')
              {where_due}
            ORDER BY id ASC
            LIMIT 200
            """,
            params,
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



def get_open_signal_predictions(
    symbol: str,
    mode: str,
    user_id: int | None = None,
    limit: int = 2,
) -> list[dict]:
    """Lấy kế hoạch đang mở để model không hiểu nhầm lệnh chờ thành lệnh ngược.

    Chỉ lấy theo đúng user + symbol + mode để không lộ dữ liệu user khác và không làm
    prompt dài. Dùng cho awareness khi user phân tích lại cùng coin/mode trong lúc
    tín hiệu cũ vẫn PENDING_ENTRY hoặc ENTRY_FILLED.
    """
    if user_id is None:
        return []

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, direction, entry_low, entry_high, sl, tp1, tp2,
                   result, entry_status, entry_filled_at, entry_price, result_reason
            FROM predictions
            WHERE symbol=? AND mode=? AND user_id=?
              AND result IN ('PENDING_ENTRY', 'ENTRY_FILLED')
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol, mode, user_id, limit),
        ).fetchall()

    return [
        {
            "id": row[0],
            "created_at": row[1],
            "direction": row[2],
            "entry_low": row[3],
            "entry_high": row[4],
            "sl": row[5],
            "tp1": row[6],
            "tp2": row[7],
            "result": row[8],
            "entry_status": row[9],
            "entry_filled_at": row[10],
            "entry_price": row[11],
            "result_reason": row[12],
        }
        for row in rows
    ]


def _price_vs_entry_text(current_price: float | None, entry_low: float | None, entry_high: float | None) -> str:
    if current_price is None or entry_low is None or entry_high is None:
        return "không đủ dữ liệu để so với giá hiện tại"
    low = min(float(entry_low), float(entry_high))
    high = max(float(entry_low), float(entry_high))
    if low <= current_price <= high:
        return "giá hiện tại đang nằm trong vùng Entry cũ"
    if current_price < low:
        return "giá hiện tại đang thấp hơn vùng Entry cũ"
    return "giá hiện tại đang cao hơn vùng Entry cũ"


def format_open_signal_context(open_signals: list[dict], current_price: float | None) -> str:
    """Tạo block awareness ngắn gọn cho các kế hoạch đang mở.

    Mục tiêu chính: tránh tình huống model đưa LONG chờ hồi, user phân tích lại rồi
    model đuổi giá hoặc user hiểu Entry LONG là TP cho lệnh SHORT.
    """
    if not open_signals:
        return "KẾ HOẠCH ĐANG MỞ: Không có kế hoạch đang chờ/đã khớp cho user này ở cùng coin và mode."

    lines = ["KẾ HOẠCH ĐANG MỞ CÙNG USER/COIN/MODE:"]
    for p in open_signals:
        entry = f"{fmt(p.get('entry_low'))}-{fmt(p.get('entry_high'))}" if p.get("entry_low") is not None and p.get("entry_high") is not None else "N/A"
        status = p.get("result") or p.get("entry_status") or "N/A"
        relation = _price_vs_entry_text(current_price, p.get("entry_low"), p.get("entry_high"))
        extra = ""
        if status == "ENTRY_FILLED":
            extra = f"; đã khớp lúc {str(p.get('entry_filled_at') or '')[:16]}, giá khớp {fmt(p.get('entry_price'))}"
        elif status == "PENDING_ENTRY":
            extra = "; chưa khớp Entry"
        lines.append(
            f"- #{p.get('id')} {str(p.get('created_at') or '')[:16]} {p.get('direction')} {status}{extra}; "
            f"Entry {entry}, SL {fmt(p.get('sl'))}, TP1 {fmt(p.get('tp1'))}, TP2 {fmt(p.get('tp2'))}; {relation}."
        )

    lines.extend([
        "Cách dùng kế hoạch đang mở:",
        "- Nếu kế hoạch cũ là LONG chờ hồi, vùng Entry LONG KHÔNG phải TP cho lệnh SHORT ngược lại. Nếu kế hoạch cũ là SHORT chờ hồi, vùng Entry SHORT KHÔNG phải TP cho lệnh LONG ngược lại.",
        "- Khi phân tích lại, phải xem kế hoạch cũ là còn hiệu lực, bị hủy, hay cần thay bằng kế hoạch mới. Nếu thay kế hoạch, nêu ngắn lý do trong Kịch bản chính.",
        "- Nếu giá không hồi về Entry cũ mà đã chạy theo hướng dự báo, không được đuổi giá chỉ vì giá đang chạy. Chỉ cho vào ngay khi giá hiện tại nằm trong vùng Entry mới hợp lý và đã có xác nhận rõ; nếu không, ưu tiên NO_TRADE hoặc chờ kiểm tra lại.",
    ])
    return "\n".join(lines)


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


def _interval_to_timedelta(interval: str) -> timedelta:
    """Khoảng thời gian của nến Binance, dùng để fetch lùi 1 cây tránh miss nến overlap lúc tạo signal."""
    m = re.fullmatch(r"(\d+)([mhdw])", interval.strip().lower())
    if not m:
        return timedelta(minutes=15)
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    if unit == "w":
        return timedelta(weeks=n)
    return timedelta(minutes=15)


def _range_low_high(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    low = min(float(a), float(b))
    high = max(float(a), float(b))
    return low, high


def _entry_touched(direction: str, entry_low: float | None, entry_high: float | None, high: float, low: float) -> bool:
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return False
    # Một nến chạm vùng Entry khi biên [low, high] của nến giao với biên Entry.
    return low <= high_zone and high >= low_zone


def _price_in_entry_range(price: float | None, entry_low: float | None, entry_high: float | None) -> bool:
    if price is None:
        return False
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return False
    return low_zone <= float(price) <= high_zone


def _entry_price(direction: str, entry_low: float | None, entry_high: float | None, fill_price: float | None = None) -> float | None:
    if fill_price is not None:
        return float(fill_price)
    low_zone, high_zone = _range_low_high(entry_low, entry_high)
    if low_zone is None or high_zone is None:
        return None
    return (low_zone + high_zone) / 2


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


def evaluate_prediction_lifecycle(
    pred: dict,
    candles: pd.DataFrame | None,
    current_price: float | None = None,
) -> dict:
    """
    Chấm vòng đời prediction.

    Quy tắc quan trọng:
    - Signal tạo lúc T thì chỉ xét dữ liệu có close_time sau T.
    - PENDING_ENTRY được fill nếu current price hiện tại nằm trong vùng Entry.
    - Entry range được hiểu là vùng giá: entry_low <= price <= entry_high, không phụ thuộc LONG/SHORT.
    - Sau khi Entry đã khớp, TP/SL chỉ được xét từ entry_filled_at trở đi.
    """
    now = utc_now()
    created = parse_utc_datetime(pred.get("created_at"))
    entry_filled_at = parse_utc_datetime(pred.get("entry_filled_at"))
    if created is None:
        return {"action": "skip", "reason": "Không đọc được thời gian tạo prediction."}

    status = pred.get("result") or pred.get("entry_status") or "PENDING_ENTRY"

    if status == "PENDING_ENTRY":
        entry_deadline = created + timedelta(hours=int(pred.get("entry_wait_hours") or 24))

        # Check live price trước để không bỏ lỡ trường hợp giá hiện tại đang nằm trong vùng Entry.
        # Ví dụ Entry 50000-50500, current price 50300 => ENTRY_FILLED ngay.
        if now <= entry_deadline and _price_in_entry_range(current_price, pred.get("entry_low"), pred.get("entry_high")):
            return {
                "action": "fill",
                "price": _entry_price(pred["direction"], pred.get("entry_low"), pred.get("entry_high"), current_price),
                "filled_at": now,
                "reason": f"Giá hiện tại {current_price} đang nằm trong vùng Entry.",
            }

        if candles is None or candles.empty:
            if now >= entry_deadline:
                return {
                    "action": "close",
                    "result": "NOT_FILLED",
                    "price": current_price,
                    "reason": f"Hết thời gian chờ Entry {pred.get('entry_wait_hours')}h nhưng không có dữ liệu nến để xác nhận giá đã chạm Entry.",
                    "closed_at": now,
                }
            return {"action": "reschedule", "reason": "Không có dữ liệu nến."}

        # Fetch có thể đã lùi 1 cây để bắt nến overlap, nhưng chỉ xét nến đóng sau thời điểm tạo signal.
        pending_candles = candles[candles["close_time"] > pd.Timestamp(created)]
        # Không fill Entry bằng nến đóng sau deadline chờ Entry.
        pending_candles = pending_candles[pending_candles["close_time"] <= pd.Timestamp(entry_deadline)]

        for _, row in pending_candles.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if _entry_touched(pred["direction"], pred.get("entry_low"), pred.get("entry_high"), high, low):
                filled_at_ts = row["close_time"]
                filled_at = filled_at_ts.to_pydatetime() if hasattr(filled_at_ts, "to_pydatetime") else now
                entry_price = _entry_price(pred["direction"], pred.get("entry_low"), pred.get("entry_high"))
                post = candles[candles["close_time"] >= row["close_time"]]
                result, price, reason, closed_at = _tp_sl_result({**pred, "entry_price": entry_price}, post)
                if result in ("WIN", "LOSS", "AMBIGUOUS"):
                    return {
                        "action": "close",
                        "result": result,
                        "price": price,
                        "reason": f"Entry khớp rồi {reason}",
                        "closed_at": closed_at or filled_at,
                        "entry_price": entry_price,
                        "entry_filled_at": filled_at,
                    }
                return {
                    "action": "fill",
                    "price": entry_price,
                    "filled_at": filled_at,
                    "reason": f"Entry đã khớp trong nến đóng lúc {str(row['close_time'])[:16]}.",
                }

        if now >= entry_deadline:
            fallback_price = current_price
            if fallback_price is None and candles is not None and not candles.empty:
                fallback_price = float(candles.iloc[-1]["close"])
            return {
                "action": "close",
                "result": "NOT_FILLED",
                "price": fallback_price,
                "reason": f"Hết thời gian chờ Entry {pred.get('entry_wait_hours')}h nhưng giá chưa chạm vùng Entry.",
                "closed_at": now,
            }
        return {"action": "reschedule", "reason": "Chưa chạm Entry, tiếp tục chờ."}

    if status == "ENTRY_FILLED":
        if entry_filled_at is None:
            return {"action": "reschedule", "reason": "Thiếu entry_filled_at."}
        if candles is None or candles.empty:
            return {"action": "reschedule", "reason": "Không có dữ liệu nến."}
        filled_candles = candles[candles["close_time"] > pd.Timestamp(entry_filled_at)]
        if filled_candles.empty:
            return {"action": "reschedule", "reason": "Chưa có nến đóng sau thời điểm khớp Entry."}
        result, price, reason, closed_at = _tp_sl_result(pred, filled_candles)
        if result in ("WIN", "LOSS", "AMBIGUOUS"):
            return {
                "action": "close",
                "result": result,
                "price": price,
                "reason": reason,
                "closed_at": closed_at or now,
                "entry_price": pred.get("entry_price"),
                "entry_filled_at": entry_filled_at,
            }
        hold_deadline = entry_filled_at + timedelta(hours=int(pred.get("max_hold_hours") or 72))
        if now >= hold_deadline:
            return {
                "action": "close",
                "result": "EXPIRED",
                "price": price or current_price,
                "reason": f"Đã khớp Entry nhưng quá thời gian giữ lệnh {pred.get('max_hold_hours')}h mà chưa chạm TP1/SL.",
                "closed_at": now,
                "entry_price": pred.get("entry_price"),
                "entry_filled_at": entry_filled_at,
            }
        return {"action": "reschedule", "reason": reason}

    return {"action": "skip", "reason": f"Trạng thái {status} không cần kiểm tra."}





async def auto_check_pending_predictions(force: bool = False) -> dict:
    """Check predictions đang mở, chỉ cập nhật DB và trả về số liệu tóm tắt.

    Hàm này cố ý không tạo notification để gửi cho user/admin nữa.
    User muốn xem kết quả thì chủ động dùng /history, /stats hoặc /dashboard.
    """
    init_prediction_db()
    due = get_due_predictions(force=force)
    entry_filled_count = 0
    closed_count = 0
    rescheduled_count = 0
    skipped_count = 0

    check_label = "all active predictions" if force else "due predictions"
    print(f"[AUTO_CHECK] Checking {len(due)} {check_label} at {iso(utc_now())}", flush=True)

    for pred in due:
        start_dt = parse_utc_datetime(pred.get("entry_filled_at")) or parse_utc_datetime(pred.get("created_at"))
        if start_dt is None:
            skipped_count += 1
            continue
        result_interval = get_result_check_interval(pred.get("mode", "short"))
        fetch_start = start_dt - _interval_to_timedelta(result_interval)
        current_price = None
        if (pred.get("result") or pred.get("entry_status")) == "PENDING_ENTRY":
            current_price = await asyncio.to_thread(get_current_price_raw, pred["symbol"])
        candles = await asyncio.to_thread(get_binance_klines_since, pred["symbol"], result_interval, fetch_start)
        decision = evaluate_prediction_lifecycle(pred, candles, current_price=current_price)
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
        "force": force,
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
    # REJECTED_PLAN và NO_TRADE là bản ghi học nội bộ, không hiển thị trong /history, /stats, /dashboard
    # để user/admin không nhầm chúng là tín hiệu thật. get_recent_predictions() vẫn đọc được
    # các bản ghi này để Claude học từ lỗi validator hoặc các lần nên đứng ngoài.
    if not include_rejected:
        clauses.append("result NOT IN ('REJECTED_PLAN', 'NO_TRADE')")
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


def _window_tail(df: pd.DataFrame | None, hours: int | None = None, max_candles: int | None = None) -> pd.DataFrame | None:
    """Lấy dữ liệu theo cửa sổ thời gian thay vì số cây cố định.

    Coinglass dùng 12h/24h/48h như một *lookback window*; không phải nghĩa là
    phải dùng nến 12H/24H. Với Teopard, ta vẫn dùng nến nhỏ hơn để giữ độ phân giải,
    nhưng chỉ xét các cây nằm trong cửa sổ thời gian đó.
    """
    if df is None or df.empty:
        return None
    data = df.copy()
    if hours is not None:
        time_col = "close_time" if "close_time" in data.columns else "timestamp"
        ref_time = data[time_col].max()
        start_time = ref_time - pd.Timedelta(hours=hours)
        data = data[data[time_col] >= start_time]
    if max_candles is not None and len(data) > max_candles:
        data = data.tail(max_candles)
    return data.reset_index(drop=True)



def _find_pivots(df: pd.DataFrame | None, side: str, lookback: int | None = 100, left: int = 2, right: int = 2) -> list[dict]:
    if df is None or df.empty:
        return []
    data = df.tail(lookback).reset_index(drop=True) if lookback else df.reset_index(drop=True)
    col = "high" if side == "high" else "low"
    pivots: list[dict] = []
    if len(data) < left + right + 1:
        return pivots
    for i in range(left, len(data) - right):
        val = float(data.loc[i, col])
        window = data.loc[i - left:i + right, col]
        if side == "high" and val >= float(window.max()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"], "index": i, "kind": "pivot", "weight": 1.0})
        elif side == "low" and val <= float(window.min()):
            pivots.append({"price": val, "time": data.loc[i, "timestamp"], "index": i, "kind": "pivot", "weight": 1.0})
    return pivots


def _cluster_zone(prices: list[float], current_price: float, side: str, atr: float | None) -> tuple[float | None, float | None, int]:
    """Legacy helper giữ lại cho một vài fallback cũ nếu cần."""
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
    else:
        candidates = [c for c in clusters if sum(c) / len(c) >= current_price]

    if not candidates:
        return None, None, 0
    candidates.sort(key=lambda c: (len(c), -abs(current_price - sum(c) / len(c))), reverse=True)
    best = candidates[0]
    low = min(best) - buf
    high = max(best) + buf
    return low, high, len(best)


def _liquidity_ref_atr(current_price: float, atr: float | None) -> float:
    """ATR tham chiếu để gom vùng. Fallback theo % giá để tránh vùng quá mỏng khi ATR rỗng."""
    return max(float(atr or 0), current_price * 0.0018)


def _liquidity_tolerance(current_price: float, atr: float | None, role: str = "main") -> float:
    ref_atr = _liquidity_ref_atr(current_price, atr)
    role_mult = {"near": 0.20, "main": 0.24, "deep": 0.28}.get(role, 0.24)
    return max(ref_atr * role_mult, current_price * 0.00065)


def _liquidity_buffer(current_price: float, atr: float | None, role: str = "main") -> float:
    ref_atr = _liquidity_ref_atr(current_price, atr)
    role_mult = {"near": 0.12, "main": 0.14, "deep": 0.16}.get(role, 0.14)
    return max(ref_atr * role_mult, current_price * 0.00035)


def _candle_wick_stats(row) -> tuple[float, float, float, float]:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    rng = max(high - low, 1e-12)
    upper = max(high - max(open_, close), 0.0) / rng
    lower = max(min(open_, close) - low, 0.0) / rng
    body = abs(close - open_) / rng
    return upper, lower, body, rng


def _collect_liquidity_points(
    window_df: pd.DataFrame | None,
    side: str,
    current_price: float,
    atr: float | None,
    role: str = "main",
) -> list[dict]:
    """Thu thập điểm thanh khoản ước lượng từ pivot, equal high/low và nến quét râu.

    Mục tiêu là chọn vùng có ý nghĩa giao dịch nhất, không phải ép near/main/deep
    phải cách xa nhau. Nếu cùng một cụm được chạm nhiều lần trong nhiều cửa sổ,
    vùng đó có thể xuất hiện lại, nhưng sẽ có thống kê chạm/quét/vol để AI hiểu
    đúng chất lượng vùng.
    """
    if window_df is None or window_df.empty:
        return []

    data = window_df.reset_index(drop=True)
    left_right = 1 if len(data) < 12 else 2
    points = _find_pivots(data, side, lookback=None, left=left_right, right=left_right)
    tol = _liquidity_tolerance(current_price, atr, role)

    col = "high" if side == "high" else "low"

    # Thêm các cú rút râu/quét đỉnh-đáy. Đây thường là nơi stop/liq bị quét.
    for i, row in data.iterrows():
        upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
        price = float(row[col])
        if side == "high":
            is_sweep_like = upper_wick >= 0.32 and upper_wick >= body_pct * 0.8
        else:
            is_sweep_like = lower_wick >= 0.32 and lower_wick >= body_pct * 0.8
        if is_sweep_like:
            points.append({
                "price": price,
                "time": row.get("timestamp"),
                "index": int(i),
                "kind": "wick_sweep",
                "weight": 1.25,
            })

    # Thêm các cụm equal high/equal low: 2 lần chạm gần nhau trong phạm vi tol.
    # Không thêm mọi nến để tránh biến thành volume profile giả.
    values = [float(v) for v in data[col].tail(min(len(data), 80)).tolist()]
    offset = len(data) - len(values)
    for i in range(1, len(values)):
        prev = values[i - 1]
        cur = values[i]
        if abs(cur - prev) <= tol:
            row = data.iloc[offset + i]
            points.append({
                "price": cur,
                "time": row.get("timestamp"),
                "index": int(offset + i),
                "kind": "equal_touch",
                "weight": 0.85,
            })

    # Luôn thêm cực trị của cửa sổ để không bỏ sót high/low quan trọng khi pivot rỗng.
    if not data.empty:
        if side == "high":
            idx = int(data["high"].idxmax())
            price = float(data.loc[idx, "high"])
        else:
            idx = int(data["low"].idxmin())
            price = float(data.loc[idx, "low"])
        points.append({
            "price": price,
            "time": data.loc[idx, "timestamp"],
            "index": idx,
            "kind": "window_extreme",
            "weight": 0.9,
        })

    return points


def _zone_side_state(zone: tuple | None, current_price: float | None) -> str:
    if not zone or current_price is None:
        return "unknown"
    low = zone[0] if len(zone) > 0 else None
    high = zone[1] if len(zone) > 1 else None
    if low is None or high is None:
        return "unknown"
    if float(low) <= current_price <= float(high):
        return "touching"
    if float(high) < current_price:
        return "below"
    if float(low) > current_price:
        return "above"
    return "overlap"


def _zone_meta_default(role: str = "main") -> dict:
    return {"role": role, "touches": 0, "sweeps": 0, "vol_ratio": None, "score": 0.0}


def _cluster_zone_from_pivots(
    pivots: list[dict],
    current_price: float,
    side: str,
    atr: float | None,
    window_df: pd.DataFrame | None,
    role: str = "main",
) -> tuple[float | None, float | None, int, dict]:
    """Gom điểm thanh khoản thành vùng và chọn vùng có chất lượng cao nhất.

    Điểm số ưu tiên vùng có nhiều lần chạm, có quét râu, volume tốt, còn mới và
    khoảng cách hợp vai trò. Không ép vùng phải tách xa nhau; nếu thị trường thật
    sự đang giao dịch quanh cùng một cụm thanh khoản thì vùng gần/chính/sâu có
    thể gần nhau, nhưng metadata sẽ báo rõ đang chạm giá/trùng vai trò.
    """
    if not pivots:
        return None, None, 0, _zone_meta_default(role)

    tol = _liquidity_tolerance(current_price, atr, role)
    buf = _liquidity_buffer(current_price, atr, role)
    sorted_pivots = sorted(pivots, key=lambda p: float(p["price"]))

    clusters: list[list[dict]] = []
    cur = [sorted_pivots[0]]
    for pivot in sorted_pivots[1:]:
        center = sum(float(p["price"]) for p in cur) / len(cur)
        if abs(float(pivot["price"]) - center) <= tol:
            cur.append(pivot)
        else:
            clusters.append(cur)
            cur = [pivot]
    clusters.append(cur)

    if side == "low":
        candidates = [c for c in clusters if (sum(float(p["price"]) for p in c) / len(c)) <= current_price]
    else:
        candidates = [c for c in clusters if (sum(float(p["price"]) for p in c) / len(c)) >= current_price]

    if not candidates:
        return None, None, 0, _zone_meta_default(role)

    data = window_df.reset_index(drop=True) if window_df is not None and not window_df.empty else None
    ref_time = data["timestamp"].max() if data is not None and "timestamp" in data.columns else None
    ref_atr = _liquidity_ref_atr(current_price, atr)

    def cluster_stats(cluster: list[dict]) -> dict:
        prices = [float(p["price"]) for p in cluster]
        raw_low, raw_high = min(prices), max(prices)
        low, high = raw_low - buf, raw_high + buf
        center = sum(prices) / len(prices)
        touch_count = 0
        sweep_count = 0
        vol_values: list[float] = []
        recent_touch_age_hours = None

        if data is not None:
            for _, row in data.iterrows():
                high_v = float(row["high"])
                low_v = float(row["low"])
                close_v = float(row["close"])
                open_v = float(row["open"])
                upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
                price_v = high_v if side == "high" else low_v
                touched = (low - tol) <= price_v <= (high + tol)
                if touched:
                    touch_count += 1
                    vol_ratio = _safe_float(row.get("vol_ratio"))
                    if vol_ratio is not None and np.isfinite(vol_ratio):
                        vol_values.append(float(vol_ratio))
                    if ref_time is not None:
                        try:
                            age = max((pd.Timestamp(ref_time) - pd.Timestamp(row["timestamp"])).total_seconds() / 3600.0, 0.0)
                            recent_touch_age_hours = age if recent_touch_age_hours is None else min(recent_touch_age_hours, age)
                        except Exception:
                            pass

                if side == "high":
                    # Quét lên: chọc qua vùng high/liquidity rồi đóng thấp lại với râu trên rõ.
                    swept = high_v >= low and close_v < center and upper_wick >= 0.25 and high_v > max(open_v, close_v)
                else:
                    # Quét xuống: chọc xuống vùng low/liquidity rồi đóng cao lại với râu dưới rõ.
                    swept = low_v <= high and close_v > center and lower_wick >= 0.25 and low_v < min(open_v, close_v)
                if swept:
                    sweep_count += 1

        if touch_count == 0:
            touch_count = len(cluster)

        avg_vol = float(np.mean(vol_values)) if vol_values else None
        point_weight = sum(float(p.get("weight", 1.0)) for p in cluster)
        pivot_hits = sum(1 for p in cluster if p.get("kind") == "pivot")
        equal_hits = sum(1 for p in cluster if p.get("kind") == "equal_touch")
        wick_hits = sum(1 for p in cluster if p.get("kind") == "wick_sweep")

        distance_atr = abs(center - current_price) / max(ref_atr, 1e-12)
        if role == "near":
            distance_score = max(0.0, 1.0 - distance_atr / 4.0) * 1.8
        elif role == "deep":
            # Deep không bị ép xa, nhưng không thưởng quá mạnh cho vùng đang sát giá.
            distance_score = max(0.0, min(distance_atr / 3.0, 1.0)) * 0.8
        else:
            distance_score = max(0.0, 1.0 - abs(distance_atr - 1.6) / 5.0) * 1.1

        recency_score = 0.0
        if recent_touch_age_hours is not None:
            recency_score = 1.2 / (1.0 + recent_touch_age_hours / 18.0)
        elif ref_time is not None:
            try:
                last_touch = max(pd.Timestamp(p["time"]) for p in cluster if p.get("time") is not None)
                age_hours = max((pd.Timestamp(ref_time) - last_touch).total_seconds() / 3600.0, 0.0)
                recency_score = 0.9 / (1.0 + age_hours / 24.0)
            except Exception:
                recency_score = 0.0

        vol_score = 0.0
        if avg_vol is not None:
            # Volume cao là tốt, nhưng không để một cây volume dị thường áp đảo mọi thứ.
            vol_score = min(max(avg_vol - 0.8, 0.0), 1.8) * 0.8

        score = (
            min(point_weight, 8.0) * 1.1
            + min(touch_count, 8) * 0.9
            + min(sweep_count, 5) * 1.25
            + min(pivot_hits, 5) * 0.35
            + min(equal_hits, 5) * 0.25
            + min(wick_hits, 5) * 0.35
            + vol_score
            + recency_score
            + distance_score
        )

        return {
            "low": low,
            "high": high,
            "center": center,
            "hits": max(len(cluster), touch_count),
            "touches": touch_count,
            "sweeps": sweep_count,
            "vol_ratio": avg_vol,
            "score": score,
            "distance_atr": distance_atr,
            "pivot_hits": pivot_hits,
            "equal_hits": equal_hits,
            "wick_hits": wick_hits,
            "role": role,
        }

    scored = [cluster_stats(c) for c in candidates]
    best = max(scored, key=lambda m: m["score"])
    meta = {
        "role": role,
        "touches": int(best["touches"]),
        "sweeps": int(best["sweeps"]),
        "vol_ratio": best["vol_ratio"],
        "score": round(float(best["score"]), 2),
        "distance_atr": round(float(best["distance_atr"]), 2),
        "pivot_hits": int(best["pivot_hits"]),
        "equal_hits": int(best["equal_hits"]),
        "wick_hits": int(best["wick_hits"]),
    }
    return best["low"], best["high"], int(best["hits"]), meta


def _fallback_zone(
    df: pd.DataFrame | None,
    side: str,
    current_price: float,
    atr: float | None,
    window: int | None = None,
    role: str = "main",
) -> tuple[float | None, float | None, dict]:
    if df is None or df.empty:
        return None, None, _zone_meta_default(role)
    data = df.tail(window) if window else df
    if data.empty:
        return None, None, _zone_meta_default(role)
    buf = _liquidity_buffer(current_price, atr, role)
    if side == "low":
        idx = data["low"].idxmin()
        price = float(data.loc[idx, "low"])
    else:
        idx = data["high"].idxmax()
        price = float(data.loc[idx, "high"])

    low, high = price - buf, price + buf
    # Nếu toàn bộ cực trị fallback đã nằm sai phía so với giá hiện tại thì báo N/A.
    # Ví dụ giá vừa phá đỉnh 48h, không nên lấy đỉnh cũ dưới giá làm “vùng trên”.
    if side == "high" and high < current_price:
        return None, None, _zone_meta_default(role)
    if side == "low" and low > current_price:
        return None, None, _zone_meta_default(role)

    meta = _zone_meta_default(role)
    meta.update({"touches": 1, "score": 1.0, "fallback": True})
    return low, high, meta


def _zone_for_window(
    df: pd.DataFrame | None,
    current_price: float,
    atr: float | None,
    side: str,
    hours: int,
    max_candles: int | None = None,
    role: str = "main",
) -> tuple[float | None, float | None, int, dict]:
    data = _window_tail(df, hours=hours, max_candles=max_candles)
    if data is None or data.empty:
        return None, None, 0, _zone_meta_default(role)

    points = _collect_liquidity_points(data, side, current_price, atr, role)
    low, high, hits, meta = _cluster_zone_from_pivots(points, current_price, side, atr, data, role)
    if low is None or high is None:
        low, high, meta = _fallback_zone(data, side, current_price, atr, role=role)
        hits = 1 if low is not None else 0
    meta = dict(meta or _zone_meta_default(role))
    meta["side_state"] = _zone_side_state((low, high, hits, meta), current_price)
    return low, high, hits, meta


def _liquidity_zones_by_windows(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float,
) -> dict:
    """Vùng quét kỹ thuật theo cửa sổ thời gian và chất lượng hành vi giá.

    SCALP dùng tinh thần 12h/24h/48h: nến nhỏ giữ độ phân giải, cửa sổ thời gian
    giữ phạm vi giống cách xem heatmap hơn. SWING dùng 24h/3D/7D.

    near/main/deep là vai trò + cửa sổ dữ liệu, không phải lệnh ép các vùng phải
    cách xa nhau. Nếu vùng gần/chính/sâu trùng nhau vì cùng một cụm thanh khoản
    đang chi phối thị trường, output sẽ giữ vùng đó và ghi metadata chạm/quét/vol.
    """
    if mode == "short":
        near_df = timeframe_data.get("15M")
        main_df = timeframe_data.get("1H")
        deep_df = timeframe_data.get("1H")
        if deep_df is None or deep_df.empty:
            deep_df = timeframe_data.get("4H")
        near_atr = _current_atr(near_df) or _current_atr(main_df)
        main_atr = _current_atr(main_df) or near_atr
        deep_atr = main_atr or _current_atr(timeframe_data.get("4H"))
        windows = [
            ("near", "gần 12h", near_df, near_atr, 12, 64),
            ("main", "chính 24h", main_df, main_atr, 24, 36),
            ("deep", "sâu 48h", deep_df, deep_atr, 48, 60),
        ]
    else:
        near_df = timeframe_data.get("4H")
        main_df = timeframe_data.get("4H")
        if main_df is None or main_df.empty:
            main_df = timeframe_data.get("1D")
        deep_df = timeframe_data.get("1D")
        if deep_df is None or deep_df.empty:
            deep_df = timeframe_data.get("4H")
        near_atr = _current_atr(near_df) or _current_atr(timeframe_data.get("1D"))
        main_atr = near_atr or _current_atr(timeframe_data.get("1D"))
        deep_atr = _current_atr(deep_df) or main_atr
        windows = [
            ("near", "gần 24h", near_df, near_atr, 24, 12),
            ("main", "chính 3D", main_df, main_atr, 72, 24),
            ("deep", "sâu 7D", deep_df, deep_atr, 24 * 7, 14),
        ]

    result: dict = {"meta": [label for _, label, *_ in windows]}
    for key, label, df, atr, hours, max_candles in windows:
        lower = _zone_for_window(df, current_price, atr, "low", hours, max_candles, role=key)
        upper = _zone_for_window(df, current_price, atr, "high", hours, max_candles, role=key)
        result[f"lower_{key}"] = lower
        result[f"upper_{key}"] = upper
        result[f"label_{key}"] = label
    return result



# ─── Liquidity V5: fractal swing pools, not broad support/resistance bands ────
# Ghi chú: các hàm bên dưới cố ý override bộ liquidity V4 ở trên.
# Mục tiêu là ước lượng stop/liquidation pool từ OHLCV:
# - Lấy swing/fractal high-low làm level thanh khoản.
# - Box nằm NGOÀI swing: dưới đáy cho long liquidation, trên đỉnh cho short liquidation.
# - M15/khung nhỏ chỉ dùng để đánh dấu đã có sweep, không dùng để tạo box rộng quanh giá.
# - Không gom các level cách nhau xa thành một “cụm thanh khoản” rộng.

def _liq_role_params(role: str, mode: str = "short") -> dict:
    if mode == "short":
        params = {
            "near": {"tol_pct": 0.00028, "box_pct": 0.00075, "min_box_pct": 0.00028, "max_box_pct": 0.00105, "atr_mult": 0.12, "target_atr": 0.8},
            "main": {"tol_pct": 0.00036, "box_pct": 0.00105, "min_box_pct": 0.00035, "max_box_pct": 0.00145, "atr_mult": 0.16, "target_atr": 1.6},
            "deep": {"tol_pct": 0.00045, "box_pct": 0.00135, "min_box_pct": 0.00045, "max_box_pct": 0.00190, "atr_mult": 0.20, "target_atr": 2.6},
        }
    else:
        # SWING dùng H4/D1/W1 nên box được phép rộng hơn scalp, nhưng vẫn là stop-pool
        # nằm ngoài swing high/low, không phải một dải sideway quanh giá hiện tại.
        # near H4: vùng quanh swing gần để canh Entry; main D1: vùng TP/SL chính; deep W1/D1: vùng xa.
        params = {
            "near": {"tol_pct": 0.00055, "box_pct": 0.00180, "min_box_pct": 0.00070, "max_box_pct": 0.00320, "atr_mult": 0.16, "target_atr": 1.2},
            "main": {"tol_pct": 0.00085, "box_pct": 0.00350, "min_box_pct": 0.00110, "max_box_pct": 0.00650, "atr_mult": 0.22, "target_atr": 2.5},
            "deep": {"tol_pct": 0.00120, "box_pct": 0.00550, "min_box_pct": 0.00160, "max_box_pct": 0.01000, "atr_mult": 0.28, "target_atr": 4.0},
        }
    return params.get(role, params["main"])


def _liquidity_ref_atr(current_price: float, atr: float | None) -> float:
    # Fallback thấp hơn bản cũ để scalp không gom vùng quá rộng khi ATR rỗng/lớn.
    return max(float(atr or 0), current_price * 0.0012)


def _liquidity_tolerance(current_price: float, atr: float | None, role: str = "main", mode: str = "short") -> float:
    params = _liq_role_params(role, mode)
    ref_atr = _liquidity_ref_atr(current_price, atr)
    tol = max(current_price * params["tol_pct"], ref_atr * 0.055)
    # Đây là tolerance để nhận equal high/equal low, không phải width của vùng.
    return min(tol, current_price * params["max_box_pct"] * 0.45)


def _liquidity_buffer(current_price: float, atr: float | None, role: str = "main", mode: str = "short") -> float:
    # Giữ wrapper cũ cho fallback/legacy, nhưng V5 chủ yếu dùng _liq_box_width.
    return _liq_box_width(current_price, atr, role, mode) * 0.50


def _liq_box_width(current_price: float, atr: float | None, role: str, mode: str = "short") -> float:
    params = _liq_role_params(role, mode)
    ref_atr = _liquidity_ref_atr(current_price, atr)
    raw = max(current_price * params["box_pct"], ref_atr * params["atr_mult"])
    return min(max(raw, current_price * params["min_box_pct"]), current_price * params["max_box_pct"])


def _fractal_swing_points(
    df: pd.DataFrame | None,
    side: str,
    lookback: int | None = None,
    m: int = 2,
) -> list[dict]:
    if df is None or df.empty:
        return []
    data = df.tail(lookback).reset_index(drop=True) if lookback else df.reset_index(drop=True)
    if len(data) < m * 2 + 1:
        return []
    col = "high" if side == "high" else "low"
    points: list[dict] = []
    for i in range(m, len(data) - m):
        price = float(data.loc[i, col])
        left = data.loc[i - m:i - 1, col]
        right = data.loc[i + 1:i + m, col]
        if side == "high":
            is_swing = price > float(left.max()) and price >= float(right.max())
        else:
            is_swing = price < float(left.min()) and price <= float(right.min())
        if not is_swing:
            continue
        vol_ratio = _safe_float(data.loc[i].get("vol_ratio"), 1.0) or 1.0
        points.append({
            "price": price,
            "index": int(i),
            "time": data.loc[i].get("timestamp"),
            "volume_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else 1.0,
            "kind": "fractal",
        })
    return points


def _equal_touch_score(data: pd.DataFrame | None, side: str, level: float, tol: float) -> int:
    if data is None or data.empty:
        return 0
    col = "high" if side == "high" else "low"
    vals = data[col].astype(float)
    return int(((vals - level).abs() <= tol).sum())


def _sweep_stats_against_level(
    sweep_df: pd.DataFrame | None,
    side: str,
    level: float,
    tol: float,
) -> tuple[int, float | None]:
    if sweep_df is None or sweep_df.empty:
        return 0, None
    sweeps = 0
    vols: list[float] = []
    data = sweep_df.tail(120).reset_index(drop=True)
    for _, row in data.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        upper_wick, lower_wick, body_pct, _rng = _candle_wick_stats(row)
        vol_ratio = _safe_float(row.get("vol_ratio"), 1.0) or 1.0
        if side == "high":
            # Quét short-liq: chọc lên trên swing high rồi đóng lại dưới level.
            swept = high >= level + tol * 0.25 and close < level and upper_wick >= 0.28 and upper_wick >= body_pct * 0.7
        else:
            # Quét long-liq: chọc xuống dưới swing low rồi đóng lại trên level.
            swept = low <= level - tol * 0.25 and close > level and lower_wick >= 0.28 and lower_wick >= body_pct * 0.7
        if swept:
            sweeps += 1
            if np.isfinite(vol_ratio):
                vols.append(float(vol_ratio))
    return sweeps, (float(np.mean(vols)) if vols else None)


def _cluster_liq_levels(points: list[dict], current_price: float, atr: float | None, role: str, mode: str) -> list[list[dict]]:
    if not points:
        return []
    tol = _liquidity_tolerance(current_price, atr, role, mode)
    clusters: list[list[dict]] = []
    for point in sorted(points, key=lambda p: float(p["price"])):
        if not clusters:
            clusters.append([point])
            continue
        cur = clusters[-1]
        center = sum(float(p["price"]) for p in cur) / len(cur)
        if abs(float(point["price"]) - center) <= tol:
            cur.append(point)
        else:
            clusters.append([point])
    return clusters


def _liq_zone_from_level(level_low: float, level_high: float, side: str, width: float) -> tuple[float, float]:
    # Vùng thanh khoản nằm NGOÀI level, không bao quanh current price như support/resistance.
    if side == "low":
        top = level_high
        return top - width, top
    bottom = level_low
    return bottom, bottom + width


def _score_liq_cluster(
    cluster: list[dict],
    data: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    current_price: float,
    atr: float | None,
    side: str,
    role: str,
    mode: str,
) -> dict:
    prices = [float(p["price"]) for p in cluster]
    level_low, level_high = min(prices), max(prices)
    level = sum(prices) / len(prices)
    tol = _liquidity_tolerance(current_price, atr, role, mode)
    width = _liq_box_width(current_price, atr, role, mode)

    # Nếu cluster bị rộng bất thường thì không biến nguyên cụm thành zone rộng.
    # Chỉ lấy cạnh ngoài gần stop-pool nhất để tránh output kiểu 62,620–62,820.
    max_level_span = max(tol * 1.65, current_price * _liq_role_params(role, mode)["max_box_pct"] * 0.35)
    if (level_high - level_low) > max_level_span:
        if side == "low":
            level_low = level_high = max(prices)  # đáy cao nhất gần giá hơn là stop-pool gần nhất bên dưới
        else:
            level_low = level_high = min(prices)  # đỉnh thấp nhất gần giá hơn là stop-pool gần nhất bên trên
        level = level_low

    zone_low, zone_high = _liq_zone_from_level(level_low, level_high, side, width)
    center = (zone_low + zone_high) / 2.0
    ref_atr = _liquidity_ref_atr(current_price, atr)
    distance_atr = abs(level - current_price) / max(ref_atr, 1e-12)

    touches = _equal_touch_score(data, side, level, tol)
    sweep_count, sweep_vol = _sweep_stats_against_level(sweep_df, side, level, tol)
    vol_values = [float(p.get("volume_ratio", 1.0)) for p in cluster if np.isfinite(float(p.get("volume_ratio", 1.0)))]
    avg_vol = float(np.mean(vol_values)) if vol_values else None
    if sweep_vol is not None:
        avg_vol = max(avg_vol or 0.0, sweep_vol)

    latest_idx = max(int(p.get("index", 0)) for p in cluster)
    total_len = len(data) if data is not None and not data.empty else latest_idx + 1
    age_ratio = max((total_len - 1 - latest_idx) / max(total_len, 1), 0.0)
    recency_score = 1.25 * (1.0 - min(age_ratio, 1.0))

    params = _liq_role_params(role, mode)
    target = params["target_atr"]
    if role == "near":
        distance_score = max(0.0, 1.0 - distance_atr / 3.5) * 1.6
    elif role == "deep":
        distance_score = min(distance_atr / max(target, 0.1), 1.4) * 0.9
    else:
        distance_score = max(0.0, 1.0 - abs(distance_atr - target) / 3.5) * 1.1

    side_ok = level <= current_price if side == "low" else level >= current_price
    if not side_ok:
        distance_score -= 3.0

    vol_score = 0.0
    if avg_vol is not None:
        vol_score = min(max(avg_vol - 0.8, 0.0), 2.2) * 0.7

    score = (
        min(len(cluster), 4) * 0.90
        + min(touches, 6) * 0.55
        + min(sweep_count, 4) * 1.15
        + vol_score
        + recency_score
        + distance_score
    )

    strength = "mạnh" if (avg_vol or 0) >= 1.5 or sweep_count >= 2 or touches >= 4 else "vừa"
    if touches <= 1 and sweep_count == 0 and (avg_vol or 1.0) < 1.1:
        strength = "yếu"

    return {
        "low": zone_low,
        "high": zone_high,
        "center": center,
        "level": level,
        "score": score,
        "hits": max(len(cluster), touches),
        "touches": touches,
        "sweeps": sweep_count,
        "vol_ratio": avg_vol,
        "distance_atr": distance_atr,
        "strength": strength,
        "role": role,
        "level_span": level_high - level_low,
        "width": zone_high - zone_low,
    }


def _zone_for_liq_pools(
    level_df: pd.DataFrame | None,
    sweep_df: pd.DataFrame | None,
    current_price: float,
    atr: float | None,
    side: str,
    role: str,
    mode: str,
    lookback: int | None,
    m: int = 2,
) -> tuple[float | None, float | None, int, dict]:
    data = level_df.tail(lookback).reset_index(drop=True) if level_df is not None and not level_df.empty else None
    if data is None or data.empty:
        return None, None, 0, _zone_meta_default(role)

    points = _fractal_swing_points(data, side, lookback=None, m=m)
    if not points:
        # Fallback chỉ lấy một cực trị còn đúng phía, vẫn tạo box ngoài cực trị.
        col = "low" if side == "low" else "high"
        idx = int(data[col].idxmin() if side == "low" else data[col].idxmax())
        points = [{
            "price": float(data.loc[idx, col]),
            "index": idx,
            "time": data.loc[idx].get("timestamp"),
            "volume_ratio": _safe_float(data.loc[idx].get("vol_ratio"), 1.0) or 1.0,
            "kind": "extreme_fallback",
        }]

    # Chỉ giữ level đúng phía. Vùng dưới là stop pool dưới swing low; vùng trên là stop pool trên swing high.
    if side == "low":
        points = [p for p in points if float(p["price"]) <= current_price]
    else:
        points = [p for p in points if float(p["price"]) >= current_price]
    if not points:
        return None, None, 0, _zone_meta_default(role)

    clusters = _cluster_liq_levels(points, current_price, atr, role, mode)
    scored = [_score_liq_cluster(c, data, sweep_df, current_price, atr, side, role, mode) for c in clusters]
    # Loại zone cực rộng còn sót lại vì dữ liệu nhiễu. Với BTC scalp width > cap là không dùng.
    max_width = current_price * _liq_role_params(role, mode)["max_box_pct"] * 1.10
    scored = [s for s in scored if (s["high"] - s["low"]) <= max_width]
    if not scored:
        return None, None, 0, _zone_meta_default(role)

    best = max(scored, key=lambda x: x["score"])
    meta = {
        "role": role,
        "touches": int(best["touches"]),
        "sweeps": int(best["sweeps"]),
        "vol_ratio": best["vol_ratio"],
        "score": round(float(best["score"]), 2),
        "distance_atr": round(float(best["distance_atr"]), 2),
        "strength": best["strength"],
        "swing_level": round(float(best["level"]), 2),
        "zone_width": round(float(best["width"]), 2),
        "method": "fractal_swing_pool",
    }
    meta["side_state"] = _zone_side_state((best["low"], best["high"], int(best["hits"]), meta), current_price)
    return best["low"], best["high"], int(best["hits"]), meta


def _first_valid_df(*dfs: pd.DataFrame | None) -> pd.DataFrame | None:
    for df in dfs:
        if df is not None and not df.empty:
            return df
    return None


def _liquidity_zones_by_windows(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float,
) -> dict:
    """Vùng thanh khoản V5: stop-pool ngoài swing, không phải band giá đang kẹt.

    Theo gợi ý OHLCV-only: khung lớn xác định level, khung nhỏ xác nhận sweep.
    SCALP: 1H cho vùng gần, 4H cho vùng chính/sâu, 15M chỉ kiểm tra sweep.
    SWING: H4 cho vùng gần, D1 cho vùng chính, W1/D1 cho vùng sâu, 1H/4H chỉ kiểm tra sweep.
    """
    if mode == "short":
        sweep_df = timeframe_data.get("15M")
        near_df = _first_valid_df(timeframe_data.get("1H"), timeframe_data.get("15M"))
        main_df = _first_valid_df(timeframe_data.get("4H"), timeframe_data.get("1H"))
        deep_df = _first_valid_df(timeframe_data.get("4H"), timeframe_data.get("1H"))
        windows = [
            ("near", "gần 1H", near_df, _current_atr(near_df) or _current_atr(sweep_df), 72, 2),
            ("main", "chính H4", main_df, _current_atr(main_df) or _current_atr(near_df), 90, 2),
            ("deep", "sâu H4", deep_df, _current_atr(deep_df) or _current_atr(main_df), 120, 3),
        ]
        calc_mode = "short"
    else:
        # SWING: level lấy từ H4/D1/W1, sweep lấy từ 1H trước rồi fallback 4H.
        # Không dùng 1H để tạo liquidity box vì sẽ khiến vùng swing bị nhiễu như scalp.
        sweep_df = _first_valid_df(timeframe_data.get("1H"), timeframe_data.get("4H"))
        near_df = timeframe_data.get("4H")
        main_df = _first_valid_df(timeframe_data.get("1D"), timeframe_data.get("4H"))
        deep_df = _first_valid_df(timeframe_data.get("1W"), timeframe_data.get("1D"), timeframe_data.get("4H"))
        windows = [
            ("near", "gần H4", near_df, _current_atr(near_df) or _current_atr(sweep_df), 120, 2),
            ("main", "chính D1", main_df, _current_atr(main_df) or _current_atr(near_df), 90, 2),
            ("deep", "sâu W1/D1", deep_df, _current_atr(deep_df) or _current_atr(main_df), 120, 3),
        ]
        calc_mode = "long"

    result: dict = {"meta": [label for _, label, *_ in windows], "liquidity_method": "fractal_swing_pool_v5"}
    for key, label, df, atr, lookback, m in windows:
        lower = _zone_for_liq_pools(df, sweep_df, current_price, atr, "low", key, calc_mode, lookback, m=m)
        upper = _zone_for_liq_pools(df, sweep_df, current_price, atr, "high", key, calc_mode, lookback, m=m)
        result[f"lower_{key}"] = lower
        result[f"upper_{key}"] = upper
        result[f"label_{key}"] = label
    return result

def _liquidity_zones(df: pd.DataFrame | None, current_price: float, atr: float | None) -> dict:
    """Legacy wrapper: giữ lại để tránh lỗi nếu còn đoạn code cũ gọi."""
    low_pivots = [p["price"] for p in _find_pivots(df, "low", 100)]
    high_pivots = [p["price"] for p in _find_pivots(df, "high", 100)]
    long_low, long_high, long_hits = _cluster_zone(low_pivots, current_price, "low", atr)
    short_low, short_high, short_hits = _cluster_zone(high_pivots, current_price, "high", atr)

    if long_low is None or long_high is None:
        long_low, long_high, _meta = _fallback_zone(df, "low", current_price, atr, 80, role="near")
        long_hits = 1 if long_low is not None else 0
    if short_low is None or short_high is None:
        short_low, short_high, _meta = _fallback_zone(df, "high", current_price, atr, 80, role="near")
        short_hits = 1 if short_low is not None else 0

    deep_long_low, deep_long_high, _meta = _fallback_zone(df, "low", current_price, atr, 150, role="deep")
    deep_short_low, deep_short_high, _meta = _fallback_zone(df, "high", current_price, atr, 150, role="deep")
    return {
        "long_near": (long_low, long_high, long_hits),
        "short_near": (short_low, short_high, short_hits),
        "long_deep": (deep_long_low, deep_long_high),
        "short_deep": (deep_short_low, deep_short_high),
    }


def _fmt_zone_tuple(zone: tuple | None, current_price: float | None = None) -> str:
    if not zone:
        return "N/A"
    low = zone[0] if len(zone) > 0 else None
    high = zone[1] if len(zone) > 1 else None
    hits = zone[2] if len(zone) > 2 else 0
    meta = zone[3] if len(zone) > 3 and isinstance(zone[3], dict) else {}
    if low is None or high is None:
        return "N/A"

    details: list[str] = []
    touches = meta.get("touches")
    sweeps = meta.get("sweeps")
    vol_ratio = meta.get("vol_ratio")
    distance_atr = meta.get("distance_atr")
    strength = meta.get("strength")
    swing_level = meta.get("swing_level")
    zone_width = meta.get("zone_width")

    if strength:
        details.append(f"{strength}")
    if swing_level is not None:
        details.append(f"level {fmt(swing_level)}")
    if touches:
        details.append(f"{int(touches)} chạm")
    elif hits:
        details.append(f"{int(hits)} điểm")
    if sweeps:
        details.append(f"quét {int(sweeps)}")
    if vol_ratio is not None and np.isfinite(vol_ratio):
        details.append(f"vol {float(vol_ratio):.2f}x")
    if distance_atr is not None and np.isfinite(distance_atr):
        details.append(f"~{float(distance_atr):.1f}ATR")
    if zone_width is not None and np.isfinite(zone_width):
        details.append(f"rộng {fmt(zone_width)}")
    if current_price is not None and float(low) <= current_price <= float(high):
        details.append("đang chạm giá")
    if meta.get("fallback"):
        details.append("fallback cực trị")

    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{fmt(low)}–{fmt(high)}{detail_text}"

def _zones_have_meaningful_overlap(a: tuple | None, b: tuple | None) -> bool:
    if not a or not b:
        return False
    a_low, a_high = a[0], a[1]
    b_low, b_high = b[0], b[1]
    if a_low is None or a_high is None or b_low is None or b_high is None:
        return False
    overlap = max(0.0, min(float(a_high), float(b_high)) - max(float(a_low), float(b_low)))
    width = max(min(float(a_high) - float(a_low), float(b_high) - float(b_low)), 1e-12)
    return overlap / width >= 0.55


def _liquidity_overlap_note(zones: dict, side: str) -> str:
    pairs = [
        ("gần", "chính", zones.get(f"{side}_near"), zones.get(f"{side}_main")),
        ("chính", "sâu", zones.get(f"{side}_main"), zones.get(f"{side}_deep")),
        ("gần", "sâu", zones.get(f"{side}_near"), zones.get(f"{side}_deep")),
    ]
    overlapped = [f"{a}/{b}" for a, b, za, zb in pairs if _zones_have_meaningful_overlap(za, zb)]
    if not overlapped:
        return ""
    return f" | Lưu ý: vùng {', '.join(overlapped)} đang trùng mạnh, xem là cùng một cụm thanh khoản thay vì 3 mục tiêu riêng."


def _format_liquidity_window_line(prefix: str, zones: dict, side: str, current_price: float | None = None) -> str:
    # side = lower hoặc upper
    return (
        f"- {prefix}: "
        f"{zones.get('label_near', 'gần')} {_fmt_zone_tuple(zones.get(f'{side}_near'), current_price)}; "
        f"{zones.get('label_main', 'chính')} {_fmt_zone_tuple(zones.get(f'{side}_main'), current_price)}; "
        f"{zones.get('label_deep', 'sâu')} {_fmt_zone_tuple(zones.get(f'{side}_deep'), current_price)}"
        f"{_liquidity_overlap_note(zones, side)}"
    )

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
        # Risk floor chỉ là mốc chống nhiễu để AI không đặt SL quá sát.
        # Bản cũ dùng 2.5 ATR 15M / 1.2 ATR 1H / 0.6% giá nên Entry-SL/TP dễ bị quá rộng.
        return max(atr_main * 1.6, atr_confirm * 0.7, current_price * 0.0035)
    atr_main = _current_atr(timeframe_data.get("4H")) or 0
    atr_confirm = _current_atr(timeframe_data.get("1D")) or 0
    # Swing cần rộng hơn scalp, nhưng vẫn tránh ép SL/TP quá xa.
    return max(atr_main * 1.6, atr_confirm * 0.45, current_price * 0.015)




def _analysis_row(df: pd.DataFrame | None):
    """
    Dùng nến đã đóng gần nhất để đọc indicator/volume.

    Binance thường trả kèm nến hiện tại đang chạy; volume của nến này rất thấp
    nếu vừa mở nến, dễ làm Claude hiểu nhầm là thanh khoản yếu và chọn NO_TRADE.
    Vì vậy indicator/regime/snapshot dùng nến -2 khi có đủ dữ liệu.
    """
    if df is None or df.empty:
        return None
    return df.iloc[-2] if len(df) >= 2 else df.iloc[-1]


REGIME_LABEL_VI = {
    "EMA_TANG": "EMA nghiêng tăng",
    "EMA_GIAM": "EMA nghiêng giảm",
    "EMA_DAN_XEN": "EMA đan xen",
    "TRENDING_UP": "xu hướng tăng rõ",
    "TRENDING_DOWN": "xu hướng giảm rõ",
    "RANGE_CHOPPY": "đi ngang/nhiễu",
    "MIXED_TRANSITION": "trạng thái chuyển pha",
    "BEAR_TREND": "xu hướng giảm",
    "BULL_TREND": "xu hướng tăng",
    "MIXED_UNCLEAR": "chưa rõ xu hướng",
    "HIGH_VOLATILITY": "biến động mạnh",
    "LOW_VOLATILITY": "biến động thấp",
    "NORMAL_VOLATILITY": "biến động bình thường",
    "HIGH_VOLUME": "khối lượng cao",
    "LOW_VOLUME": "khối lượng thấp",
    "NORMAL_VOLUME": "khối lượng bình thường",
    "LOW_LIQUIDITY_RISK": "rủi ro thanh khoản thấp",
    "HIGH_VOLATILITY_RISK": "rủi ro biến động mạnh",
    "LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE": "khung nhỏ đang hồi ngược cấu trúc lớn",
}


def _label_vi(code: str) -> str:
    return REGIME_LABEL_VI.get(code, code)


def _ema_state_from_last(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "N/A"
    last = _analysis_row(df)
    if last is None:
        return "N/A"
    if last["ema_7"] > last["ema_25"] > last["ema_50"]:
        return "EMA_TANG"
    if last["ema_7"] < last["ema_25"] < last["ema_50"]:
        return "EMA_GIAM"
    return "EMA_DAN_XEN"


def _timeframe_regime_details(label: str, df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {
            "label": label,
            "trend_tag": "N/A",
            "vol_tag": "N/A",
            "volume_tag": "N/A",
            "ema_state": "N/A",
            "text": f"{label}: không đủ dữ liệu",
        }
    last = _analysis_row(df)
    if last is None:
        return {
            "label": label,
            "trend_tag": "N/A",
            "vol_tag": "N/A",
            "volume_tag": "N/A",
            "ema_state": "N/A",
            "text": f"{label}: không đủ dữ liệu",
        }
    close = float(last["close"])
    ema_state = _ema_state_from_last(df)
    rsi = _safe_float(last.get("rsi_14"), 50.0) or 50.0
    atr_pct = _safe_float(last.get("atr_pct"), 0.0) or 0.0
    vol_ratio = _safe_float(last.get("vol_ratio"), 1.0) or 1.0
    ema_spread_pct = abs(float(last["ema_7"]) - float(last["ema_50"])) / max(close, 1e-12) * 100

    if ema_state == "EMA_TANG" and close >= float(last["ema_25"]) and rsi >= 52:
        trend_tag = "TRENDING_UP"
    elif ema_state == "EMA_GIAM" and close <= float(last["ema_25"]) and rsi <= 48:
        trend_tag = "TRENDING_DOWN"
    elif ema_spread_pct < 0.20 or 42 <= rsi <= 58:
        trend_tag = "RANGE_CHOPPY"
    else:
        trend_tag = "MIXED_TRANSITION"

    if atr_pct >= 1.20:
        vol_tag = "HIGH_VOLATILITY"
    elif atr_pct <= 0.25:
        vol_tag = "LOW_VOLATILITY"
    else:
        vol_tag = "NORMAL_VOLATILITY"

    if vol_ratio >= 1.50:
        volume_tag = "HIGH_VOLUME"
    elif vol_ratio <= 0.70:
        volume_tag = "LOW_VOLUME"
    else:
        volume_tag = "NORMAL_VOLUME"

    return {
        "label": label,
        "trend_tag": trend_tag,
        "vol_tag": vol_tag,
        "volume_tag": volume_tag,
        "ema_state": ema_state,
        "text": (
            f"{label}: {_label_vi(trend_tag)}, {_label_vi(vol_tag)}, {_label_vi(volume_tag)}; "
            f"EMA={_label_vi(ema_state)}, RSI14={fmt(rsi,1)}, ATR%={fmt(atr_pct,2)}, Vol={fmt(vol_ratio,2)}x"
        ),
    }


def _timeframe_regime(label: str, df: pd.DataFrame | None) -> str:
    return _timeframe_regime_details(label, df)["text"]


def build_market_regime_block(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    main_label, structure_label, big_label = _mode_labels(mode)
    main_state = _timeframe_regime_details(main_label, timeframe_data.get(main_label))
    structure_state = _timeframe_regime_details(structure_label, timeframe_data.get(structure_label))
    big_state = _timeframe_regime_details(big_label, timeframe_data.get(big_label))

    states = [main_state, structure_state, big_state]
    down_count = sum(s["trend_tag"] == "TRENDING_DOWN" for s in states)
    up_count = sum(s["trend_tag"] == "TRENDING_UP" for s in states)
    range_count = sum(s["trend_tag"] == "RANGE_CHOPPY" for s in states)
    low_volume_count = sum(s["volume_tag"] == "LOW_VOLUME" for s in states)
    high_vol_count = sum(s["vol_tag"] == "HIGH_VOLATILITY" for s in states)

    if down_count >= 2:
        overall_code = "BEAR_TREND"
    elif up_count >= 2:
        overall_code = "BULL_TREND"
    elif range_count >= 2:
        overall_code = "RANGE_CHOPPY"
    else:
        overall_code = "MIXED_UNCLEAR"

    modifiers = []
    if low_volume_count >= 2:
        modifiers.append("LOW_LIQUIDITY_RISK")
    if high_vol_count >= 2:
        modifiers.append("HIGH_VOLATILITY_RISK")
    if (main_state["trend_tag"] == "TRENDING_UP" and structure_state["trend_tag"] == "TRENDING_DOWN") or (
        main_state["trend_tag"] == "TRENDING_DOWN" and structure_state["trend_tag"] == "TRENDING_UP"
    ):
        modifiers.append("LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE")
    modifier_text = ", ".join(_label_vi(m) for m in modifiers) if modifiers else "không có ghi chú rủi ro lớn"

    return "\n".join([
        "Phân loại thị trường do Python:",
        f"- Xu hướng chính: {_label_vi(overall_code)}; ghi chú: {modifier_text}",
        f"- {main_state['text']}",
        f"- {structure_state['text']}",
        f"- {big_state['text']}",
        "- Cách dùng: đi ngang/nhiễu, chưa rõ xu hướng hoặc thanh khoản thấp là cảnh báo rủi ro. Không cố tạo LONG/SHORT nếu lợi thế không rõ; chỉ dùng lệnh chờ khi vùng Entry thật sự đẹp và có lý do kỹ thuật rõ ràng.",
    ])


def _format_candle_compact(row) -> str:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    volume = float(row.get("volume", 0) or 0)
    rng = max(high - low, 1e-12)
    body_pct = abs(close - open_) / rng * 100
    upper_pct = (high - max(open_, close)) / rng * 100
    lower_pct = (min(open_, close) - low) / rng * 100
    direction = "xanh" if close > open_ else "đỏ" if close < open_ else "doji"
    taker_ratio = None
    try:
        if volume > 0:
            taker_ratio = float(row.get("taker_buy_volume", 0) or 0) / volume * 100
    except Exception:
        taker_ratio = None
    taker_text = f" TakerBuy:{fmt(taker_ratio,1)}%" if taker_ratio is not None else ""
    return (
        f"{str(row['timestamp'])[:16]} {direction} "
        f"O:{fmt(open_)} H:{fmt(high)} L:{fmt(low)} C:{fmt(close)} "
        f"Body:{body_pct:.0f}% U:{upper_pct:.0f}% D:{lower_pct:.0f}% "
        f"Vol:{fmt(row.get('vol_ratio'),2)}x{taker_text}"
    )


def build_raw_candle_context(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    """Gửi thêm nến thô có body/wick để Sonnet đọc hành vi giá, nhưng vẫn giữ gọn."""
    main_label, structure_label, _ = _mode_labels(mode)
    blocks = ["RAW_CANDLE_CONTEXT_CHON_LOC:"]
    for label, n in [(main_label, 24), (structure_label, 12)]:
        df = timeframe_data.get(label)
        if df is None or df.empty:
            blocks.append(f"- {label}: Không đủ dữ liệu nến thô.")
            continue
        rows = ["  " + _format_candle_compact(row) for _, row in df.tail(n).iterrows()]
        blocks.append(f"- {label}: {n} nến gần nhất, dùng để đọc phá giả/rút râu/đuối lực:")
        blocks.extend(rows)
    return "\n".join(blocks)


def build_feature_engineering_block(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    main_label, structure_label, big_label = _mode_labels(mode)
    price = current_price or _last_close_from_data(timeframe_data)
    if price is None:
        return "Dữ liệu kỹ thuật: Không đủ dữ liệu để tính cấu trúc, Fibonacci, ATR và vùng quét. Không được tự bịa các phần này."

    main_df = timeframe_data.get(main_label)
    structure_df = timeframe_data.get(structure_label)
    if structure_df is None or structure_df.empty:
        structure_df = main_df
    atr_main = _current_atr(main_df)
    atr_structure = _current_atr(structure_df)
    zones = _liquidity_zones_by_windows(timeframe_data, mode, price)
    structure = _structure_info(structure_df, price)
    risk = _risk_floor(timeframe_data, mode, price)

    fib = structure.get("fib", {})

    lines = [
        "Dữ liệu kỹ thuật do Python tính sẵn:",
        f"- Mode: {'SCALP' if mode == 'short' else 'SWING'} | Khung vào lệnh: {main_label} | Khung cấu trúc: {structure_label} | Khung lớn: {big_label}",
        build_market_regime_block(timeframe_data, mode),
        f"- ATR14 {main_label}: {fmt(atr_main)} | ATR14 {structure_label}: {fmt(atr_structure)} | Rủi ro tham chiếu: {fmt(risk)} USDT",
        f"- Chuỗi nến {main_label}: {_consecutive_candles(main_df)} | Nến cuối: {_wick_body_info(main_df)}",
        f"- Cấu trúc {structure_label}: {structure.get('trend', 'N/A')}; đỉnh/đáy gần {fmt(structure.get('recent_low'))}–{fmt(structure.get('recent_high'))}; biên lớn {fmt(structure.get('major_low'))}–{fmt(structure.get('major_high'))}",
        f"- Fibonacci {structure_label}: 0.382={fmt(fib.get('0.382'))}; 0.5={fmt(fib.get('0.5'))}; 0.618={fmt(fib.get('0.618'))}",
        _format_liquidity_window_line("Vùng thanh khoản dưới giá ước lượng", zones, "lower", price),
        _format_liquidity_window_line("Vùng thanh khoản trên giá ước lượng", zones, "upper", price),
        "- Vai trò vùng quét: Entry ưu tiên dùng vùng gần/chính nếu hợp xu hướng và có xác nhận. Với SCALP không dùng vùng sâu làm Entry mặc định; với SWING, vùng sâu chỉ dùng khi đó là pullback lớn có xác nhận H4/1H rõ. TP dùng vùng đối diện: TP1 ưu tiên vùng đối diện gần/chính, TP2 có thể dùng vùng đối diện chính/sâu. SL đặt ngoài vùng Entry + buffer ATR, không đặt ngay trong vùng quét.",
        "- Quy tắc rủi ro: AI tự lập Entry/SL/TP. Rủi ro tham chiếu là mốc chống nhiễu, không phải lệnh bắt buộc; Entry–SL có thể thấp hơn một chút nếu có đỉnh/đáy vô hiệu rõ. TP1 nên khoảng >= 0.8R, TP2 nên khoảng >= 1.3R, không kéo TP quá xa chỉ để đẹp tỷ lệ.",
        "- Ghi chú: Vùng quét là vùng thanh khoản kỹ thuật ước lượng theo cửa sổ thời gian, không phải dữ liệu thanh lý thật hay liquidation heatmap. Block này là bản đồ kỹ thuật, không phải lệnh giao dịch chốt sẵn.",
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
    zones = _liquidity_zones_by_windows(timeframe_data, mode, price)
    structure = _structure_info(structure_df, price)
    risk = _risk_floor(timeframe_data, mode, price)
    fib = structure.get("fib", {})

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
            f"RSI14 {fmt(last['rsi_14'], 1)}, {macd_momentum_text(last['macd_hist'])}, "
            f"ATR14 {fmt(last.get('atr_14'))}, vol {fmt(last['vol_ratio'], 2)}x"
        )

    parts = [
        f"Mode {'SCALP' if mode == 'short' else 'SWING'}; frame entry {main_label}, structure {structure_label}, big {big_label}",
        build_market_regime_block(timeframe_data, mode).replace("\n", " / "),
        compact_tf(main_label, main_df),
        compact_tf(structure_label, structure_df),
        compact_tf(big_label, big_df),
        f"Cấu trúc {structure_label}: {structure.get('trend', 'N/A')}; đỉnh/đáy gần {fmt(structure.get('recent_low'))}-{fmt(structure.get('recent_high'))}; biên lớn {fmt(structure.get('major_low'))}-{fmt(structure.get('major_high'))}",
        f"Fib {structure_label}: 0.382 {fmt(fib.get('0.382'))}, 0.5 {fmt(fib.get('0.5'))}, 0.618 {fmt(fib.get('0.618'))}",
        f"Vùng dưới: {zones.get('label_near')} {_fmt_zone_tuple(zones.get('lower_near'), price)}; {zones.get('label_main')} {_fmt_zone_tuple(zones.get('lower_main'), price)}; {zones.get('label_deep')} {_fmt_zone_tuple(zones.get('lower_deep'), price)}",
        f"Vùng trên: {zones.get('label_near')} {_fmt_zone_tuple(zones.get('upper_near'), price)}; {zones.get('label_main')} {_fmt_zone_tuple(zones.get('upper_main'), price)}; {zones.get('label_deep')} {_fmt_zone_tuple(zones.get('upper_deep'), price)}",
        f"ATR/risk: ATR {main_label} {fmt(atr_main)}, ATR {structure_label} {fmt(atr_structure)}, rủi ro tham chiếu {fmt(risk)}",
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


def macd_momentum_text(macd_hist: float | None, decimals: int = 4) -> str:
    """Diễn giải MACD histogram bằng tiếng Việt để không lộ jargon `Hist` ra output."""
    if macd_hist is None or (isinstance(macd_hist, float) and np.isnan(macd_hist)):
        return "động lượng MACD N/A"
    value = fmt(macd_hist, decimals)
    if macd_hist > 0:
        return f"động lượng MACD dương {value}"
    if macd_hist < 0:
        return f"động lượng MACD âm {value}"
    return "động lượng MACD trung tính 0"


def summarize_timeframe(label: str, df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return f"\nKHUNG {label}: Không đủ dữ liệu.\n"

    last = _analysis_row(df)
    if last is None:
        return f"\nKHUNG {label}: Không đủ dữ liệu.\n"
    last_pos = df.index.get_loc(last.name) if hasattr(last, "name") else len(df) - 1
    prev = df.iloc[max(0, int(last_pos) - 1)] if len(df) >= 2 else last
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
        f"  MACD={fmt(last['macd_line'],4)} Signal={fmt(last['macd_signal'],4)}; {macd_momentum_text(last['macd_hist'])} → {macd_dir}{macd_cross}",
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

        last = _analysis_row(df)
        if last is None:
            lines.append(f"{label}: no data")
            continue
        ema_align = "mixed"
        if last["ema_7"] > last["ema_25"] > last["ema_50"]:
            ema_align = "bullish"
        elif last["ema_7"] < last["ema_25"] < last["ema_50"]:
            ema_align = "bearish"

        lines.append(
            f"{label}: close={fmt(last['close'])}, EMA={ema_align} "
            f"(7={fmt(last['ema_7'])},25={fmt(last['ema_25'])},50={fmt(last['ema_50'])}), "
            f"RSI14={fmt(last['rsi_14'], 1)}, {macd_momentum_text(last['macd_hist'])}, "
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
    open_signal_context: str | None = None,
) -> str:
    mode_label = "SCALP (ngắn hạn)" if mode == "short" else "SWING (dài hạn)"
    focus      = (
        "Dùng 15M để timing entry, 1H để xác nhận momentum, 4H để xác định xu hướng chính."
        if mode == "short" else
        "Dùng 4H để timing entry, 1D để xác nhận xu hướng, 1W để xác định big picture."
    )

    history_block = format_prediction_history(history)
    open_signal_context = open_signal_context or "KẾ HOẠCH ĐANG MỞ: Không có kế hoạch đang chờ/đã khớp cho user này ở cùng coin và mode."
    tf_blocks     = "".join(summarize_timeframe(lbl, df) for lbl, df in timeframe_data.items())
    raw_candle_block = build_raw_candle_context(timeframe_data, mode)
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
{open_signal_context}
═══════════════════════════════
{raw_candle_block}
═══════════════════════════════
{tf_blocks}
═══════════════════════════════

Yêu cầu:
1. Python chỉ cung cấp dữ liệu cứng: EMA/RSI/MACD/ATR, market regime, cấu trúc, Fibonacci, vùng quét thanh khoản ước lượng, raw candle context, rủi ro tham chiếu. Không có kế hoạch LONG/SHORT chốt sẵn.
2. Model phải tự phân tích và tự lập Entry/SL/TP dựa trên dữ liệu cứng đó. Không được tự tạo thêm Fibonacci/vùng quét nếu block Python ghi N/A hoặc không đủ dữ liệu.
3. Trước khi quyết định, hãy so sánh NỘI BỘ 3 lựa chọn LONG / SHORT / NO_TRADE theo xu hướng đa khung, vị trí giá, vùng quét ước lượng theo cửa sổ thời gian, Fibonacci, nến thô, volume và lịch sử cùng user. Không in bảng so sánh này ra user.
4. Chỉ chọn LONG hoặc SHORT khi một hướng có lợi thế rõ hơn hướng còn lại, Entry hợp lý và risk/reward đạt yêu cầu. Nếu thị trường nhiễu, xác suất chỉ ngang nhau, vùng vào lệnh không rõ, hoặc Entry/SL/TP bị gượng ép → chọn NO_TRADE. Không dùng NO_TRADE chỉ vì giá chưa chạm Entry; chỉ dùng lệnh chờ khi vùng Entry thật sự đẹp và có lý do kỹ thuật rõ ràng.
5. Cách dùng vùng quét: Entry ưu tiên vùng gần/chính nếu hợp hướng setup và có xác nhận. Vùng sâu chủ yếu dùng cho TP2, SL, hoặc cảnh báo vùng nguy hiểm; không dùng vùng sâu làm Entry scalp mặc định trừ khi có cú quét sâu rất rõ.
6. TP dùng vùng đối diện: với LONG nhìn vùng thanh khoản trên, với SHORT nhìn vùng thanh khoản dưới. TP1 ưu tiên vùng đối diện gần/chính; TP2 có thể dùng vùng đối diện chính/sâu. SL đặt ngoài vùng Entry + buffer ATR, không đặt ngay trong vùng quét.
7. Không mặc định mọi tín hiệu thành lệnh chờ. Nếu giá hiện tại đang nằm trong vùng Entry hợp lý và tín hiệu xác nhận đã đủ, hãy đặt Entry bao quanh/sát giá hiện tại và ghi “Có thể vào ngay trong vùng Entry...”.
8. Nếu giá hiện tại chưa vào vùng Entry hoặc còn thiếu xác nhận, mới ghi “Lệnh chờ, chưa vào ngay...” và nêu rõ điều kiện chờ.
9. Nếu chọn LONG/SHORT: Entry/SL/TP phải hợp logic với hướng giao dịch và tham chiếu ATR/giá. Không đặt SL quá sát, nhưng cũng không kéo SL/TP quá xa chỉ để đạt tỷ lệ lời/lỗ đẹp.
10. Nếu chọn NO_TRADE: không cần Entry/SL/TP; trả quyết định NO_TRADE và lý do ngắn. Python sẽ không gửi plan đó thành tín hiệu. Được chọn NO_TRADE khi lợi thế chưa đủ rõ, kể cả khi vẫn có thể vẽ ra một vùng Entry hợp lệ nhưng kèo không đáng vào.
11. Đọc kỹ RECENT LEARNING SUMMARY, đặc biệt Decision why, Outcome, Market then và Feature then, nhưng không hiện mục “Nhìn lại lịch sử” trong câu trả lời.
12. Đọc kỹ KẾ HOẠCH ĐANG MỞ nếu có. Không được hiểu vùng Entry của một lệnh chờ LONG là mục tiêu TP cho lệnh SHORT ngược lại, hoặc vùng Entry của lệnh chờ SHORT là mục tiêu TP cho lệnh LONG ngược lại.
13. Nếu đang có kế hoạch cũ PENDING_ENTRY mà giá đã chạy xa khỏi Entry theo đúng hướng dự báo, không được đuổi giá chỉ vì giá chạy. Chỉ cho vào ngay khi có vùng Entry mới bao quanh giá hiện tại và xác nhận rõ; nếu không thì NO_TRADE hoặc chờ kiểm tra lại.
14. Nếu kế hoạch mới thay thế kế hoạch cũ, ghi ngắn trong “📊 Kịch bản chính” lý do kế hoạch cũ bị hủy/thay thế.
15. Không copy phân tích cũ. Chỉ dùng summary để tránh lặp lại lỗi.
16. QUYẾT ĐỊNH cuối cùng chỉ được là LONG, SHORT hoặc NO_TRADE. Không dùng “CHỜ” làm quyết định cuối cùng.
"""


# ─── Tóm tắt reasoning bằng call Haiku thứ 2 (rất ngắn, rẻ) ─────────────────

def get_ai_api_key() -> str | None:
    """Trả về API key theo provider hiện tại."""
    if AI_PROVIDER in ("openrouter", "glm", "zai", "z.ai"):
        return OPENROUTER_API_KEY
    return ANTHROPIC_API_KEY


def get_ai_model_name() -> str:
    if AI_PROVIDER in ("openrouter", "glm", "zai", "z.ai"):
        return OPENROUTER_MODEL
    return CLAUDE_MODEL


def ensure_ai_config() -> None:
    if AI_PROVIDER in ("openrouter", "glm", "zai", "z.ai"):
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY. Set AI_PROVIDER=openrouter and OPENROUTER_API_KEY in Railway variables.")
        return
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env/Railway variables.")


def _anthropic_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
        "timeout": timeout,
    }
    if system:
        kwargs["system"] = system

    # Claude Sonnet 5 / Opus 4.x dùng output_config.effort để điều khiển mức suy luận.
    # high là mặc định của API; max là mức sâu nhất, phù hợp phân tích chính của Teopard.
    if reasoning_effort is None:
        effective_effort = (ANTHROPIC_EFFORT or "").strip().lower()
    else:
        effective_effort = (reasoning_effort or "").strip().lower()

    if effective_effort in {"max", "xhigh", "high", "medium", "low"}:
        kwargs["output_config"] = {"effort": effective_effort}

    response = client.messages.create(**kwargs)
    return {
        "text": "".join(b.text for b in response.content if hasattr(b, "text")),
        "stop_reason": getattr(response, "stop_reason", None),
        "usage": getattr(response, "usage", None),
        "effort": effective_effort or None,
    }


def _openrouter_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Gọi GLM/OpenRouter bằng Chat Completions API, không cần thêm thư viện openai.
    OpenRouter khuyến nghị max_completion_tokens thay cho max_tokens."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-OpenRouter-Title": OPENROUTER_APP_NAME,
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": payload_messages,
        "max_completion_tokens": max_tokens,
    }

    # Một số provider có hỗ trợ reasoning effort.
    # - Phân tích chính: dùng OPENROUTER_REASONING_EFFORT nếu có set.
    # - Nếu đang dùng GLM qua OpenRouter mà Railway chưa set biến này, mặc định dùng xhigh
    #   để GLM có ngân sách suy luận sâu hơn cho Entry/SL/TP.
    # - Summary: truyền reasoning_effort="off" để không đốt hết 120-600 token vào reasoning ẩn.
    if reasoning_effort is None:
        default_effort = "xhigh" if "glm" in (OPENROUTER_MODEL or "").lower() else ""
        effective_reasoning_effort = os.getenv("OPENROUTER_REASONING_EFFORT", default_effort).strip()
    else:
        effective_reasoning_effort = (reasoning_effort or "").strip()

    if effective_reasoning_effort.lower() not in ("", "off", "none", "false", "0"):
        payload["reasoning"] = {"effort": effective_reasoning_effort}

    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"OpenRouter API error: {r.status_code} - {r.text[:1000]}") from exc

    data = r.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return {
        "text": content,
        "stop_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "effort": effective_reasoning_effort or None,
    }


def llm_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    ensure_ai_config()
    if AI_PROVIDER in ("openrouter", "glm", "zai", "z.ai"):
        return _openrouter_create_once(system, messages, max_tokens, timeout, reasoning_effort=reasoning_effort)
    return _anthropic_create_once(system, messages, max_tokens, timeout, reasoning_effort=reasoning_effort)


def _is_length_stop(stop_reason) -> bool:
    if stop_reason is None:
        return False
    return str(stop_reason).lower() in ("max_tokens", "length", "token_limit", "output_limit")


def create_with_continuation(
    *,
    system: str | None,
    messages: list,
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    timeout: int = 300,
    allow_continuation: bool = True,
    reasoning_effort: str | None = None,
    call_type: str = "main",
) -> str:
    """
    Gọi model hiện tại; nếu provider báo bị cắt vì max token thì gọi tiếp để nối output.
    Không dùng Python sửa nội dung chiến lược, chỉ yêu cầu model viết tiếp phần bị ngắt.
    """
    convo = list(messages)
    full_text = ""
    max_attempts = LLM_MAX_CONTINUATIONS + 1 if allow_continuation else 1
    for attempt in range(max_attempts):
        result = llm_create_once(
            system,
            convo,
            max_tokens=max_tokens,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
        chunk = result.get("text") or ""
        full_text += chunk
        stop_reason = result.get("stop_reason")
        try:
            print(
                f"[LLM_RESPONSE] call_type={call_type} provider={AI_PROVIDER} model={get_ai_model_name()} "
                f"effort={result.get('effort')} attempt={attempt + 1} stop_reason={stop_reason} usage={result.get('usage')}",
                flush=True,
            )
        except Exception:
            pass
        if not _is_length_stop(stop_reason):
            break
        if not allow_continuation:
            print(
                "[LLM_LENGTH_NO_CONTINUE] Model trả stop_reason=length nhưng call này không continuation.",
                flush=True,
            )
            break
        print("[LLM_TRUNCATED] Model trả stop_reason=length, gọi tiếp để nối phần còn lại...", flush=True)
        convo = convo + [
            {"role": "assistant", "content": chunk},
            {
                "role": "user",
                "content": (
                    "Tiếp tục viết nốt phần còn lại ngay từ chỗ bị ngắt, "
                    "không lặp lại nội dung đã viết, không giải thích gì thêm."
                ),
            },
        ]
    return full_text.strip()


# ─── Tóm tắt reasoning bằng call model thứ 2 (rất ngắn) ──────────────────────
def summarize_reasoning(full_response: str) -> str:
    """
    Tóm tắt lý do ra quyết định thành ~50 từ.
    Dùng cùng provider đang cấu hình: Anthropic hoặc OpenRouter/GLM.
    """
    if not get_ai_api_key():
        return ""
    try:
        summary_effort = (
            OPENROUTER_SUMMARY_REASONING_EFFORT
            if AI_PROVIDER in ("openrouter", "glm", "zai", "z.ai")
            else ANTHROPIC_SUMMARY_EFFORT
        )
        text = create_with_continuation(
            system=None,
            messages=[{
                "role": "user",
                "content": (
                    "Tóm tắt trong 1-2 câu (tối đa 60 từ) lý do kỹ thuật chính "
                    "dẫn đến quyết định LONG/SHORT/NO_TRADE trong phân tích sau. "
                    "Chỉ nêu các chỉ báo cụ thể (EMA, RSI, MACD, volume, ATR, vùng giá) và mức giá. "
                    "Không dùng chữ Hist, MACD_hist, Histogram; hãy viết động lượng MACD âm/dương hoặc MACD còn âm/dương. "
                    "Không giải thích, không lời mở đầu.\n\n"
                    + full_response[:2000]
                ),
            }],
            max_tokens=LLM_SUMMARY_MAX_OUTPUT_TOKENS,
            timeout=60,
            allow_continuation=False,
            reasoning_effort=summary_effort,
            call_type="summary",
        )
        return sanitize_user_output(text.strip())
    except Exception as exc:
        print(f"Lỗi summarize_reasoning: {exc}", flush=True)
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
    m = re.search(r"QUYẾT ĐỊNH[:\s]+(LONG|SHORT|NO[_\s-]?TRADE)", output, re.IGNORECASE)
    if m:
        direction = m.group(1).upper().replace(" ", "_").replace("-", "_")
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








def sanitize_user_output(output: str) -> str:
    """Dọn wording dễ gây nhầm và nhãn kỹ thuật nội bộ trước khi gửi user/lưu full_response."""
    replacements = {
        "swing gần": "đỉnh/đáy gần",
        "Swing gần": "Đỉnh/đáy gần",
        "swing lớn": "biên lớn",
        "Swing lớn": "Biên lớn",
        "MARKET_REGIME_DO_PYTHON_PHAN_LOAI": "phân loại thị trường do Python",
        "FEATURE_ENGINEERING_DO_PYTHON_TINH_SAN": "dữ liệu kỹ thuật do Python tính sẵn",
        "REGIME_CHINH": "xu hướng chính",
        "BULL_TREND": "xu hướng tăng",
        "BEAR_TREND": "xu hướng giảm",
        "RANGE_CHOPPY": "đi ngang/nhiễu",
        "MIXED_UNCLEAR": "chưa rõ xu hướng",
        "MIXED_TRANSITION": "trạng thái chuyển pha",
        "TRENDING_UP": "xu hướng tăng rõ",
        "TRENDING_DOWN": "xu hướng giảm rõ",
        "HIGH_VOLATILITY_RISK": "rủi ro biến động mạnh",
        "LOW_LIQUIDITY_RISK": "rủi ro thanh khoản thấp",
        "LOWER_TIMEFRAME_PULLBACK_AGAINST_STRUCTURE": "khung nhỏ đang hồi ngược cấu trúc lớn",
        "HIGH_VOLATILITY": "biến động mạnh",
        "LOW_VOLATILITY": "biến động thấp",
        "NORMAL_VOLATILITY": "biến động bình thường",
        "HIGH_VOLUME": "khối lượng cao",
        "LOW_VOLUME": "khối lượng thấp",
        "NORMAL_VOLUME": "khối lượng bình thường",
        "EMA_TANG": "EMA nghiêng tăng",
        "EMA_GIAM": "EMA nghiêng giảm",
        "EMA_DAN_XEN": "EMA đan xen",
        "modifier": "ghi chú",
    }
    text = output or ""
    # Replace longer internal labels first so overlapping terms do not leave fragments.
    for old in sorted(replacements, key=len, reverse=True):
        text = text.replace(old, replacements[old])

    # Dọn riêng các nhãn MACD histogram bằng regex để không làm hỏng chữ như "history".
    text = re.sub(r"\bMACD[_\s-]*hist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)
    return text








def build_no_trade_summary(output: str) -> str:
    text = (output or "").strip().replace("\n", " ")
    if not text:
        return "Claude chọn NO_TRADE nhưng không có lý do rõ."
    return "NO_TRADE: " + text[:600]


def log_hidden_rejection(symbol: str, mode: str, pred: dict, validation_errors: list[str], output: str) -> None:
    """Log nội bộ để debug trên Railway."""
    try:
        print("[TEOPARD_REJECTED]", flush=True)
        print(f"symbol={symbol} mode={mode} direction={pred.get('direction')}", flush=True)
        print("errors=" + " | ".join(str(e) for e in (validation_errors or [])), flush=True)
        print("output_preview=" + (output or "")[:1500].replace("\n", " "), flush=True)
    except Exception:
        pass




def load_system_prompt() -> str:
    for p in [Path("analyze_system_prompt.txt"), Path("analysis_system_prompt.txt")]:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise FileNotFoundError("Không tìm thấy analyze_system_prompt.txt")


def load_timeframe_data(binance_symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
    """Sync helper: fetch Binance candles then calculate indicators."""
    return add_indicators(get_binance_klines(binance_symbol, interval, limit))


def request_claude_analysis(system_prompt: str, user_prompt: str) -> str:
    """Sync helper: gọi model hiện tại qua Anthropic hoặc OpenRouter/GLM."""
    return create_with_continuation(
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=300,
        call_type="main",
    )

def call_claude_analysis(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> str:
    """
    Legacy synchronous entry point.

    Không gọi hàm này trực tiếp trong Telegram async handler, vì bên trong có
    requests.get(), AI API sync và SQLite. Handler phải gọi analyze_symbol(),
    hàm đó sẽ đưa các phần blocking sang worker thread bằng asyncio.to_thread().
    """

    ensure_ai_config()

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
    open_signals                     = get_open_signal_predictions(binance_symbol, mode, user_id=user_id)
    open_signal_context              = format_open_signal_context(open_signals, current_price)
    user_prompt                      = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        history=history,
        feature_block=feature_block,
        open_signal_context=open_signal_context,
    )

    output = sanitize_user_output(request_claude_analysis(system_prompt, user_prompt))

    pred = parse_prediction_from_output(output)

    # Sonnet-trust mode:
    # - Không dùng Python risk/format validator để ẩn phản hồi của Claude nữa.
    # - Model trả gì thì gửi user đúng nội dung đó.
    # - Python chỉ parse tối thiểu Entry/SL/TP để lưu auto-check nếu đủ số.
    direction = (pred.get("direction") or "").upper()

    if direction == "NO_TRADE":
        save_no_trade_prediction(
            symbol=binance_symbol,
            mode=mode,
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=build_no_trade_summary(output),
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
        return output

    can_track = (
        direction in ("LONG", "SHORT")
        and pred.get("entry_low") is not None
        and pred.get("entry_high") is not None
        and pred.get("sl") is not None
        and pred.get("tp1") is not None
        and pred.get("tp2") is not None
    )

    if can_track:
        reasoning_summary = summarize_reasoning(output)
        save_prediction(
            symbol=binance_symbol,
            mode=mode,
            direction=direction,
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
    else:
        # Không ẩn output. Chỉ lưu hidden để learning/debug vì bot không đủ số để auto-check.
        missing = []
        if direction not in ("LONG", "SHORT"):
            missing.append("Không parse được QUYẾT ĐỊNH LONG/SHORT/NO_TRADE.")
        for field in ("entry_low", "entry_high", "sl", "tp1", "tp2"):
            if pred.get(field) is None:
                missing.append(f"Không parse được {field}.")
        log_hidden_rejection(binance_symbol, mode, pred, missing, output)
        save_rejected_prediction(
            symbol=binance_symbol,
            mode=mode,
            direction=direction or pred.get("direction"),
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary="Không đủ số để auto-check: " + " ; ".join(missing[:5]),
            full_response=output,
            validation_errors=missing,
            user_id=user_id,
            chat_id=chat_id,
        )

    return output

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

    Không gọi requests.get(), AI API sync hoặc SQLite trực tiếp trên event loop.
    Các phần I/O blocking được chuyển sang worker thread bằng asyncio.to_thread().
    """
    ensure_ai_config()

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
    open_signal_task = asyncio.to_thread(get_open_signal_predictions, binance_symbol, mode, user_id)

    system_prompt, fear_greed_info, price_tuple, history, open_signals = await asyncio.gather(
        system_prompt_task,
        fear_greed_task,
        current_price_task,
        history_task,
        open_signal_task,
    )
    current_price_str, current_price = price_tuple

    feature_block = build_feature_engineering_block(timeframe_data, mode, current_price)
    feature_snapshot = build_feature_snapshot(timeframe_data, mode, current_price)
    market_snapshot = build_market_snapshot(
        timeframe_data,
        fear_greed_info,
        current_price_str,
    )
    open_signal_context = format_open_signal_context(open_signals, current_price)
    user_prompt = build_user_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_str=current_price_str,
        history=history,
        feature_block=feature_block,
        open_signal_context=open_signal_context,
    )

    # AI API đang sync, nên gọi trong worker thread để không block bot.
    output = sanitize_user_output(await asyncio.to_thread(request_claude_analysis, system_prompt, user_prompt))

    pred = parse_prediction_from_output(output)

    # Sonnet-trust mode:
    # - Không dùng Python risk/format validator để ẩn phản hồi của Claude nữa.
    # - Model trả gì thì gửi user đúng nội dung đó.
    # - Python chỉ parse tối thiểu Entry/SL/TP để lưu auto-check nếu đủ số.
    direction = (pred.get("direction") or "").upper()

    if direction == "NO_TRADE":
        await asyncio.to_thread(
            save_no_trade_prediction,
            symbol=binance_symbol,
            mode=mode,
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=build_no_trade_summary(output),
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
        return output

    can_track = (
        direction in ("LONG", "SHORT")
        and pred.get("entry_low") is not None
        and pred.get("entry_high") is not None
        and pred.get("sl") is not None
        and pred.get("tp1") is not None
        and pred.get("tp2") is not None
    )

    if can_track:
        reasoning_summary = await asyncio.to_thread(summarize_reasoning, output)
        await asyncio.to_thread(
            save_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=direction,
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary=reasoning_summary,
            full_response=output,
            user_id=user_id,
            chat_id=chat_id,
        )
    else:
        # Không ẩn output. Chỉ lưu hidden để learning/debug vì bot không đủ số để auto-check.
        missing = []
        if direction not in ("LONG", "SHORT"):
            missing.append("Không parse được QUYẾT ĐỊNH LONG/SHORT/NO_TRADE.")
        for field in ("entry_low", "entry_high", "sl", "tp1", "tp2"):
            if pred.get(field) is None:
                missing.append(f"Không parse được {field}.")
        log_hidden_rejection(binance_symbol, mode, pred, missing, output)
        await asyncio.to_thread(
            save_rejected_prediction,
            symbol=binance_symbol,
            mode=mode,
            direction=direction or pred.get("direction"),
            entry_low=pred.get("entry_low"),
            entry_high=pred.get("entry_high"),
            sl=pred.get("sl"),
            tp1=pred.get("tp1"),
            tp2=pred.get("tp2"),
            market_snapshot=market_snapshot,
            feature_snapshot=feature_snapshot,
            reasoning_summary="Không đủ số để auto-check: " + " ; ".join(missing[:5]),
            full_response=output,
            validation_errors=missing,
            user_id=user_id,
            chat_id=chat_id,
        )

    return output
