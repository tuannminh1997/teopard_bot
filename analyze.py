import asyncio
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import anthropic
except Exception:
    anthropic = None
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_URL   = "https://api.binance.com/api/v3/klines"

# ─── AI provider config ───────────────────────────────────────────────────────
# V33: chốt dùng GLM/Z.AI native làm provider chính.
# OpenRouter/Claude code vẫn còn để không làm vỡ import cũ, nhưng Railway không cần set các biến đó nữa.
AI_PROVIDER = os.getenv("AI_PROVIDER", "zai").strip().lower()

OPENROUTER_PROVIDER_NAMES = {"openrouter", "or", "glm_openrouter", "openrouter_glm"}
ZAI_PROVIDER_NAMES = {"zai", "z.ai", "z_ai", "zai_native", "zai-official", "zai_official", "glm_native"}
ANTHROPIC_PROVIDER_NAMES = {"anthropic", "claude", "claude_native"}
# Backward compatible: trước đây AI_PROVIDER=glm được hiểu là GLM qua OpenRouter.
OPENROUTER_LEGACY_PROVIDER_NAMES = {"glm"}


def _is_openrouter_provider(provider: str | None = None) -> bool:
    p = (provider or AI_PROVIDER or "").strip().lower()
    return p in OPENROUTER_PROVIDER_NAMES or p in OPENROUTER_LEGACY_PROVIDER_NAMES


def _is_zai_provider(provider: str | None = None) -> bool:
    p = (provider or AI_PROVIDER or "").strip().lower()
    return p in ZAI_PROVIDER_NAMES


def _is_anthropic_provider(provider: str | None = None) -> bool:
    p = (provider or AI_PROVIDER or "").strip().lower()
    return p in ANTHROPIC_PROVIDER_NAMES or not (_is_openrouter_provider(p) or _is_zai_provider(p))


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

# Z.AI native/chính chủ. Không cần thêm SDK, vẫn gọi HTTP OpenAI-compatible bằng requests.
ZAI_API_KEY  = os.getenv("ZAI_API_KEY")
ZAI_MODEL    = os.getenv("ZAI_MODEL", "glm-5.2")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4").rstrip("/")
# Vì bạn muốn giảm lag, native Z.AI mặc định dùng high thay vì max. Có thể đổi max/xhigh trên Railway nếu muốn.
ZAI_REASONING_EFFORT = os.getenv("ZAI_REASONING_EFFORT", "high").strip()
ZAI_SUMMARY_REASONING_EFFORT = os.getenv("ZAI_SUMMARY_REASONING_EFFORT", "none").strip()
ZAI_APP_NAME = os.getenv("ZAI_APP_NAME", "Teopard Bot")

LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "8000"))
LLM_MAX_CONTINUATIONS = int(os.getenv("LLM_MAX_CONTINUATIONS", "2"))
# Call tóm tắt reasoning dùng token riêng và KHÔNG continuation để tránh model đốt token reasoning ẩn.
LLM_SUMMARY_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_SUMMARY_MAX_OUTPUT_TOKENS", "600"))
# Mặc định tắt reasoning cho summary. Phân tích chính vẫn dùng provider-specific reasoning effort nếu bạn set.
OPENROUTER_SUMMARY_REASONING_EFFORT = os.getenv("OPENROUTER_SUMMARY_REASONING_EFFORT", "off").strip()
# Giữ tên cũ để code cũ không crash nếu còn tham chiếu.
CLAUDE_MAX_TOKENS = LLM_MAX_OUTPUT_TOKENS

DB_PATH           = os.getenv("DB_PATH", "bot.db")

# V33 timeframe roles:
# SCALP: 15M chỉ là trigger/timing; 1H là khung setup chính; 4H là trend filter; 1D là macro context.
SHORT_TERM_TIMEFRAMES = {
    "15M": ("15m", 480),   # ~5 ngày, trigger/timing, sweep/wick; không dùng làm bias chính
    "1H":  ("1h",  360),   # ~15 ngày, setup/momentum chính cho SCALP
    "4H":  ("4h",  360),   # ~60 ngày, trend filter + target/liquidity chính
    "1D":  ("1d",  365),   # ~1 năm, bối cảnh lớn; tránh scalp ngược macro quá rõ
}

# SWING: 1H chỉ timing phụ; 4H setup; 1D decision/trend chính; 1W macro context.
LONG_TERM_TIMEFRAMES = {
    "1H": ("1h",  480),   # entry trigger / pullback timing phụ cho SWING
    "4H": ("4h",  360),   # setup + invalidation gần
    "1D": ("1d",  365),   # trend/decision chính cho SWING
    "1W": ("1w",  208),   # macro context/liquidity sâu
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
# V19: REJECTED_PLAN/NO_TRADE không còn được lưu vào predictions sau mỗi lần phân tích.
# Biến này vẫn giữ để lọc dữ liệu cũ trong DB của các bản trước.
HIDDEN_LEARNING_RESULTS = ("REJECTED_PLAN", "NO_TRADE")
TRADE_CANDIDATE_RETENTION_LIMIT = int(os.getenv("TRADE_CANDIDATE_RETENTION_LIMIT", "20"))
TRADE_CANDIDATE_EXPIRE_HOURS = int(os.getenv("TRADE_CANDIDATE_EXPIRE_HOURS", "24"))
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

        # V19: phân tích hợp lệ chỉ lưu vào bảng draft/candidate.
        # Chỉ khi user bấm "Tôi đã trade theo lệnh này" mới copy sang predictions để auto-check.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_candidates (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id             INTEGER,
                chat_id             INTEGER,
                symbol              TEXT NOT NULL,
                mode                TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                expires_at          TEXT NOT NULL,
                direction           TEXT NOT NULL,
                entry_low           REAL,
                entry_high          REAL,
                sl                  REAL,
                tp1                 REAL,
                tp2                 REAL,
                market_snapshot     TEXT,
                feature_snapshot    TEXT,
                reasoning_summary   TEXT,
                full_response       TEXT,
                status              TEXT NOT NULL DEFAULT 'DRAFT',
                confirmed_at        TEXT,
                confirmed_prediction_id INTEGER
            )
        """)
        for col, definition in [
            ("expires_at", "TEXT"),
            ("status", "TEXT NOT NULL DEFAULT 'DRAFT'"),
            ("confirmed_at", "TEXT"),
            ("confirmed_prediction_id", "INTEGER"),
        ]:
            try:
                conn.execute(f"ALTER TABLE trade_candidates ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass

        # Index nhẹ cho history/stats/learning/auto-check khi DB lớn hơn.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id_id ON predictions(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_symbol_mode_id ON predictions(user_id, symbol, mode, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_result_next_check ON predictions(result, next_check_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trade_candidates_user_status_id ON trade_candidates(user_id, status, id DESC)")
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



def prune_trade_candidates(user_id: int | None = None) -> None:
    """Giữ bảng draft gọn. Candidate chỉ là phân tích chờ user xác nhận, không phải history."""
    now_s = iso(utc_now())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM trade_candidates
            WHERE status='DRAFT' AND expires_at < ?
            """,
            (now_s,),
        )
        if user_id is not None:
            conn.execute(
                """
                DELETE FROM trade_candidates
                WHERE user_id=?
                  AND id NOT IN (
                      SELECT id FROM trade_candidates
                      WHERE user_id=?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (user_id, user_id, TRADE_CANDIDATE_RETENTION_LIMIT),
            )
        conn.commit()


def save_trade_candidate(
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
    """Lưu bản nháp có thể track. Không xuất hiện trong /history, /stats, auto-check."""
    now = utc_now()
    expires_at = now + timedelta(hours=TRADE_CANDIDATE_EXPIRE_HOURS)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO trade_candidates
                (user_id, chat_id, symbol, mode, created_at, expires_at, direction,
                 entry_low, entry_high, sl, tp1, tp2,
                 market_snapshot, feature_snapshot, reasoning_summary, full_response, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'DRAFT')
            """,
            (user_id, chat_id, symbol, mode, iso(now), iso(expires_at), direction,
             entry_low, entry_high, sl, tp1, tp2,
             market_snapshot, feature_snapshot, reasoning_summary, full_response),
        )
        candidate_id = cursor.lastrowid
        conn.commit()
    prune_trade_candidates(user_id)
    return int(candidate_id)


def get_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict | None:
    init_prediction_db()
    clauses = ["id=?"]
    params: list = [candidate_id]
    if user_id is not None:
        clauses.append("user_id=?")
        params.append(user_id)
    where = " AND ".join(clauses)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            SELECT id, user_id, chat_id, symbol, mode, created_at, expires_at, direction,
                   entry_low, entry_high, sl, tp1, tp2,
                   market_snapshot, feature_snapshot, reasoning_summary, full_response,
                   status, confirmed_prediction_id
            FROM trade_candidates
            WHERE {where}
            """,
            params,
        ).fetchone()
    if not row:
        return None
    keys = [
        "id", "user_id", "chat_id", "symbol", "mode", "created_at", "expires_at", "direction",
        "entry_low", "entry_high", "sl", "tp1", "tp2",
        "market_snapshot", "feature_snapshot", "reasoning_summary", "full_response",
        "status", "confirmed_prediction_id",
    ]
    return dict(zip(keys, row))


def _candidate_entry_price(candidate: dict, live_price: float | None = None) -> float | None:
    low = candidate.get("entry_low")
    high = candidate.get("entry_high")
    direction = (candidate.get("direction") or "").upper()
    if low is None or high is None:
        return live_price
    low_f = min(float(low), float(high))
    high_f = max(float(low), float(high))
    if live_price is not None and low_f <= float(live_price) <= high_f:
        return float(live_price)
    # User bấm "đã trade" nhưng bot không biết giá khớp thực tế. Dùng mép bất lợi để chấm bảo thủ.
    if direction == "LONG":
        return high_f
    if direction == "SHORT":
        return low_f
    return (low_f + high_f) / 2.0


def confirm_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict:
    """User xác nhận đã trade theo bot -> copy candidate sang predictions và bắt đầu auto-check.

    V20: mỗi nút xác nhận gắn với đúng một trade_candidates.id.
    Hàm này claim candidate bằng UPDATE status='CONFIRMING' trước khi save_prediction để chống double-click/race.
    Vì vậy user bấm nhiều lần hoặc Telegram gửi callback lặp lại cũng không tạo nhiều prediction.
    """
    init_prediction_db()
    candidate = get_trade_candidate(candidate_id, user_id=user_id)
    if not candidate:
        return {"ok": False, "message": "Không tìm thấy lệnh nháp này, hoặc lệnh không thuộc user hiện tại."}

    status = (candidate.get("status") or "DRAFT").upper()
    if status == "CONFIRMED" and candidate.get("confirmed_prediction_id"):
        return {
            "ok": True,
            "already_confirmed": True,
            "prediction_id": int(candidate["confirmed_prediction_id"]),
            "message": f"Lệnh nháp #{candidate_id} đã được lưu theo dõi trước đó. Mã theo dõi: #{candidate['confirmed_prediction_id']}.",
        }
    if status == "CONFIRMING":
        return {"ok": True, "message": f"Lệnh nháp #{candidate_id} đang được xử lý xác nhận. Vui lòng không bấm lại."}
    if status != "DRAFT":
        return {"ok": False, "message": f"Lệnh nháp #{candidate_id} không còn hiệu lực để lưu. Trạng thái hiện tại: {status}."}

    expires = parse_utc_datetime(candidate.get("expires_at"))
    if expires is not None and utc_now() > expires:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE trade_candidates SET status='EXPIRED' WHERE id=? AND status='DRAFT'",
                (candidate_id,),
            )
            conn.commit()
        return {"ok": False, "message": f"Lệnh nháp #{candidate_id} đã quá hạn xác nhận. Hãy phân tích lại để có dữ liệu mới."}

    # Claim atomically before doing network/DB work. This prevents duplicate saves on double-click.
    with sqlite3.connect(DB_PATH) as conn:
        if user_id is None:
            cur = conn.execute(
                "UPDATE trade_candidates SET status='CONFIRMING' WHERE id=? AND status='DRAFT'",
                (candidate_id,),
            )
        else:
            cur = conn.execute(
                "UPDATE trade_candidates SET status='CONFIRMING' WHERE id=? AND user_id=? AND status='DRAFT'",
                (candidate_id, user_id),
            )
        conn.commit()
        claimed = cur.rowcount == 1

    if not claimed:
        latest = get_trade_candidate(candidate_id, user_id=user_id)
        latest_status = (latest or {}).get("status", "UNKNOWN")
        latest_pid = (latest or {}).get("confirmed_prediction_id")
        if str(latest_status).upper() == "CONFIRMED" and latest_pid:
            return {
                "ok": True,
                "already_confirmed": True,
                "prediction_id": int(latest_pid),
                "message": f"Lệnh nháp #{candidate_id} đã được lưu theo dõi trước đó. Mã theo dõi: #{latest_pid}.",
            }
        return {"ok": True, "message": f"Lệnh nháp #{candidate_id} đang được xử lý hoặc đã đổi trạng thái: {latest_status}."}

    try:
        live_price = get_current_price_raw(candidate["symbol"])
        entry_price = _candidate_entry_price(candidate, live_price)

        prediction_id = save_prediction(
            symbol=candidate["symbol"],
            mode=candidate["mode"],
            direction=candidate["direction"],
            entry_low=candidate.get("entry_low"),
            entry_high=candidate.get("entry_high"),
            sl=candidate.get("sl"),
            tp1=candidate.get("tp1"),
            tp2=candidate.get("tp2"),
            market_snapshot=candidate.get("market_snapshot"),
            feature_snapshot=candidate.get("feature_snapshot"),
            reasoning_summary=candidate.get("reasoning_summary"),
            full_response=candidate.get("full_response"),
            user_id=candidate.get("user_id"),
            chat_id=candidate.get("chat_id"),
        )

        # Vì user bấm "đã trade", bắt đầu theo dõi như lệnh đã khớp thay vì chờ Entry lần nữa.
        if entry_price is not None:
            mark_entry_filled(prediction_id, float(entry_price), utc_now(), candidate["mode"])

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                UPDATE trade_candidates
                SET status='CONFIRMED', confirmed_at=?, confirmed_prediction_id=?
                WHERE id=?
                """,
                (iso(utc_now()), prediction_id, candidate_id),
            )
            conn.commit()

        return {
            "ok": True,
            "prediction_id": int(prediction_id),
            "entry_price": entry_price,
            "message": (
                f"Đã lưu lệnh nháp #{candidate_id} thành lệnh theo dõi #{prediction_id}. "
                f"Giá vào theo dõi: {fmt(entry_price)}."
            ),
        }
    except Exception as exc:
        # Nếu lỗi sau khi claim, mở lại DRAFT để user có thể bấm lại sau khi lỗi tạm thời qua đi.
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE trade_candidates SET status='DRAFT' WHERE id=? AND status='CONFIRMING'",
                (candidate_id,),
            )
            conn.commit()
        return {"ok": False, "message": f"Lưu lệnh nháp #{candidate_id} thất bại: {exc}"}


def discard_trade_candidate(candidate_id: int, user_id: int | None = None) -> dict:
    init_prediction_db()
    candidate = get_trade_candidate(candidate_id, user_id=user_id)
    if not candidate:
        return {"ok": False, "message": "Không tìm thấy lệnh nháp này, hoặc lệnh không thuộc user hiện tại."}
    if (candidate.get("status") or "").upper() != "DRAFT":
        return {"ok": True, "message": "Lệnh này không còn là nháp, không cần bỏ qua."}
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE trade_candidates SET status='DISCARDED' WHERE id=?", (candidate_id,))
        conn.commit()
    return {"ok": True, "message": "Đã bỏ qua lệnh này. Bot sẽ không lưu vào history và không theo dõi kết quả."}

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
             market_snapshot, feature_snapshot, reasoning_summary or "Claude chọn NO TRADE.", full_response,
             "Claude chọn NO TRADE vì chưa có setup đủ rõ hoặc tỷ lệ lời/lỗ chưa đáng để tạo tín hiệu.", iso(now)),
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
    - Khi phân tích cho user nào, AI chỉ nhận lịch sử lệnh mà chính user đó đã bấm xác nhận trade theo bot.
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
              AND result NOT IN ('REJECTED_PLAN', 'NO_TRADE')
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
        "- Nếu giá không hồi về Entry cũ mà đã chạy theo hướng dự báo, không được đuổi giá chỉ vì giá đang chạy. Chỉ cho vào ngay khi giá hiện tại nằm trong vùng Entry mới hợp lý và đã có xác nhận rõ; nếu không, ưu tiên NO TRADE hoặc chờ kiểm tra lại.",
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
        f"Tổng lệnh đã trade theo bot: {total}",
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

    # V21: /history hiển thị số thứ tự ổn định theo cửa sổ 10 lệnh gần nhất: cũ → mới.
    # Ví dụ có #1..#10, khi lệnh thứ 11 được lưu thì lệnh cũ nhất bị prune,
    # lệnh cũ #2 sẽ thành #1, ... và lệnh mới nhất thành #10.
    # DB id vẫn giữ nguyên ở trong DB, nhưng không dùng làm số hiển thị cho user.
    rows = list(reversed(rows))

    # user_id=None chỉ được dùng cho admin, nên admin sẽ thấy lệnh thuộc user nào.
    is_admin_scope = user_id is None
    lines = [f"🧾 10 lệnh đã trade theo bot gần nhất {format_scope_label(symbol, user_id)}"]
    for display_idx, row in enumerate(rows, 1):
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
            f"#{display_idx} {sym} {mode_label} {direction} → {result}\n"
            f"{owner_line}"
            f"Thời gian phân tích: {created_label}\n"
            f"Entry {fmt(entry_low)}–{fmt(entry_high)} | SL {fmt(sl)} | TP1 {fmt(tp1)} | TP2 {fmt(tp2)}"
            + (f" | Giá check {fmt(result_price)}" if result_price else "")
            + reason_line
        )
    return "\n\n".join(lines)


def clear_prediction_history() -> dict:
    init_prediction_db()
    with sqlite3.connect(DB_PATH) as conn:
        visible_count = int(conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE result NOT IN ('REJECTED_PLAN', 'NO_TRADE')"
        ).fetchone()[0])
        total_prediction_count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
        try:
            draft_count = int(conn.execute("SELECT COUNT(*) FROM trade_candidates").fetchone()[0])
        except sqlite3.Error:
            draft_count = 0
        conn.execute("DELETE FROM predictions")
        try:
            conn.execute("DELETE FROM trade_candidates")
        except sqlite3.Error:
            pass
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='trade_candidates'")
        except sqlite3.Error:
            pass
        conn.commit()
    return {
        "visible_count": visible_count,
        "total_prediction_count": total_prediction_count,
        "draft_count": draft_count,
    }


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
    row = _analysis_row(df) if "_analysis_row" in globals() else (df.iloc[-2] if len(df) >= 2 else df.iloc[-1])
    return _safe_float(row.get("atr_14"))


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



def _zone_gap_to_price(zone: tuple | None, current_price: float, side: str) -> float:
    """Khoảng cách từ giá hiện tại tới mép trong của zone.

    side="lower": zone nằm dưới giá, gap = current - high.
    side="upper": zone nằm trên giá, gap = low - current.
    Nếu zone đang ôm/chạm giá thì gap = 0.
    """
    if not zone or len(zone) < 2 or zone[0] is None or zone[1] is None:
        return float("inf")
    low, high = float(zone[0]), float(zone[1])
    if low <= current_price <= high:
        return 0.0
    if side == "lower":
        return max(current_price - high, 0.0)
    return max(low - current_price, 0.0)


def _liq_zone_overlap_ratio(a: tuple | None, b: tuple | None) -> float:
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return 0.0
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    overlap = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    smaller = max(min(a_high - a_low, b_high - b_low), 1e-12)
    return overlap / smaller


def _liq_zone_external_gap(a: tuple | None, b: tuple | None) -> float:
    """Khoảng cách rỗng giữa 2 liquidity box; overlap/chạm nhau thì bằng 0."""
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return float("inf")
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    if a_high < b_low:
        return b_low - a_high
    if b_high < a_low:
        return a_low - b_high
    return 0.0


def _liq_zone_width(zone: tuple | None) -> float:
    if not zone or zone[0] is None or zone[1] is None:
        return 0.0
    return max(float(zone[1]) - float(zone[0]), 0.0)


def _mark_zone_merged_pool(zone: tuple, merged_with: str | None = None) -> tuple:
    """Đánh dấu nội bộ khi near/main/deep bị gộp vì cùng cụm thanh khoản."""
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["merged_pool"] = True
    if merged_with:
        roles = set(str(meta.get("merged_roles", "")).split("/")) if meta.get("merged_roles") else set()
        roles.add(str(meta.get("role", "")))
        roles.add(str(merged_with))
        roles = {r for r in roles if r}
        meta["merged_roles"] = "/".join(sorted(roles))
    return (zone[0], zone[1], zone[2], meta)


def _liq_zones_same_pool(a: tuple | None, b: tuple | None, current_price: float, mode: str) -> bool:
    """Tránh in cùng một pool thành gần/chính/sâu.

    Đây là lỗi chính ở các bản trước: 2 box không overlap nhiều nhưng chỉ cách nhau
    rất ít vẫn bị gán thành near/main/deep khác nhau. Với scalp, nếu 2 vùng chỉ
    cách nhau dưới khoảng 0.10% giá hoặc dưới ~1 box-width thì xem là cùng cụm
    stop/liquidity pool, không in thành nhiều mục tiêu riêng.
    """
    if not a or not b:
        return False

    if _liq_zone_overlap_ratio(a, b) >= 0.20:
        return True

    gap = _liq_zone_external_gap(a, b)
    width_a = _liq_zone_width(a)
    width_b = _liq_zone_width(b)
    max_width = max(width_a, width_b, 1e-12)
    avg_width = max((width_a + width_b) / 2.0, 1e-12)

    # Ngưỡng này là để gộp các role gần nhau, KHÔNG phải ép vùng cách xa nhau.
    # Scalp cần gộp chặt để tránh output kiểu gần/chính/sâu chỉ lệch vài USDT.
    gap_pct = 0.0010 if mode == "short" else 0.0022
    close_gap_threshold = max(current_price * gap_pct, avg_width * 0.85)
    if gap <= close_gap_threshold:
        return True

    ma = a[3] if len(a) > 3 and isinstance(a[3], dict) else {}
    mb = b[3] if len(b) > 3 and isinstance(b[3], dict) else {}
    la = ma.get("swing_level")
    lb = mb.get("swing_level")
    if la is None or lb is None:
        return False

    level_pct = 0.0013 if mode == "short" else 0.0028
    level_threshold = max(current_price * level_pct, max_width * 1.35)
    return abs(float(la) - float(lb)) <= level_threshold


def _copy_zone_with_assigned_role(zone: tuple, role: str) -> tuple:
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["assigned_role"] = role
    meta["role"] = role
    return (zone[0], zone[1], zone[2], meta)


def _normalize_liquidity_role_order(zones: dict, current_price: float, mode: str) -> None:
    """Chuẩn hóa near/main/deep sau khi tính candidate độc lập.

    Lý do: cùng một H4/D1 có thể sinh ra nhiều candidate, nhưng nếu mỗi role tự chọn
    độc lập thì sẽ có lỗi kiểu "sâu" gần hơn "chính", hoặc cùng một vùng bị in lặp.
    Hàm này không bịa vùng mới; chỉ sắp xếp lại candidate theo khoảng cách thật:
    - lower: càng gần giá thì swing low càng cao.
    - upper: càng gần giá thì swing high càng thấp.
    - vùng trùng pool bị bỏ bớt.
    - nếu candidate thứ hai quá xa so với scalp thì gán vào "sâu", để "chính" là N/A.
    """
    far_pct_cut = 0.025 if mode == "short" else 0.060
    far_atr_cut = 10.0 if mode == "short" else 12.0

    for side in ("lower", "upper"):
        raw: list[tuple] = []
        for role in ("near", "main", "deep"):
            z = zones.get(f"{side}_{role}")
            if z and z[0] is not None and z[1] is not None:
                raw.append(z)

        # Sort trước theo gap, sau đó score giảm dần để candidate gần hơn luôn được xét trước.
        raw.sort(
            key=lambda z: (
                _zone_gap_to_price(z, current_price, side),
                -float((z[3] if len(z) > 3 and isinstance(z[3], dict) else {}).get("score", 0.0)),
            )
        )

        unique: list[tuple] = []
        for z in raw:
            if any(_liq_zones_same_pool(z, kept, current_price, mode) for kept in unique):
                # Nếu trùng/quá sát pool, giữ 1 candidate duy nhất; không in thành gần/chính/sâu riêng.
                for i, kept in enumerate(unique):
                    if _liq_zones_same_pool(z, kept, current_price, mode):
                        mz = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
                        mk = kept[3] if len(kept) > 3 and isinstance(kept[3], dict) else {}
                        wz = abs(float(z[1]) - float(z[0]))
                        wk = abs(float(kept[1]) - float(kept[0]))
                        z_role = str(mz.get("role", ""))
                        kept_role = str(mk.get("role", ""))
                        # Cùng pool thì ưu tiên box hẹp hơn để scalp không bị in vùng rộng vô nghĩa.
                        # Nếu độ rộng gần như nhau, dùng score để chọn candidate chất lượng hơn.
                        choose_z = wz < wk * 0.92 or (
                            abs(wz - wk) <= wk * 0.08
                            and float(mz.get("score", 0.0)) > float(mk.get("score", 0.0))
                        )
                        if choose_z:
                            unique[i] = _mark_zone_merged_pool(z, kept_role)
                        else:
                            unique[i] = _mark_zone_merged_pool(kept, z_role)
                        break
                continue
            unique.append(z)

        assigned = {"near": None, "main": None, "deep": None}
        if unique:
            assigned["near"] = _copy_zone_with_assigned_role(unique[0], "near")

        for z in unique[1:]:
            gap = _zone_gap_to_price(z, current_price, side)
            gap_pct = gap / max(current_price, 1e-12)
            meta = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
            distance_atr = float(meta.get("distance_atr", 0.0) or 0.0)
            is_far = gap_pct >= far_pct_cut or distance_atr >= far_atr_cut

            if is_far:
                if assigned["deep"] is None:
                    assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")
                # Nếu đã có deep, bỏ candidate xa hơn/yếu hơn để tránh prompt loãng.
                continue

            if assigned["main"] is None:
                assigned["main"] = _copy_zone_with_assigned_role(z, "main")
            elif assigned["deep"] is None:
                assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")

        # Nếu chưa có main nhưng có deep rất gần do chỉ có 2 candidate, không kéo deep lên main.
        # Nếu có main và deep, đảm bảo deep thật sự xa hơn main.
        if assigned["main"] is not None and assigned["deep"] is not None:
            main_gap = _zone_gap_to_price(assigned["main"], current_price, side)
            deep_gap = _zone_gap_to_price(assigned["deep"], current_price, side)
            if deep_gap < main_gap:
                assigned["main"], assigned["deep"] = assigned["deep"], assigned["main"]
                assigned["main"] = _copy_zone_with_assigned_role(assigned["main"], "main")
                assigned["deep"] = _copy_zone_with_assigned_role(assigned["deep"], "deep")

        for role in ("near", "main", "deep"):
            zones[f"{side}_{role}"] = assigned[role]

    # Sau khi chuẩn hóa vai trò, label nên là vai trò giao dịch, không còn label theo timeframe cũ.
    zones["label_near"] = "gần"
    zones["label_main"] = "chính"
    zones["label_deep"] = "sâu"
    zones["liquidity_method"] = "fractal_swing_pool_v13_longer_lookback_tp_guard"

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



def _zone_gap_to_price(zone: tuple | None, current_price: float, side: str) -> float:
    """Khoảng cách từ giá hiện tại tới mép trong của zone.

    side="lower": zone nằm dưới giá, gap = current - high.
    side="upper": zone nằm trên giá, gap = low - current.
    Nếu zone đang ôm/chạm giá thì gap = 0.
    """
    if not zone or len(zone) < 2 or zone[0] is None or zone[1] is None:
        return float("inf")
    low, high = float(zone[0]), float(zone[1])
    if low <= current_price <= high:
        return 0.0
    if side == "lower":
        return max(current_price - high, 0.0)
    return max(low - current_price, 0.0)


def _liq_zone_overlap_ratio(a: tuple | None, b: tuple | None) -> float:
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return 0.0
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    overlap = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    smaller = max(min(a_high - a_low, b_high - b_low), 1e-12)
    return overlap / smaller


def _liq_zone_external_gap(a: tuple | None, b: tuple | None) -> float:
    """Khoảng cách rỗng giữa 2 liquidity box; overlap/chạm nhau thì bằng 0."""
    if not a or not b or a[0] is None or a[1] is None or b[0] is None or b[1] is None:
        return float("inf")
    a_low, a_high = float(a[0]), float(a[1])
    b_low, b_high = float(b[0]), float(b[1])
    if a_high < b_low:
        return b_low - a_high
    if b_high < a_low:
        return a_low - b_high
    return 0.0


def _liq_zone_width(zone: tuple | None) -> float:
    if not zone or zone[0] is None or zone[1] is None:
        return 0.0
    return max(float(zone[1]) - float(zone[0]), 0.0)


def _mark_zone_merged_pool(zone: tuple, merged_with: str | None = None) -> tuple:
    """Đánh dấu nội bộ khi near/main/deep bị gộp vì cùng cụm thanh khoản."""
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["merged_pool"] = True
    if merged_with:
        roles = set(str(meta.get("merged_roles", "")).split("/")) if meta.get("merged_roles") else set()
        roles.add(str(meta.get("role", "")))
        roles.add(str(merged_with))
        roles = {r for r in roles if r}
        meta["merged_roles"] = "/".join(sorted(roles))
    return (zone[0], zone[1], zone[2], meta)


def _liq_zones_same_pool(a: tuple | None, b: tuple | None, current_price: float, mode: str) -> bool:
    """Tránh in cùng một pool thành gần/chính/sâu.

    Đây là lỗi chính ở các bản trước: 2 box không overlap nhiều nhưng chỉ cách nhau
    rất ít vẫn bị gán thành near/main/deep khác nhau. Với scalp, nếu 2 vùng chỉ
    cách nhau dưới khoảng 0.10% giá hoặc dưới ~1 box-width thì xem là cùng cụm
    stop/liquidity pool, không in thành nhiều mục tiêu riêng.
    """
    if not a or not b:
        return False

    if _liq_zone_overlap_ratio(a, b) >= 0.20:
        return True

    gap = _liq_zone_external_gap(a, b)
    width_a = _liq_zone_width(a)
    width_b = _liq_zone_width(b)
    max_width = max(width_a, width_b, 1e-12)
    avg_width = max((width_a + width_b) / 2.0, 1e-12)

    # Ngưỡng này là để gộp các role gần nhau, KHÔNG phải ép vùng cách xa nhau.
    # Scalp cần gộp chặt để tránh output kiểu gần/chính/sâu chỉ lệch vài USDT.
    gap_pct = 0.0010 if mode == "short" else 0.0022
    close_gap_threshold = max(current_price * gap_pct, avg_width * 0.85)
    if gap <= close_gap_threshold:
        return True

    ma = a[3] if len(a) > 3 and isinstance(a[3], dict) else {}
    mb = b[3] if len(b) > 3 and isinstance(b[3], dict) else {}
    la = ma.get("swing_level")
    lb = mb.get("swing_level")
    if la is None or lb is None:
        return False

    level_pct = 0.0013 if mode == "short" else 0.0028
    level_threshold = max(current_price * level_pct, max_width * 1.35)
    return abs(float(la) - float(lb)) <= level_threshold


def _copy_zone_with_assigned_role(zone: tuple, role: str) -> tuple:
    if not zone or len(zone) < 4 or not isinstance(zone[3], dict):
        return zone
    meta = dict(zone[3])
    meta["assigned_role"] = role
    meta["role"] = role
    return (zone[0], zone[1], zone[2], meta)


def _normalize_liquidity_role_order(zones: dict, current_price: float, mode: str) -> None:
    """Chuẩn hóa near/main/deep sau khi tính candidate độc lập.

    Lý do: cùng một H4/D1 có thể sinh ra nhiều candidate, nhưng nếu mỗi role tự chọn
    độc lập thì sẽ có lỗi kiểu "sâu" gần hơn "chính", hoặc cùng một vùng bị in lặp.
    Hàm này không bịa vùng mới; chỉ sắp xếp lại candidate theo khoảng cách thật:
    - lower: càng gần giá thì swing low càng cao.
    - upper: càng gần giá thì swing high càng thấp.
    - vùng trùng pool bị bỏ bớt.
    - nếu candidate thứ hai quá xa so với scalp thì gán vào "sâu", để "chính" là N/A.
    """
    far_pct_cut = 0.025 if mode == "short" else 0.060
    far_atr_cut = 10.0 if mode == "short" else 12.0

    for side in ("lower", "upper"):
        raw: list[tuple] = []
        for role in ("near", "main", "deep"):
            z = zones.get(f"{side}_{role}")
            if z and z[0] is not None and z[1] is not None:
                raw.append(z)

        # Sort trước theo gap, sau đó score giảm dần để candidate gần hơn luôn được xét trước.
        raw.sort(
            key=lambda z: (
                _zone_gap_to_price(z, current_price, side),
                -float((z[3] if len(z) > 3 and isinstance(z[3], dict) else {}).get("score", 0.0)),
            )
        )

        unique: list[tuple] = []
        for z in raw:
            if any(_liq_zones_same_pool(z, kept, current_price, mode) for kept in unique):
                # Nếu trùng/quá sát pool, giữ 1 candidate duy nhất; không in thành gần/chính/sâu riêng.
                for i, kept in enumerate(unique):
                    if _liq_zones_same_pool(z, kept, current_price, mode):
                        mz = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
                        mk = kept[3] if len(kept) > 3 and isinstance(kept[3], dict) else {}
                        wz = abs(float(z[1]) - float(z[0]))
                        wk = abs(float(kept[1]) - float(kept[0]))
                        z_role = str(mz.get("role", ""))
                        kept_role = str(mk.get("role", ""))
                        # Cùng pool thì ưu tiên box hẹp hơn để scalp không bị in vùng rộng vô nghĩa.
                        # Nếu độ rộng gần như nhau, dùng score để chọn candidate chất lượng hơn.
                        choose_z = wz < wk * 0.92 or (
                            abs(wz - wk) <= wk * 0.08
                            and float(mz.get("score", 0.0)) > float(mk.get("score", 0.0))
                        )
                        if choose_z:
                            unique[i] = _mark_zone_merged_pool(z, kept_role)
                        else:
                            unique[i] = _mark_zone_merged_pool(kept, z_role)
                        break
                continue
            unique.append(z)

        assigned = {"near": None, "main": None, "deep": None}
        if unique:
            assigned["near"] = _copy_zone_with_assigned_role(unique[0], "near")

        for z in unique[1:]:
            gap = _zone_gap_to_price(z, current_price, side)
            gap_pct = gap / max(current_price, 1e-12)
            meta = z[3] if len(z) > 3 and isinstance(z[3], dict) else {}
            distance_atr = float(meta.get("distance_atr", 0.0) or 0.0)
            is_far = gap_pct >= far_pct_cut or distance_atr >= far_atr_cut

            if is_far:
                if assigned["deep"] is None:
                    assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")
                # Nếu đã có deep, bỏ candidate xa hơn/yếu hơn để tránh prompt loãng.
                continue

            if assigned["main"] is None:
                assigned["main"] = _copy_zone_with_assigned_role(z, "main")
            elif assigned["deep"] is None:
                assigned["deep"] = _copy_zone_with_assigned_role(z, "deep")

        # Nếu chưa có main nhưng có deep rất gần do chỉ có 2 candidate, không kéo deep lên main.
        # Nếu có main và deep, đảm bảo deep thật sự xa hơn main.
        if assigned["main"] is not None and assigned["deep"] is not None:
            main_gap = _zone_gap_to_price(assigned["main"], current_price, side)
            deep_gap = _zone_gap_to_price(assigned["deep"], current_price, side)
            if deep_gap < main_gap:
                assigned["main"], assigned["deep"] = assigned["deep"], assigned["main"]
                assigned["main"] = _copy_zone_with_assigned_role(assigned["main"], "main")
                assigned["deep"] = _copy_zone_with_assigned_role(assigned["deep"], "deep")

        for role in ("near", "main", "deep"):
            zones[f"{side}_{role}"] = assigned[role]

    # Sau khi chuẩn hóa vai trò, label nên là vai trò giao dịch, không còn label theo timeframe cũ.
    zones["label_near"] = "gần"
    zones["label_main"] = "chính"
    zones["label_deep"] = "sâu"
    zones["liquidity_method"] = "fractal_swing_pool_v13_longer_lookback_tp_guard"

def _liquidity_zones_by_windows(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float,
) -> dict:
    """Vùng thanh khoản V8: stop-pool ngoài swing, gộp near/main/deep nếu cùng cụm.

    Theo gợi ý OHLCV-only: khung lớn xác định level, khung nhỏ xác nhận sweep.
    SCALP: 1H cho vùng gần, 4H cho vùng chính/sâu, 15M chỉ kiểm tra sweep.
    SWING: H4 cho vùng gần, D1 cho vùng chính, W1/D1 cho vùng sâu, 1H/4H chỉ kiểm tra sweep.
    """
    if mode == "short":
        sweep_df = _closed_candles(timeframe_data.get("15M"))
        near_df = _first_valid_df(_closed_candles(timeframe_data.get("1H")), _closed_candles(timeframe_data.get("15M")))
        main_df = _first_valid_df(_closed_candles(timeframe_data.get("4H")), _closed_candles(timeframe_data.get("1H")))
        deep_df = _first_valid_df(_closed_candles(timeframe_data.get("4H")), _closed_candles(timeframe_data.get("1H")))
        windows = [
            # V13: lookback dài hơn để tránh vùng thanh khoản chỉ xoay quanh vài nến sát giá.
            # near vẫn dùng 1H nhưng lấy ~7 ngày; main/deep dùng H4 ~30-50 ngày.
            ("near", "gần 1H", near_df, _current_atr(near_df) or _current_atr(sweep_df), 168, 2),
            ("main", "chính H4", main_df, _current_atr(main_df) or _current_atr(near_df), 180, 2),
            ("deep", "sâu H4", deep_df, _current_atr(deep_df) or _current_atr(main_df), 300, 3),
        ]
        calc_mode = "short"
    else:
        # SWING: level lấy từ H4/D1/W1 đã đóng; sweep lấy từ 1H/4H đã đóng để không nhiễu nến live.
        # Không dùng 1H để tạo liquidity box vì sẽ khiến vùng swing bị nhiễu như scalp.
        sweep_df = _first_valid_df(_closed_candles(timeframe_data.get("1H")), _closed_candles(timeframe_data.get("4H")))
        near_df = _closed_candles(timeframe_data.get("4H"))
        main_df = _first_valid_df(_closed_candles(timeframe_data.get("1D")), _closed_candles(timeframe_data.get("4H")))
        deep_df = _first_valid_df(_closed_candles(timeframe_data.get("1W")), _closed_candles(timeframe_data.get("1D")), _closed_candles(timeframe_data.get("4H")))
        windows = [
            # V13: SWING cần vùng rộng lịch sử hơn: H4 ~40 ngày, D1 ~6-9 tháng, W1 vài năm.
            ("near", "gần H4", near_df, _current_atr(near_df) or _current_atr(sweep_df), 240, 2),
            ("main", "chính D1", main_df, _current_atr(main_df) or _current_atr(near_df), 240, 2),
            ("deep", "sâu W1/D1", deep_df, _current_atr(deep_df) or _current_atr(main_df), 180, 3),
        ]
        calc_mode = "long"

    result: dict = {"meta": [label for _, label, *_ in windows], "liquidity_method": "fractal_swing_pool_v13_longer_lookback_tp_guard"}
    for key, label, df, atr, lookback, m in windows:
        lower = _zone_for_liq_pools(df, sweep_df, current_price, atr, "low", key, calc_mode, lookback, m=m)
        upper = _zone_for_liq_pools(df, sweep_df, current_price, atr, "high", key, calc_mode, lookback, m=m)
        result[f"lower_{key}"] = lower
        result[f"upper_{key}"] = upper
        result[f"label_{key}"] = label

    _normalize_liquidity_role_order(result, current_price, calc_mode)
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
    df = _closed_candles(df)
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
    data = _closed_candles(df) if "_closed_candles" in globals() else df
    if data is None or len(data) < 2:
        return "Không đủ dữ liệu"
    count = 0
    last_dir = None
    for _, row in data.tail(12).iloc[::-1].iterrows():
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
    data = _closed_candles(df) if "_closed_candles" in globals() else df
    if data is None or data.empty:
        return "Không đủ dữ liệu"
    row = data.iloc[-1]
    high, low, open_, close = map(float, [row["high"], row["low"], row["open"], row["close"]])
    rng = max(high - low, 1e-12)
    body = abs(close - open_)
    upper = high - max(open_, close)
    lower = min(open_, close) - low
    return f"nến đã đóng: thân {body / rng * 100:.0f}%, râu trên {upper / rng * 100:.0f}%, râu dưới {lower / rng * 100:.0f}%"


def _mode_labels(mode: str) -> tuple[str, str, str]:
    # main/structure/big là các khung dùng để quyết định, không nhất thiết là khung trigger nhỏ nhất.
    # SCALP: 1H quyết định setup, 4H xác nhận xu hướng, 1D bối cảnh lớn. 15M chỉ timing.
    if mode == "short":
        return "1H", "4H", "1D"
    # SWING: 4H setup/entry zone, 1D quyết định xu hướng chính, 1W macro. 1H chỉ timing phụ.
    return "4H", "1D", "1W"


def _mode_trigger_label(mode: str) -> str:
    return "15M" if mode == "short" else "1H"


def _mode_role_text(mode: str) -> str:
    if mode == "short":
        return (
            "SCALP roles: 1H là khung setup/chính; 4H là xu hướng xác nhận; "
            "1D là bối cảnh lớn; 15M chỉ dùng để timing entry, đọc sweep/râu nến và không được đảo bias một mình."
        )
    return (
        "SWING roles: 1D là xu hướng/chính; 4H là setup/vùng vào; "
        "1W là bối cảnh lớn; 1H chỉ timing entry phụ và không được đảo bias swing một mình."
    )


def _risk_floor(timeframe_data: dict[str, pd.DataFrame | None], mode: str, current_price: float) -> float:
    """Mốc rủi ro tham chiếu để đưa vào prompt, không phải ngưỡng reject cứng.

    V11 dùng 0.35% giá cho SCALP nên BTC/ETH dễ bị ép risk quá lớn, TP gần không đạt
    RR và bot chuyển NO_TRADE liên tục. V12 giảm vai trò price% và dùng ATR thực tế nhiều hơn.
    """
    if mode == "short":
        atr_main = _current_atr(timeframe_data.get("15M")) or 0.0
        atr_confirm = _current_atr(timeframe_data.get("1H")) or 0.0
        return max(atr_main * 1.15, atr_confirm * 0.45, current_price * 0.0016)
    atr_main = _current_atr(timeframe_data.get("4H")) or 0.0
    atr_confirm = _current_atr(timeframe_data.get("1D")) or 0.0
    return max(atr_main * 1.20, atr_confirm * 0.38, current_price * 0.0090)


def _minimum_stop_distance(timeframe_data: dict[str, pd.DataFrame | None], mode: str, current_price: float) -> float:
    """Ngưỡng chống SL quá sát dùng để reject plan.

    Khác _risk_floor: đây là mức nhiễu tối thiểu, có cap theo % giá để không làm bot
    NO_TRADE quá nhiều khi ATR 1H/4H phình to. SL vẫn được đặt theo cấu trúc; hàm này chỉ
    chặn các plan có Entry-SL quá bé so với nhiễu thị trường.
    """
    if mode == "short":
        atr_main = _current_atr(timeframe_data.get("15M")) or 0.0
        atr_confirm = _current_atr(timeframe_data.get("1H")) or 0.0
        # V26: chỉ còn là ngưỡng chống SL vô lý siêu sát, không phải bộ lọc RR gắt.
        raw = max(atr_main * 0.35, atr_confirm * 0.10, current_price * 0.0003)
        return min(raw, current_price * 0.0012)
    atr_main = _current_atr(timeframe_data.get("4H")) or 0.0
    atr_confirm = _current_atr(timeframe_data.get("1D")) or 0.0
    raw = max(atr_main * 0.70, atr_confirm * 0.22, current_price * 0.0045)
    return min(raw, current_price * 0.0120)




def _closed_candles(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Dùng nến đã đóng để tìm swing/invalidation, tránh lấy nến realtime chưa chốt làm SL."""
    if df is None or df.empty:
        return None
    if len(df) >= 3:
        return df.iloc[:-1].copy()
    return df.copy()


def _structural_sl_buffer(timeframe_data: dict[str, pd.DataFrame | None], mode: str, current_price: float) -> float:
    """ATR buffer cho SL cấu trúc.

    Buffer này không dùng để kéo SL đại ra xa. Nó chỉ đặt SL ra ngoài swing/invalidation
    gần nhất một khoảng đủ tránh nhiễu nến. Nếu sau đó RR không đạt, plan sẽ bị NO TRADE.
    """
    main_label, structure_label, _ = _mode_labels(mode)
    trigger_label = _mode_trigger_label(mode)
    atr_trigger = _current_atr(timeframe_data.get(trigger_label)) or 0.0
    atr_main = _current_atr(timeframe_data.get(main_label)) or 0.0
    atr_structure = _current_atr(timeframe_data.get(structure_label)) or 0.0
    if mode == "short":
        return max(atr_main * 0.30, atr_structure * 0.10, current_price * 0.00025)
    return max(atr_main * 0.55, atr_structure * 0.20, current_price * 0.0030)


def _extra_user_sl_buffer_pct() -> float:
    """Phần trăm nới SL thêm sau khi có SL cấu trúc.

    Biến Railway TEOPARD_EXTRA_SL_BUFFER_PCT nhập theo đơn vị phần trăm:
      2    = nới 2% theo giá SL
      0.2  = nới 0.2%
      0    = tắt
    LONG: SL cuối = SL * (1 - pct)
    SHORT: SL cuối = SL * (1 + pct)
    """
    raw = os.getenv("TEOPARD_EXTRA_SL_BUFFER_PCT")
    if raw is None or str(raw).strip() == "":
        raw = os.getenv("TEOPARD_SL_EXTRA_BUFFER_PCT", "0")
    try:
        pct = float(str(raw).strip()) / 100.0
    except Exception:
        pct = 0.0
    if not np.isfinite(pct) or pct < 0:
        return 0.0
    return min(pct, 0.10)


def _apply_extra_sl_buffer(sl: float, direction: str) -> float:
    pct = _extra_user_sl_buffer_pct()
    if pct <= 0:
        return float(sl)
    if direction == "LONG":
        return float(sl) * (1.0 - pct)
    if direction == "SHORT":
        return float(sl) * (1.0 + pct)
    return float(sl)


def _apply_extra_sl_buffer_to_plan(pred: dict, output: str | None = None) -> tuple[dict, str | None]:
    """Áp buffer SL theo phong cách user SAU khi model/Python đã có plan gốc.

    Mặc định V32: model tự chọn SL. Python chỉ cộng/trừ phần trăm nếu user set
    TEOPARD_EXTRA_SL_BUFFER_PCT. RR guard mặc định dùng SL gốc trước buffer,
    nên style buffer không làm plan bị NO TRADE.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT") or pred.get("sl") is None:
        return pred, output
    pct = _extra_user_sl_buffer_pct()
    if pct <= 0:
        return pred, output
    normalized = dict(pred)
    old_sl = float(normalized["sl"])
    new_sl = _apply_extra_sl_buffer(old_sl, direction)
    normalized["_sl_before_extra_buffer"] = old_sl
    normalized["_extra_sl_buffer_pct"] = pct
    normalized["sl"] = new_sl
    if abs(new_sl - old_sl) > max(abs(old_sl) * 1e-7, 1e-8):
        output = _update_output_trade_numbers(output or "", normalized)
    return normalized, output


def _extra_user_tp_buffer_pct(tp_name: str) -> float:
    """Phần trăm nới TP thêm theo phong cách user.

    Biến Railway nhập theo đơn vị phần trăm:
      TEOPARD_EXTRA_TP1_BUFFER_PCT=1.2
      TEOPARD_EXTRA_TP2_BUFFER_PCT=1.2

    Có thể set chung TEOPARD_EXTRA_TP_BUFFER_PCT nếu muốn TP1/TP2 dùng cùng một %.
    LONG: TP cuối = TP * (1 + pct)
    SHORT: TP cuối = TP * (1 - pct)
    """
    name = (tp_name or "").strip().upper()
    raw = os.getenv(f"TEOPARD_EXTRA_{name}_BUFFER_PCT")
    if raw is None or str(raw).strip() == "":
        raw = os.getenv(f"TEOPARD_{name}_EXTRA_BUFFER_PCT")
    if raw is None or str(raw).strip() == "":
        raw = os.getenv("TEOPARD_EXTRA_TP_BUFFER_PCT", "0")
    try:
        pct = float(str(raw).strip()) / 100.0
    except Exception:
        pct = 0.0
    if not np.isfinite(pct) or pct < 0:
        return 0.0
    return min(pct, 0.20)


def _apply_extra_tp_buffer(tp: float, direction: str, tp_name: str) -> float:
    pct = _extra_user_tp_buffer_pct(tp_name)
    if pct <= 0:
        return float(tp)
    if direction == "LONG":
        return float(tp) * (1.0 + pct)
    if direction == "SHORT":
        return float(tp) * (1.0 - pct)
    return float(tp)


def _rr_guard_uses_extra_tp_buffer() -> bool:
    """Mặc định False trong V31: TP buffer là style user sau phân tích model.

    Nếu muốn RR guard dùng TP đã nới theo %, set:
      TEOPARD_RR_USE_EXTRA_TP_BUFFER=1
    """
    raw = (os.getenv("TEOPARD_RR_USE_EXTRA_TP_BUFFER") or "0").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _tp_for_rr_guard(pred: dict, key: str) -> float:
    if not _rr_guard_uses_extra_tp_buffer():
        before_key = f"_{key}_before_extra_buffer"
        if pred.get(before_key) is not None:
            try:
                return float(pred[before_key])
            except Exception:
                pass
    return float(pred[key])


def _apply_extra_tp_buffers_to_plan(pred: dict, output: str | None = None) -> tuple[dict, str | None]:
    """Nới TP1/TP2 sau khi đã chuẩn hóa target cấu trúc.

    Đây là lớp chỉnh style xuất lệnh/tracking giống TEOPARD_EXTRA_SL_BUFFER_PCT.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return pred, output
    if pred.get("tp1") is None and pred.get("tp2") is None:
        return pred, output

    normalized = dict(pred)
    changed = False
    for key in ("tp1", "tp2"):
        if normalized.get(key) is None:
            continue
        pct = _extra_user_tp_buffer_pct(key.upper())
        if pct <= 0:
            continue
        old_tp = float(normalized[key])
        new_tp = _apply_extra_tp_buffer(old_tp, direction, key.upper())
        normalized[f"_{key}_before_extra_buffer"] = old_tp
        normalized[f"_extra_{key}_buffer_pct"] = pct
        normalized[key] = new_tp
        changed = changed or abs(new_tp - old_tp) > max(abs(old_tp) * 1e-7, 1e-8)

    if changed:
        output = _update_output_trade_numbers(output or "", normalized)
    return normalized, output


def _collect_structural_levels(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    side: str,
) -> list[dict]:
    """Lấy các swing high/low gần nhất làm vùng invalidation.

    SCALP: ưu tiên 15M + 1H. SWING: ưu tiên 4H + 1D.
    Chỉ lấy nến đã đóng để SL không bị nhảy theo nến đang chạy.
    """
    main_label, structure_label, big_label = _mode_labels(mode)
    labels = [main_label, structure_label]
    if mode != "short":
        labels.append(big_label)

    levels: list[dict] = []
    for order, label in enumerate(dict.fromkeys(labels)):
        df = _closed_candles(timeframe_data.get(label))
        if df is None or df.empty:
            continue
        lookback = 120 if mode == "short" else 90
        left_right = 2 if label in ("15M", "1H") else 3
        for pvt in _find_pivots(df, side, lookback=lookback, left=left_right, right=left_right):
            levels.append({
                "price": float(pvt["price"]),
                "label": label,
                "kind": "pivot",
                "order": order,
                "time": pvt.get("time"),
            })

        # Fallback có kiểm soát: cực trị gần nhất của nến đã đóng, không phải liquidity box rộng.
        tail_n = 36 if mode == "short" else 30
        data = df.tail(tail_n)
        if not data.empty:
            col = "low" if side == "low" else "high"
            idx = data[col].idxmin() if side == "low" else data[col].idxmax()
            levels.append({
                "price": float(data.loc[idx, col]),
                "label": label,
                "kind": "recent_extreme",
                "order": order + 0.35,
                "time": data.loc[idx, "timestamp"] if "timestamp" in data.columns else None,
            })
    return levels



def _collect_recent_sweep_extremes(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    side: str,
    price_ref: float,
) -> list[dict]:
    """Lấy wick extreme của cú quét mới nhất để đặt SL đúng ngoài điểm vô hiệu.

    Lý do V15: model có thể vào lệnh dựa trên cú quét mới nhất của nến 15M/1H.
    Nếu Python chỉ lấy pivot đã đóng nến, SL có thể nằm *trên* đáy quét thật với LONG
    hoặc *dưới* đỉnh quét thật với SHORT. Khi đó chỉ cần giá retest wick là LOSS.
    Vì vậy SCALP được phép dùng wick extreme gần nhất làm invalidation, kể cả nến mới nhất,
    miễn là nó có râu đủ rõ và nằm trong khoảng hợp lý quanh Entry.
    """
    if not price_ref or price_ref <= 0:
        return []
    main_label, structure_label, _ = _mode_labels(mode)
    labels = [main_label, structure_label]
    results: list[dict] = []

    for order, label in enumerate(dict.fromkeys(labels)):
        df = timeframe_data.get(label)
        if df is None or df.empty:
            continue
        # Dùng cả nến mới nhất vì tín hiệu scalp thường xuất hiện ngay sau cú quét wick.
        data = df.tail(14 if mode == "short" else 10).reset_index(drop=True)
        if data.empty:
            continue
        total = len(data)
        for i, row in data.iterrows():
            try:
                high = float(row["high"]); low = float(row["low"]); close = float(row["close"])
            except Exception:
                continue
            upper_wick, lower_wick, body_pct, rng = _candle_wick_stats(row)
            vol_ratio = _safe_float(row.get("vol_ratio"), 1.0) or 1.0
            recency = i / max(total - 1, 1)

            if side == "low":
                # Quét xuống rồi đóng/lấy lại đáng kể khỏi đáy wick.
                reclaimed = close >= low + rng * 0.42
                is_sweep = lower_wick >= 0.34 and lower_wick >= body_pct * 0.75 and reclaimed
                if not is_sweep:
                    continue
                price = low
            else:
                # Quét lên rồi bị từ chối khỏi đỉnh wick.
                rejected = close <= high - rng * 0.42
                is_sweep = upper_wick >= 0.34 and upper_wick >= body_pct * 0.75 and rejected
                if not is_sweep:
                    continue
                price = high

            results.append({
                "price": float(price),
                "label": label,
                "kind": "recent_sweep_extreme",
                # order âm để cho biết đây là invalidation trực tiếp của setup hiện tại.
                "order": -0.35 + order * 0.05 - recency * 0.02,
                "time": row.get("timestamp"),
                "wick_pct": float(lower_wick if side == "low" else upper_wick),
                "vol_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else 1.0,
            })
    return results


def _extract_sweep_extreme_from_output(output: str | None, direction: str) -> float | None:
    """Bắt số 'quét đáy/đỉnh X' mà model dùng làm lý do vào lệnh.

    Đây chỉ là lớp bảo hiểm. Ưu tiên chính vẫn là dữ liệu nến. Nếu model đã nói rõ
    lệnh LONG dựa trên quét đáy 1,735.27 thì SL tuyệt đối không được nằm trên 1,735.27.
    """
    if not output:
        return None
    text = str(output)
    patterns = []
    if direction == "LONG":
        patterns = [
            r"quét\s+đáy\s*([0-9][0-9,\.]*)",
            r"đáy\s+quét\s*([0-9][0-9,\.]*)",
            r"sweep\s+low\s*([0-9][0-9,\.]*)",
        ]
    elif direction == "SHORT":
        patterns = [
            r"quét\s+đỉnh\s*([0-9][0-9,\.]*)",
            r"đỉnh\s+quét\s*([0-9][0-9,\.]*)",
            r"sweep\s+high\s*([0-9][0-9,\.]*)",
        ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group(1).replace(",", ""))
        except Exception:
            continue
        if np.isfinite(val) and val > 0:
            return val
    return None


def _nearest_invalidation_level(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> dict | None:
    """Tìm swing/invalidation gần nhất để đặt SL cấu trúc.

    LONG: chọn swing low gần nhất nằm dưới/trong vùng Entry, rồi SL = level - ATR buffer.
    SHORT: chọn swing high gần nhất nằm trên/trong vùng Entry, rồi SL = level + ATR buffer.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None
    if pred.get("entry_low") is None or pred.get("entry_high") is None:
        return None

    entry_low = float(pred["entry_low"])
    entry_high = float(pred["entry_high"])
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low

    price_ref = float(current_price) if current_price is not None else (entry_low + entry_high) / 2.0
    buffer = _structural_sl_buffer(timeframe_data, mode, price_ref)
    eps = max(price_ref * 0.00025, buffer * 0.15)

    if direction == "LONG":
        raw_levels = (
            _collect_recent_sweep_extremes(timeframe_data, mode, "low", price_ref)
            + _collect_structural_levels(timeframe_data, mode, "low")
        )
        candidates = [
            lv for lv in raw_levels
            if float(lv["price"]) <= entry_high + eps
        ]
        if not candidates:
            return None

        # Nếu setup vừa có cú quét đáy rõ, chính wick low đó là invalidation trực tiếp.
        # Không chọn một swing gần hơn nằm phía trên wick quét, vì như vậy SL sẽ quá sát.
        sweep_bound = max(buffer * 8.0, price_ref * 0.012, (entry_high - entry_low) + buffer * 5.0)
        sweep_candidates = [
            lv for lv in candidates
            if lv.get("kind") == "recent_sweep_extreme" and (entry_high - float(lv["price"])) <= sweep_bound
        ]
        if sweep_candidates:
            sweep_candidates.sort(key=lambda lv: (
                float(lv.get("order", 9)),
                abs(entry_low - float(lv["price"])),
                float(lv["price"]),
            ))
            chosen = sweep_candidates[0]
            sl = float(chosen["price"]) - buffer
            return {**chosen, "sl": sl, "buffer": buffer}

        # Gần nhất với Entry nhưng ưu tiên pivot hơn fallback khi mức giá gần như nhau.
        candidates.sort(key=lambda lv: (
            abs(entry_low - float(lv["price"])),
            0 if lv.get("kind") == "pivot" else 1,
            float(lv.get("order", 9)),
        ))
        chosen = candidates[0]
        sl = float(chosen["price"]) - buffer
        # Nếu SL vẫn nằm trong vùng Entry thì swing này quá gần/nhiễu, thử swing thấp hơn.
        if sl >= entry_low:
            lower = [lv for lv in candidates if float(lv["price"]) - buffer < entry_low]
            if not lower:
                return None
            lower.sort(key=lambda lv: (entry_low - float(lv["price"]), 0 if lv.get("kind") == "pivot" else 1))
            chosen = lower[0]
            sl = float(chosen["price"]) - buffer
        return {**chosen, "sl": sl, "buffer": buffer}

    raw_levels = (
        _collect_recent_sweep_extremes(timeframe_data, mode, "high", price_ref)
        + _collect_structural_levels(timeframe_data, mode, "high")
    )
    candidates = [
        lv for lv in raw_levels
        if float(lv["price"]) >= entry_low - eps
    ]
    if not candidates:
        return None

    # Nếu setup vừa có cú quét đỉnh rõ, chính wick high đó là invalidation trực tiếp.
    sweep_bound = max(buffer * 8.0, price_ref * 0.012, (entry_high - entry_low) + buffer * 5.0)
    sweep_candidates = [
        lv for lv in candidates
        if lv.get("kind") == "recent_sweep_extreme" and (float(lv["price"]) - entry_low) <= sweep_bound
    ]
    if sweep_candidates:
        sweep_candidates.sort(key=lambda lv: (
            float(lv.get("order", 9)),
            abs(float(lv["price"]) - entry_high),
            -float(lv["price"]),
        ))
        chosen = sweep_candidates[0]
        sl = float(chosen["price"]) + buffer
        return {**chosen, "sl": sl, "buffer": buffer}

    candidates.sort(key=lambda lv: (
        abs(float(lv["price"]) - entry_high),
        0 if lv.get("kind") == "pivot" else 1,
        float(lv.get("order", 9)),
    ))
    chosen = candidates[0]
    sl = float(chosen["price"]) + buffer
    if sl <= entry_high:
        higher = [lv for lv in candidates if float(lv["price"]) + buffer > entry_high]
        if not higher:
            return None
        higher.sort(key=lambda lv: (float(lv["price"]) - entry_high, 0 if lv.get("kind") == "pivot" else 1))
        chosen = higher[0]
        sl = float(chosen["price"]) + buffer
    return {**chosen, "sl": sl, "buffer": buffer}


def _rr_guard_uses_extra_sl_buffer() -> bool:
    """Có tính phần nới SL thêm vào RR guard hay không.

    Mặc định False theo phong cách của user: TEOPARD_EXTRA_SL_BUFFER_PCT là lớp đệm
    SL cuối cùng để tránh nhiễu, không được làm Python đổi kèo thành NO TRADE chỉ vì
    tỷ lệ lời/lỗ xấu đi sau khi cộng đệm.
    """
    raw = (os.getenv("TEOPARD_RR_USE_EXTRA_SL_BUFFER") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _sl_for_rr_guard(pred: dict) -> float:
    if not _rr_guard_uses_extra_sl_buffer() and pred.get("_sl_before_extra_buffer") is not None:
        try:
            return float(pred["_sl_before_extra_buffer"])
        except Exception:
            pass
    return float(pred["sl"])


def _plan_worst_case_risk_reward(
    pred: dict,
    *,
    use_rr_guard_sl: bool = False,
    use_rr_guard_tp: bool = False,
) -> dict:
    """RR bảo thủ theo mép Entry bất lợi nhất.

    LONG: giả sử fill ở mép cao của Entry. SHORT: giả sử fill ở mép thấp của Entry.
    Nếu use_rr_guard_sl=True, RR dùng SL cấu trúc gốc trước lớp đệm % thêm, trừ khi
    TEOPARD_RR_USE_EXTRA_SL_BUFFER=1.
    Nếu use_rr_guard_tp=True, RR dùng TP cuối đã nới %, trừ khi
    TEOPARD_RR_USE_EXTRA_TP_BUFFER=0.
    """
    direction = (pred.get("direction") or "").upper()
    try:
        entry_low = float(pred["entry_low"])
        entry_high = float(pred["entry_high"])
        if entry_low > entry_high:
            entry_low, entry_high = entry_high, entry_low
        sl = _sl_for_rr_guard(pred) if use_rr_guard_sl else float(pred["sl"])
        if use_rr_guard_tp:
            tp1 = _tp_for_rr_guard(pred, "tp1")
            tp2 = _tp_for_rr_guard(pred, "tp2")
        else:
            tp1 = float(pred["tp1"])
            tp2 = float(pred["tp2"])
    except Exception:
        return {}

    if direction == "LONG":
        risk = max(entry_high - sl, 0.0)
        reward1 = max(tp1 - entry_high, 0.0)
        reward2 = max(tp2 - entry_high, 0.0)
    elif direction == "SHORT":
        risk = max(sl - entry_low, 0.0)
        reward1 = max(entry_low - tp1, 0.0)
        reward2 = max(entry_low - tp2, 0.0)
    else:
        return {}

    return {
        "risk": risk,
        "reward1": reward1,
        "reward2": reward2,
        "rr1": reward1 / risk if risk > 0 else None,
        "rr2": reward2 / risk if risk > 0 else None,
    }




# V26 trade-leaning defaults: các ngưỡng được hạ thêm và có thể override bằng biến Railway.
# Vì từ V19 trở đi chỉ khi user bấm “Tôi đã trade theo lệnh này” bot mới lưu/theo dõi,
# Python guard không nên biến quá nhiều plan LONG/SHORT thành NO TRADE.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _guard_profile() -> str:
    return (os.getenv("TEOPARD_GUARD_PROFILE") or "loose").strip().lower()

def _guard_is_off() -> bool:
    return _guard_profile() in ("off", "trust", "model", "model_only")

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")

def _python_adjusts_model_sl() -> bool:
    """Mặc định tắt: model là nguồn quyết định SL, Python chỉ validate và áp buffer user.

    Bật TEOPARD_PYTHON_ADJUST_SL=1 nếu muốn quay lại kiểu Python ép SL theo
    swing/invalidation.
    """
    return _env_bool("TEOPARD_PYTHON_ADJUST_SL", False)

def _python_adjusts_model_tp() -> bool:
    """Mặc định tắt: model là nguồn quyết định TP, Python chỉ validate và áp buffer user.

    Bật TEOPARD_PYTHON_ADJUST_TP=1 nếu muốn Python tự nhảy TP sang target kế tiếp
    khi model đặt TP quá sát.
    """
    return _env_bool("TEOPARD_PYTHON_ADJUST_TP", False)

MIN_TP1_R = _env_float("TEOPARD_MIN_TP1_R", 0.40)
MIN_TP2_R = _env_float("TEOPARD_MIN_TP2_R", 0.50)
TP2_MIN_SEPARATION_MULT = _env_float("TEOPARD_TP2_SEPARATION_MULT", 1.00)

MIN_ACTION_CONFIDENCE_SCALP = _env_float("TEOPARD_MIN_SCALP_CONFIDENCE", 48.0)
MIN_REVERSAL_CONFIDENCE_SCALP = _env_float("TEOPARD_MIN_REVERSAL_CONFIDENCE", 50.0)
MIN_REVERSAL_CONFIDENCE_WITH_BAD_MOMENTUM = _env_float("TEOPARD_MIN_REVERSAL_BAD_MOMENTUM_CONFIDENCE", 52.0)


def _dedupe_price_candidates(candidates: list[dict], price_ref: float, risk: float) -> list[dict]:
    """Gộp các target gần như trùng nhau để TP không nhảy giữa vài mức sát nhau."""
    if not candidates:
        return []
    tol = max(price_ref * 0.00035, risk * 0.08, 1e-9)
    ordered = sorted(candidates, key=lambda c: float(c["price"]))
    unique: list[dict] = []
    for cand in ordered:
        price = float(cand["price"])
        matched = None
        for kept in unique:
            if abs(price - float(kept["price"])) <= tol:
                matched = kept
                break
        if matched is None:
            unique.append(dict(cand))
            continue
        # Ưu tiên pivot/liquidity/fib có score cao hơn; nếu tương đương giữ mức xa hơn theo source ổn định.
        if float(cand.get("score", 0.0)) > float(matched.get("score", 0.0)):
            matched.update(cand)
    return unique


def _collect_tp_target_candidates(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
    risk: float,
) -> list[dict]:
    """Thu thập target cấu trúc để sửa TP1/TP2 nếu model đặt quá sát.

    LONG lấy các swing high / liquidity trên / fib phía trên Entry.
    SHORT lấy các swing low / liquidity dưới / fib phía dưới Entry.
    Đây chỉ là target ứng viên; validator vẫn quyết định cuối cùng theo RR.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return []
    try:
        entry_low = float(pred["entry_low"])
        entry_high = float(pred["entry_high"])
    except Exception:
        return []
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
    edge = entry_high if direction == "LONG" else entry_low
    price_ref = float(current_price) if current_price is not None else (entry_low + entry_high) / 2.0

    candidates: list[dict] = []

    def add(price: float | None, source: str, score: float = 1.0) -> None:
        if price is None:
            return
        try:
            val = float(price)
        except Exception:
            return
        if not np.isfinite(val) or val <= 0:
            return
        if direction == "LONG" and val <= edge:
            return
        if direction == "SHORT" and val >= edge:
            return
        candidates.append({"price": val, "source": source, "score": score})

    # 1) Swing/pivot levels theo hướng TP.
    side = "high" if direction == "LONG" else "low"
    for lv in _collect_structural_levels(timeframe_data, mode, side):
        score = 2.0 if lv.get("kind") == "pivot" else 1.35
        # Khung lớn hơn đáng tin hơn cho TP xa, nhưng không để nó thắng mọi target gần.
        score += max(0.0, 0.25 - float(lv.get("order", 0)) * 0.05)
        add(float(lv["price"]), f"{lv.get('label')} {lv.get('kind')}", score)

    # 2) Fibonacci/biên cấu trúc từ khung xác nhận và khung lớn.
    _, structure_label, big_label = _mode_labels(mode)
    for label, weight in [(structure_label, 1.45), (big_label, 1.25)]:
        struct = _structure_info(timeframe_data.get(label), price_ref)
        for name, value in (struct.get("fib") or {}).items():
            add(value, f"Fib {label} {name}", weight)
        if direction == "LONG":
            add(struct.get("recent_high"), f"đỉnh gần {label}", weight + 0.15)
            add(struct.get("major_high"), f"biên cao {label}", weight + 0.05)
        else:
            add(struct.get("recent_low"), f"đáy gần {label}", weight + 0.15)
            add(struct.get("major_low"), f"biên thấp {label}", weight + 0.05)

    # 3) Liquidity box đối diện, nhưng chỉ làm target nếu nó đủ RR; không ép bám mép box hẹp.
    try:
        zones = _liquidity_zones_by_windows(timeframe_data, mode, price_ref)
        if direction == "LONG":
            for role, score in [("near", 1.6), ("main", 1.45), ("deep", 1.25)]:
                zone = zones.get(f"upper_{role}")
                if zone and zone[0] is not None and zone[1] is not None:
                    add(float(zone[0]), f"liquidity trên {role} mép gần", score)
                    add(float(zone[1]), f"liquidity trên {role} mép xa", score * 0.9)
        else:
            for role, score in [("near", 1.6), ("main", 1.45), ("deep", 1.25)]:
                zone = zones.get(f"lower_{role}")
                if zone and zone[0] is not None and zone[1] is not None:
                    add(float(zone[1]), f"liquidity dưới {role} mép gần", score)
                    add(float(zone[0]), f"liquidity dưới {role} mép xa", score * 0.9)
    except Exception:
        pass

    # Loại target cực xa bất thường so với risk/khung để tránh TP bị kéo ảo.
    # SCALP giữ target trong khoảng ~5R hoặc 2.8% giá; SWING rộng hơn.
    max_reward = max(risk * (5.0 if mode == "short" else 7.0), price_ref * (0.028 if mode == "short" else 0.12))
    bounded: list[dict] = []
    for c in candidates:
        reward = abs(float(c["price"]) - edge)
        if reward <= max_reward:
            bounded.append(c)
    candidates = bounded or candidates

    candidates = _dedupe_price_candidates(candidates, price_ref, risk)
    if direction == "LONG":
        return sorted(candidates, key=lambda c: (float(c["price"]) - edge, -float(c.get("score", 0.0))))
    return sorted(candidates, key=lambda c: (edge - float(c["price"]), -float(c.get("score", 0.0))))


def _pick_tp_candidate(candidates: list[dict], direction: str, threshold_price: float) -> dict | None:
    if direction == "LONG":
        valid = [c for c in candidates if float(c["price"]) >= threshold_price]
        return min(valid, key=lambda c: float(c["price"])) if valid else None
    valid = [c for c in candidates if float(c["price"]) <= threshold_price]
    return max(valid, key=lambda c: float(c["price"])) if valid else None


def _normalize_trade_plan_structural_tps(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
    output: str | None = None,
) -> tuple[dict, str | None]:
    """Nếu TP1/TP2 quá sát sau khi SL đã chuẩn hóa, thử chuyển sang target cấu trúc kế tiếp.

    Nếu không có target cấu trúc đủ RR, validator sẽ reject thành NO TRADE.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return pred, output
    required = ("entry_low", "entry_high", "sl", "tp1", "tp2")
    if any(pred.get(k) is None for k in required):
        return pred, output

    normalized = dict(pred)
    rr = _plan_worst_case_risk_reward(normalized, use_rr_guard_sl=True)
    risk = rr.get("risk")
    if not risk or risk <= 0:
        return normalized, output

    entry_low = float(normalized["entry_low"])
    entry_high = float(normalized["entry_high"])
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
    edge = entry_high if direction == "LONG" else entry_low
    candidates = _collect_tp_target_candidates(normalized, timeframe_data, mode, current_price, float(risk))

    reward1 = rr.get("reward1") or 0.0
    reward2 = rr.get("reward2") or 0.0
    rr1 = rr.get("rr1") or 0.0
    rr2 = rr.get("rr2") or 0.0

    if rr1 < MIN_TP1_R:
        threshold = edge + risk * MIN_TP1_R if direction == "LONG" else edge - risk * MIN_TP1_R
        cand = _pick_tp_candidate(candidates, direction, threshold)
        if cand:
            old_tp1 = float(normalized["tp1"])
            normalized["tp1"] = float(cand["price"])
            normalized["_tp1_adjusted_by_structure"] = True
            normalized["_tp1_source"] = cand.get("source")
            normalized["_tp1_old"] = old_tp1

    # Tính lại sau khi có thể đã sửa TP1.
    rr = _plan_worst_case_risk_reward(normalized, use_rr_guard_sl=True)
    risk = rr.get("risk") or risk
    reward1 = rr.get("reward1") or reward1
    reward2 = rr.get("reward2") or reward2
    rr2 = rr.get("rr2") or 0.0

    need_tp2 = rr2 < MIN_TP2_R or reward2 <= reward1 * TP2_MIN_SEPARATION_MULT
    if need_tp2:
        min_reward2 = max(risk * MIN_TP2_R, reward1 * TP2_MIN_SEPARATION_MULT)
        threshold = edge + min_reward2 if direction == "LONG" else edge - min_reward2
        cand = _pick_tp_candidate(candidates, direction, threshold)
        if cand:
            old_tp2 = float(normalized["tp2"])
            normalized["tp2"] = float(cand["price"])
            normalized["_tp2_adjusted_by_structure"] = True
            normalized["_tp2_source"] = cand.get("source")
            normalized["_tp2_old"] = old_tp2

    output = _update_output_trade_numbers(output or "", normalized)
    return normalized, output

def _update_output_trade_numbers(output: str, pred: dict) -> str:
    """Đồng bộ SL/TP/Rủi ro trong câu trả lời sau khi Python chuẩn hóa plan."""
    text = output or ""
    if pred.get("sl") is not None:
        text = re.sub(r"(\bSL\s*:\s*)([0-9,\.]+)", lambda m: m.group(1) + fmt(float(pred["sl"])), text, count=1, flags=re.IGNORECASE)
    if pred.get("tp1") is not None:
        text = re.sub(r"(\bTP1\s*:\s*)([0-9,\.]+)", lambda m: m.group(1) + fmt(float(pred["tp1"])), text, count=1, flags=re.IGNORECASE)
    if pred.get("tp2") is not None:
        text = re.sub(r"(\bTP2\s*:\s*)([0-9,\.]+)", lambda m: m.group(1) + fmt(float(pred["tp2"])), text, count=1, flags=re.IGNORECASE)
    rr = _plan_worst_case_risk_reward(pred)
    if rr.get("risk") is not None:
        text = re.sub(
            r"(Rủi ro\s+mỗi\s+lệnh\s*:\s*)~?[^\n]+",
            lambda m: m.group(1) + f"~{fmt(rr['risk'])} USDT",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return text


def _normalize_trade_plan_structural_sl(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
    output: str | None = None,
) -> tuple[dict, str | None]:
    """Ép SL theo cấu trúc: swing/invalidation gần nhất ± ATR buffer.

    Không kéo SL đại để cứu lệnh. Sau khi thay SL, validator vẫn kiểm tra RR.
    Nếu không tìm được swing/invalidation đủ rõ thì plan sẽ bị reject thành NO TRADE.
    """
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return pred, output
    required = ("entry_low", "entry_high", "sl", "tp1", "tp2")
    if any(pred.get(k) is None for k in required):
        return pred, output

    normalized = dict(pred)
    inv = _nearest_invalidation_level(normalized, timeframe_data, mode, current_price)
    if not inv:
        normalized["_structural_sl_error"] = "Không tìm được swing/invalidation đã đóng nến đủ rõ để đặt SL cấu trúc."
        return normalized, output

    old_sl = float(normalized["sl"])
    new_sl = float(inv["sl"])
    inv_price = float(inv["price"])
    buffer = float(inv["buffer"])

    # Lớp bảo hiểm: nếu model nói rõ lệnh dựa trên quét đáy/đỉnh X, SL phải nằm ngoài X.
    # Ví dụ LONG dựa vào quét đáy 1,735.27 thì SL không được là 1,737.xx.
    model_extreme = _extract_sweep_extreme_from_output(output, direction)
    if model_extreme is not None:
        entry_low = float(normalized["entry_low"]); entry_high = float(normalized["entry_high"])
        if entry_low > entry_high:
            entry_low, entry_high = entry_high, entry_low
        price_ref = float(current_price) if current_price is not None else (entry_low + entry_high) / 2.0
        sane_dist = max(buffer * 9.0, price_ref * 0.014, (entry_high - entry_low) + buffer * 6.0)
        if direction == "LONG" and model_extreme < inv_price and (entry_high - model_extreme) <= sane_dist:
            inv_price = float(model_extreme)
            new_sl = inv_price - buffer
            inv = {**inv, "price": inv_price, "kind": "model_sweep_extreme", "label": "output"}
        elif direction == "SHORT" and model_extreme > inv_price and (model_extreme - entry_low) <= sane_dist:
            inv_price = float(model_extreme)
            new_sl = inv_price + buffer
            inv = {**inv, "price": inv_price, "kind": "model_sweep_extreme", "label": "output"}

    # V31: không áp style buffer trong hàm chuẩn hóa cấu trúc.
    # Buffer theo sở thích user được áp riêng ở cuối flow để model/Python base-plan
    # không bị trộn với lớp quản trị rủi ro cá nhân.

    normalized["_structural_invalidation_level"] = float(inv_price)
    normalized["_structural_sl_buffer"] = float(buffer)
    normalized["_structural_sl_source"] = f"{inv.get('label')} {inv.get('kind')}"

    # SL luôn lấy theo cấu trúc gần nhất ± buffer. Nếu RR không còn đạt, plan bị NO TRADE.
    if abs(new_sl - old_sl) > max(abs(old_sl) * 1e-7, 1e-8):
        normalized["sl"] = new_sl
        normalized["_sl_adjusted_by_structure"] = True
        output = _update_output_trade_numbers(output or "", normalized)
    else:
        normalized["sl"] = old_sl
        output = _update_output_trade_numbers(output or "", normalized)

    return normalized, output



def _analysis_row(df: pd.DataFrame | None):
    """
    Dùng nến đã đóng gần nhất để đọc indicator/volume.

    Binance thường trả kèm nến hiện tại đang chạy; volume của nến này rất thấp
    nếu vừa mở nến, dễ làm Claude hiểu nhầm là thanh khoản yếu và chọn NO TRADE.
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
    elif vol_ratio <= 0.55:
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
    """Gửi nến thô đã đóng. Nến đang chạy được tách riêng để model không coi là xác nhận."""
    main_label, structure_label, big_label = _mode_labels(mode)
    trigger_label = _mode_trigger_label(mode)
    if mode == "short":
        ordered = [("15M", 24, "trigger/timing"), ("1H", 24, "setup chính"), ("4H", 12, "trend filter"), ("1D", 8, "macro context")]
    else:
        ordered = [("1H", 12, "trigger phụ"), ("4H", 24, "setup"), ("1D", 18, "decision chính"), ("1W", 12, "macro context")]

    blocks = ["RAW_CANDLE_CONTEXT_CHON_LOC — CHỈ NẾN ĐÃ ĐÓNG:"]
    blocks.append(f"- Vai trò khung: {_mode_role_text(mode)}")
    for label, n, role in ordered:
        df = timeframe_data.get(label)
        closed_df = _closed_candles(df)
        if closed_df is None or closed_df.empty:
            blocks.append(f"- {label} ({role}): Không đủ dữ liệu nến đã đóng.")
            continue
        rows = ["  " + _format_candle_compact(row) for _, row in closed_df.tail(n).iterrows()]
        blocks.append(f"- {label} ({role}): {min(n, len(closed_df))} nến đã đóng gần nhất, dùng để đọc phá giả/rút râu/đuối lực:")
        blocks.extend(rows)
    return "\n".join(blocks)


def build_live_candle_context(timeframe_data: dict[str, pd.DataFrame | None], mode: str) -> str:
    """Tách nến đang chạy khỏi nến đã đóng để chỉ dùng tham khảo, không xác nhận."""
    if mode == "short":
        labels = ["15M", "1H"]
    else:
        labels = ["1H", "4H"]
    blocks = ["LIVE_CANDLE_CONTEXT — NẾN ĐANG CHẠY, CHỈ THAM KHẢO:"]
    blocks.append("- Không dùng nến đang chạy để xác nhận entry/đảo chiều. Chỉ dùng để biết giá hiện tại đang di chuyển ra sao so với nến đã đóng.")
    for label in labels:
        df = timeframe_data.get(label)
        if df is None or df.empty or len(df) < 2:
            blocks.append(f"- {label}: Không đủ dữ liệu nến đang chạy.")
            continue
        row = df.iloc[-1]
        blocks.append(f"- {label} live: {_format_candle_compact(row)}")
    return "\n".join(blocks)




def _format_model_level_map(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    price: float,
    zones: dict | None = None,
    structure: dict | None = None,
) -> str:
    """Bản đồ mức giá cho model tự lập Entry/SL/TP.

    Mục tiêu V32: Python không tự sửa TP/SL mặc định, nên prompt phải đưa đủ
    candidate levels để model chọn TP/SL thực tế hơn. Các mức này KHÔNG được show
    trực tiếp ra user; chỉ dùng nội bộ để model lập plan.
    """
    zones = zones or _liquidity_zones_by_windows(timeframe_data, mode, price)
    main_label, structure_label, big_label = _mode_labels(mode)
    main_structure = _structure_info(timeframe_data.get(main_label), price)
    structure = structure or _structure_info(timeframe_data.get(structure_label), price)

    above: list[tuple[float, str]] = []
    below: list[tuple[float, str]] = []

    def add(level: float | None, name: str) -> None:
        try:
            val = float(level)
        except Exception:
            return
        if not np.isfinite(val) or val <= 0:
            return
        if val > price:
            above.append((val, name))
        elif val < price:
            below.append((val, name))

    # Cấu trúc/fib của khung setup/chính.
    if main_label != structure_label:
        add(main_structure.get("recent_high"), f"đỉnh gần {main_label}")
        add(main_structure.get("recent_low"), f"đáy gần {main_label}")
        add(main_structure.get("major_high"), f"biên cao {main_label}")
        add(main_structure.get("major_low"), f"biên thấp {main_label}")
        for k, v in (main_structure.get("fib") or {}).items():
            add(v, f"Fib {main_label} {k}")

    # Cấu trúc/fib của khung xác nhận.
    add(structure.get("recent_high"), f"đỉnh gần {structure_label}")
    add(structure.get("recent_low"), f"đáy gần {structure_label}")
    add(structure.get("major_high"), f"biên cao {structure_label}")
    add(structure.get("major_low"), f"biên thấp {structure_label}")
    for k, v in (structure.get("fib") or {}).items():
        add(v, f"Fib {structure_label} {k}")

    # Cấu trúc khung lớn hơn để có target không quá sát.
    big_structure = _structure_info(timeframe_data.get(big_label), price)
    add(big_structure.get("recent_high"), f"đỉnh gần {big_label}")
    add(big_structure.get("recent_low"), f"đáy gần {big_label}")
    add(big_structure.get("major_high"), f"biên cao {big_label}")
    add(big_structure.get("major_low"), f"biên thấp {big_label}")
    for k, v in (big_structure.get("fib") or {}).items():
        add(v, f"Fib {big_label} {k}")

    # Liquidity box: đưa cả mép gần/mép xa để model biết nếu mép gần quá sát thì chọn mép xa/target kế tiếp.
    for role in ("near", "main", "deep"):
        z = zones.get(f"upper_{role}")
        if z and z[0] is not None and z[1] is not None:
            add(float(z[0]), f"liq trên {role} mép gần")
            add(float(z[1]), f"liq trên {role} mép xa")
        z = zones.get(f"lower_{role}")
        if z and z[0] is not None and z[1] is not None:
            add(float(z[1]), f"liq dưới {role} mép gần")
            add(float(z[0]), f"liq dưới {role} mép xa")

    def compact(items: list[tuple[float, str]], side: str) -> str:
        if not items:
            return f"{side}: N/A"
        # Dedup mức gần nhau để prompt không loãng.
        items = sorted(items, key=lambda x: abs(x[0] - price))
        unique: list[tuple[float, str]] = []
        tol = max(price * 0.0005, 1e-9)
        for val, name in items:
            if any(abs(val - old_val) <= tol for old_val, _ in unique):
                continue
            unique.append((val, name))
            if len(unique) >= 12:
                break

        def item_text(v: float, name: str) -> str:
            dist = abs(v - price)
            pct = dist / max(price, 1e-12) * 100.0
            return f"{fmt(v)} ({name}, cách {fmt(dist)} / {pct:.2f}%)"

        return side + ": " + "; ".join(item_text(v, name) for v, name in unique)

    return "\n".join([
        "BẢN ĐỒ LEVEL CHO MODEL LẬP ENTRY/SL/TP — dùng nội bộ, không show user:",
        compact(above, "- Mức phía trên giá hiện tại: ứng viên TP LONG / SL SHORT / Entry SHORT"),
        compact(below, "- Mức phía dưới giá hiện tại: ứng viên TP SHORT / SL LONG / Entry LONG"),
        f"- Nguyên tắc chọn target: không lấy level gần nhất làm TP nếu nó chỉ là nhiễu nhỏ quanh Entry. Hãy chọn target ở kháng cự/hỗ trợ/Fibonacci/đỉnh đáy/vùng quét kế tiếp có ý nghĩa; nếu target gần nhất quá sát, bỏ qua và chọn bậc kế tiếp hoặc đổi sang lệnh chờ để có Entry tốt hơn.",
    ])


def _format_model_plan_contract(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    price: float,
) -> str:
    """Hợp đồng lập lệnh cho model: Python không tự sửa TP/SL nên model phải tự tính đúng.

    Block này không chốt lệnh thay model. Nó chỉ đưa ra quy trình và ngưỡng thực dụng
    để model tự chọn Entry/SL/TP từ các level đã có, tránh TP/SL quá sát hoặc bịa số.
    """
    main_label, structure_label, _ = _mode_labels(mode)
    trigger_label = _mode_trigger_label(mode)
    atr_trigger = _current_atr(timeframe_data.get(trigger_label)) or 0.0
    atr_main = _current_atr(timeframe_data.get(main_label)) or 0.0
    atr_structure = _current_atr(timeframe_data.get(structure_label)) or 0.0
    min_stop = _minimum_stop_distance(timeframe_data, mode, price)
    sl_buf = _structural_sl_buffer(timeframe_data, mode, price)
    risk_ref = _risk_floor(timeframe_data, mode, price)

    if mode == "short":
        # TP sát hơn mức này thường không đáng với phí/trượt giá/nhiễu scalp.
        tp1_noise_floor = max(price * 0.0018, atr_main * 0.45, min_stop * 0.80)
        tp2_noise_floor = max(price * 0.0035, atr_main * 0.90, min_stop * 1.25)
        entry_width_hint = max(price * 0.0007, atr_main * 0.20)
        label = "SCALP"
    else:
        tp1_noise_floor = max(price * 0.0060, atr_main * 0.60, min_stop * 0.85)
        tp2_noise_floor = max(price * 0.0120, atr_main * 1.10, min_stop * 1.35)
        entry_width_hint = max(price * 0.0020, atr_main * 0.25)
        label = "SWING"

    return "\n".join([
        "HỢP ĐỒNG LẬP LỆNH CHO MODEL — dùng nội bộ, không show user:",
        f"- Mode {label}: Python không tự cứu TP/SL mặc định. Nếu bạn xuất TP/SL quá sát hoặc sai cấu trúc, Python có thể đổi thành NO TRADE.",
        f"- Vai trò timeframe: {_mode_role_text(mode)}",
        f"- ATR tham chiếu: trigger {trigger_label}={fmt(atr_trigger)}, setup {main_label}={fmt(atr_main)}, structure {structure_label}={fmt(atr_structure)} | min_stop≈{fmt(min_stop)} | SL buffer cấu trúc≈{fmt(sl_buf)} | risk tham chiếu≈{fmt(risk_ref)}.",
        f"- Độ rộng Entry gợi ý: khoảng {fmt(entry_width_hint)}–{fmt(entry_width_hint * 2.2)} USDT tùy biến động; không làm Entry quá rộng để che sai điểm vào.",
        f"- TP1 không nên chỉ cách Entry vài tick. Trừ khi có cấu trúc cực rõ và SL rất ngắn, TP1 nên cách mép Entry bất lợi ít nhất khoảng max({fmt(tp1_noise_floor)} USDT, {fmt(MIN_TP1_R)}R) và phải trùng/tiệm cận một level thực tế.",
        f"- TP2 nên là bậc cấu trúc kế tiếp, tối thiểu khoảng max({fmt(tp2_noise_floor)} USDT, {fmt(MIN_TP2_R)}R). TP2 không phải số trang trí; nếu không có bậc kế tiếp, chọn TP2 theo biên lớn/Fib kế tiếp hoặc NO TRADE.",
        "- Quy trình bắt buộc: (1) chọn hướng; (2) chọn Entry; (3) chọn invalidation/SL ngoài đỉnh đáy hoặc râu quét; (4) tính risk theo mép Entry bất lợi; (5) chọn TP1/TP2 từ target ladder đủ xa và có lý do; (6) nếu RR xấu, ưu tiên đổi Entry thành lệnh chờ tốt hơn trước khi NO TRADE.",
        "- Không bịa SL/TP chỉ để đủ tỷ lệ. Nếu target thực tế không đủ xa hoặc SL cấu trúc làm kèo không đáng, hãy chọn NO TRADE thay vì đặt TP sát.",
    ])

def build_feature_engineering_block(
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
) -> str:
    main_label, structure_label, big_label = _mode_labels(mode)
    trigger_label = _mode_trigger_label(mode)
    price = current_price or _last_close_from_data(timeframe_data)
    if price is None:
        return "Dữ liệu kỹ thuật: Không đủ dữ liệu để tính cấu trúc, Fibonacci, ATR và vùng quét. Không được tự bịa các phần này."

    main_df = timeframe_data.get(main_label)
    trigger_df = timeframe_data.get(trigger_label)
    structure_df = timeframe_data.get(structure_label)
    if structure_df is None or structure_df.empty:
        structure_df = main_df
    atr_trigger = _current_atr(trigger_df)
    atr_main = _current_atr(main_df)
    atr_structure = _current_atr(structure_df)
    zones = _liquidity_zones_by_windows(timeframe_data, mode, price)
    structure = _structure_info(structure_df, price)
    risk = _risk_floor(timeframe_data, mode, price)

    fib = structure.get("fib", {})

    level_map = _format_model_level_map(timeframe_data, mode, price, zones, structure)
    plan_contract = _format_model_plan_contract(timeframe_data, mode, price)

    lines = [
        "Dữ liệu kỹ thuật do Python tính sẵn:",
        f"- Mode: {'SCALP' if mode == 'short' else 'SWING'} | Trigger: {trigger_label} | Khung setup/chính: {main_label} | Khung cấu trúc/xác nhận: {structure_label} | Khung lớn: {big_label}",
        f"- Vai trò timeframe: {_mode_role_text(mode)}",
        build_market_regime_block(timeframe_data, mode),
        level_map,
        plan_contract,
        f"- ATR14 {trigger_label}: {fmt(atr_trigger)} | ATR14 {main_label}: {fmt(atr_main)} | ATR14 {structure_label}: {fmt(atr_structure)} | Rủi ro tham chiếu: {fmt(risk)} USDT",
        f"- Trigger {trigger_label} đã đóng: {_consecutive_candles(trigger_df)} | {_wick_body_info(trigger_df)}",
        f"- Setup {main_label} đã đóng: {_consecutive_candles(main_df)} | {_wick_body_info(main_df)}",
        f"- Cấu trúc {structure_label}: {structure.get('trend', 'N/A')}; đỉnh/đáy gần {fmt(structure.get('recent_low'))}–{fmt(structure.get('recent_high'))}; biên lớn {fmt(structure.get('major_low'))}–{fmt(structure.get('major_high'))}",
        f"- Fibonacci {structure_label}: 0.382={fmt(fib.get('0.382'))}; 0.5={fmt(fib.get('0.5'))}; 0.618={fmt(fib.get('0.618'))}",
        _format_liquidity_window_line("Vùng thanh khoản dưới giá ước lượng", zones, "lower", price),
        _format_liquidity_window_line("Vùng thanh khoản trên giá ước lượng", zones, "upper", price),
        "- Vai trò vùng quét: Entry có thể tham khảo vùng gần/chính nếu hợp xu hướng và có xác nhận. Với SCALP, không dùng vùng thanh khoản dưới làm Entry LONG hoặc vùng thanh khoản trên làm Entry SHORT theo kiểu chạm-là-fill. Nếu cần thêm xác nhận, có thể ghi lệnh chờ kèm điều kiện rõ; chỉ chọn NO TRADE khi SL/TP, động lượng hoặc vùng vào không đạt.",
        "- TP không được ép bám sát mép box thanh khoản ước lượng. Nếu box đối diện quá gần làm RR xấu, chính model phải chủ động dùng swing high/low kế tiếp, Fibonacci, EMA/vùng cấu trúc kế tiếp hoặc vùng quét đối diện có ý nghĩa. Nếu không có target đủ đáng thì chọn NO TRADE. Không tạo TP quá gần chỉ vì box thanh khoản rất hẹp.",
        "- Quy tắc rủi ro V32: model là nguồn chính lập Entry/SL/TP. SL phải nằm ngoài swing high/low hoặc vùng invalidation gần nhất cộng/trừ ATR buffer; nếu setup dựa vào cú quét đáy/đỉnh mới nhất thì wick extreme của cú quét đó là invalidation trực tiếp và SL phải nằm ngoài wick đó. Python mặc định không tự sửa SL/TP sang mức khác; Python chỉ kiểm tra lỗi cứng rồi mới áp phần trăm buffer theo sở thích user nếu có. Vì vậy model phải chọn TP/SL hợp lý ngay từ đầu: TP1 nên >= 0.40R, TP2 nên >= 0.50R, nhưng ưu tiên target thực tế theo kháng cự/hỗ trợ/Fibonacci/swing/vùng quét thay vì target quá sát hoặc bịa.",
        "- Ghi chú: Vùng quét là vùng thanh khoản kỹ thuật ước lượng theo cửa sổ thời gian, không phải dữ liệu thanh lý thật hay liquidation heatmap. Block này là bản đồ kỹ thuật nội bộ, không phải lệnh giao dịch chốt sẵn. Không show trực tiếp các vùng thanh khoản/thanh lý/heatmap ra user; chỉ dùng chúng để lập quyết định, Entry/SL/TP, lý do và rủi ro.",
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

    closed_df = _closed_candles(df)
    if closed_df is None or closed_df.empty:
        closed_df = df
    window    = closed_df.tail(50)
    key_high  = window["high"].max()
    key_low   = window["low"].min()

    candles = "\n".join(
        f"  {str(row['timestamp'])[:16]} O:{fmt(row['open'])} H:{fmt(row['high'])} "
        f"L:{fmt(row['low'])} C:{fmt(row['close'])} "
        f"RSI14:{fmt(row['rsi_14'],1)} Vol:{fmt(row['vol_ratio'],2)}x"
        for _, row in closed_df.tail(10).iterrows()
    )

    return "\n".join([
        f"\nKHUNG {label}:",
        f"  Giá: {fmt(last['close'])} | Nến trước: {fmt(prev['close'])}",
        f"  EMA7={fmt(ema7)} EMA25={fmt(ema25)} EMA50={fmt(ema50)} → {ema_align}",
        f"  RSI(6)={fmt(last['rsi_6'],1)} RSI(14)={fmt(last['rsi_14'],1)}",
        f"  MACD={fmt(last['macd_line'],4)} Signal={fmt(last['macd_signal'],4)}; {macd_momentum_text(last['macd_hist'])} → {macd_dir}{macd_cross}",
        f"  ATR14={fmt(last.get('atr_14'))} ({fmt(last.get('atr_pct'),2)}%)",
        f"  Volume={fmt(last['vol_ratio'],2)}x → {vol_lbl}",
        f"  Nến đã đóng: {_consecutive_candles(df)} | {_wick_body_info(df)}",
        f"  High/Low 50 nến: {fmt(key_high)} / {fmt(key_low)}",
        f"  10 nến đã đóng gần nhất:",
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

    lines = [f"USER-SPECIFIC RECENT TRADED-ONLY LEARNING SUMMARY ({len(history)} user-confirmed trades for this symbol/mode only):"]
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

    lines.append("Use this traded-only user-specific summary as learning context. These are signals the user explicitly chose to trade/follow; do not learn from unconfirmed analyses and do not assume global user behavior.")
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
        "SCALP: dùng 1H làm setup/chính, 4H xác nhận xu hướng, 1D làm bối cảnh lớn; 15M chỉ timing entry/sweep, không đảo bias một mình."
        if mode == "short" else
        "SWING: dùng 1D làm xu hướng/chính, 4H làm setup/vùng vào, 1W làm bối cảnh lớn; 1H chỉ timing entry phụ."
    )

    history_block = format_prediction_history(history)
    open_signal_context = open_signal_context or "KẾ HOẠCH ĐANG MỞ: Không có kế hoạch đang chờ/đã khớp cho user này ở cùng coin và mode."
    tf_blocks     = "".join(summarize_timeframe(lbl, df) for lbl, df in timeframe_data.items())
    raw_candle_block = build_raw_candle_context(timeframe_data, mode)
    live_candle_block = build_live_candle_context(timeframe_data, mode)
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
{live_candle_block}
═══════════════════════════════
{tf_blocks}
═══════════════════════════════

Yêu cầu:
1. Python chỉ cung cấp dữ liệu cứng: EMA/RSI/MACD/ATR, market regime, cấu trúc, Fibonacci, vùng quét thanh khoản ước lượng, raw candle context, rủi ro tham chiếu. Không có kế hoạch LONG/SHORT chốt sẵn. Vùng quét/thanh khoản là dữ liệu nội bộ, không liệt kê ra user.
2. Model phải tự phân tích và tự lập Entry/SL/TP dựa trên dữ liệu cứng đó. Không được tự tạo thêm Fibonacci/vùng quét nếu block Python ghi N/A hoặc không đủ dữ liệu.
3. Trước khi quyết định, hãy so sánh NỘI BỘ 3 lựa chọn LONG / SHORT / NO TRADE theo xu hướng đa khung, vị trí giá, vùng quét ước lượng theo cửa sổ thời gian, Fibonacci, nến thô, volume và lịch sử cùng user. Không in bảng so sánh này ra user, không in mục thanh khoản/heatmap/vùng thanh lý.
4. Chỉ chọn LONG hoặc SHORT khi một hướng có lợi thế rõ hơn hướng còn lại, Entry hợp lý và tỷ lệ lời/lỗ đạt yêu cầu. Nếu thị trường nhiễu, xác suất chỉ ngang nhau, vùng vào lệnh không rõ, hoặc Entry/SL/TP bị gượng ép → chọn NO TRADE. Không dùng NO TRADE chỉ vì giá chưa chạm Entry; nếu Entry là vùng tốt nhưng giá chưa tới, hãy đưa lệnh chờ. Dùng nến đã đóng làm cơ sở xác nhận; nến live chỉ tham khảo.
5. Cách dùng vùng quét: Entry ưu tiên vùng gần/chính nếu hợp hướng setup và có xác nhận. Với SCALP, không được LONG chỉ vì giá chạm vùng thanh khoản dưới và không được SHORT chỉ vì giá chạm vùng thanh khoản trên; cần có lợi thế rõ như quét thanh khoản/rút râu/đóng nến xác nhận, hoặc một vùng chờ hợp lý với SL/TP đạt tỷ lệ. Nếu còn thiếu xác nhận, được phép đưa lệnh chờ với điều kiện kích hoạt rõ; chỉ chọn NO TRADE khi cả Entry, SL/TP hoặc động lượng đều không đủ.
6. TP/SL do model tự chọn từ dữ liệu cứng: Fibonacci, swing high/low, EMA/vùng cấu trúc, vùng quét ước lượng và raw candle context. Python mặc định không tự “cứu” TP/SL bằng cách nhảy sang số khác; nếu model đặt TP quá gần Entry, SL sai cấu trúc hoặc target không đáng đánh thì plan sẽ bị NO TRADE. Với LONG, TP1 nên là kháng cự/vùng hồi hợp lý phía trên; với SHORT, TP1 nên là hỗ trợ/vùng hút phía dưới. Không đặt TP1 quá sát chỉ để có lệnh. Ví dụ ETH entry 1700 mà TP1 1710 chỉ hợp lý nếu risk rất nhỏ và có cấu trúc rõ; nếu không, phải chọn target xa hơn như 1730–1750 theo kháng cự/Fibonacci/swing/vùng quét, hoặc NO TRADE. Sau khi model lập plan, Python chỉ áp buffer % theo biến Railway nếu user muốn.
7. Không mặc định mọi tín hiệu thành lệnh chờ. Nếu giá hiện tại đang nằm trong vùng Entry hợp lý và tín hiệu xác nhận đã đủ, hãy đặt Entry bao quanh/sát giá hiện tại và ghi “Có thể vào ngay trong vùng Entry...”. Nếu Entry chưa chạm giá hiện tại nhưng vẫn là vùng đẹp, giữ quyết định LONG/SHORT dạng lệnh chờ và ghi rõ “Chưa vào ngay, chờ giá về vùng Entry...”, không chọn NO TRADE chỉ vì chưa chạm Entry.
8. Nếu giá hiện tại chưa vào vùng Entry hoặc còn thiếu xác nhận, mới ghi “Lệnh chờ, chưa vào ngay...” và nêu rõ điều kiện chờ.
9. Nếu chọn LONG/SHORT: Entry/SL/TP phải hợp logic với hướng giao dịch và tham chiếu ATR/giá. Không đặt SL quá sát; nếu phải đặt SL quá sát mới có tỷ lệ đẹp thì chọn NO TRADE. Không kéo SL/TP quá xa chỉ để đạt tỷ lệ lời/lỗ đẹp, nhưng cũng không đặt TP quá gần khiến lợi nhuận không đáng so với phí/trượt giá/rủi ro nhiễu. SCALP không được bị 15M live làm đảo quyết định nếu 1H/4H chưa xác nhận; SWING không được bị 1H live làm đảo bias nếu 1D/1W chưa đổi.
10. Nếu chọn NO TRADE: không cần Entry/SL/TP; trả quyết định NO TRADE và lý do ngắn. Python sẽ không gửi plan đó thành tín hiệu. Được chọn NO TRADE khi lợi thế chưa đủ rõ, kể cả khi vẫn có thể vẽ ra một vùng Entry hợp lệ nhưng kèo không đáng vào.
11. Đọc kỹ RECENT LEARNING SUMMARY, đặc biệt Decision why, Outcome, Market then và Feature then, nhưng không hiện mục “Nhìn lại lịch sử” trong câu trả lời.
12. Đọc kỹ KẾ HOẠCH ĐANG MỞ nếu có. Không được hiểu vùng Entry của một lệnh chờ LONG là mục tiêu TP cho lệnh SHORT ngược lại, hoặc vùng Entry của lệnh chờ SHORT là mục tiêu TP cho lệnh LONG ngược lại.
13. Nếu đang có kế hoạch cũ PENDING_ENTRY mà giá đã chạy xa khỏi Entry theo đúng hướng dự báo, không được đuổi giá chỉ vì giá chạy. Chỉ cho vào ngay khi có vùng Entry mới bao quanh giá hiện tại và xác nhận rõ; nếu Entry còn tốt nhưng chưa tới giá thì ghi lệnh chờ, không ép thành NO TRADE chỉ vì Entry xa.
14. Nếu kế hoạch mới thay thế kế hoạch cũ, ghi ngắn trong “📊 Kịch bản chính” lý do kế hoạch cũ bị hủy/thay thế.
15. Không copy phân tích cũ. Chỉ dùng summary để tránh lặp lại lỗi.
16. QUYẾT ĐỊNH cuối cùng chỉ được là LONG, SHORT hoặc NO TRADE. Không dùng “CHỜ” làm quyết định cuối cùng.
17. Format output cho user chỉ giữ quyết định, Entry/SL/TP nếu có, Lý do và Rủi ro. Không in mục “💧 Thanh khoản”, không liệt kê vùng dưới/trên/gần/chính/sâu.
"""


# ─── Tóm tắt reasoning bằng call Haiku thứ 2 (rất ngắn, rẻ) ─────────────────

def get_ai_api_key() -> str | None:
    """Trả về API key theo provider hiện tại."""
    if _is_openrouter_provider():
        return OPENROUTER_API_KEY
    if _is_zai_provider():
        return ZAI_API_KEY
    return ANTHROPIC_API_KEY


def get_ai_model_name() -> str:
    if _is_openrouter_provider():
        return OPENROUTER_MODEL
    if _is_zai_provider():
        return ZAI_MODEL
    return CLAUDE_MODEL


def get_ai_provider_label() -> str:
    if _is_openrouter_provider():
        return "openrouter"
    if _is_zai_provider():
        return "zai"
    return "anthropic"


def ensure_ai_config() -> None:
    if _is_openrouter_provider():
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY. Set AI_PROVIDER=openrouter and OPENROUTER_API_KEY in Railway variables.")
        return
    if _is_zai_provider():
        if not ZAI_API_KEY:
            raise RuntimeError("Missing ZAI_API_KEY. Set AI_PROVIDER=zai and ZAI_API_KEY in Railway variables.")
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
    if anthropic is None:
        raise RuntimeError("Anthropic SDK is not installed. This build is intended to run with AI_PROVIDER=zai.")
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


def _zai_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    """Gọi Z.AI native/chính chủ bằng OpenAI-compatible Chat Completions HTTP API."""
    headers = {
        "Authorization": f"Bearer {ZAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
        "X-Client-Name": ZAI_APP_NAME,
    }

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    payload = {
        "model": ZAI_MODEL,
        "messages": payload_messages,
        "max_tokens": max_tokens,
    }

    # Z.AI dùng top-level reasoning_effort + thinking.
    # Summary mặc định truyền none/off để không tốn token và giảm latency.
    if reasoning_effort is None:
        effective_reasoning_effort = (ZAI_REASONING_EFFORT or "high").strip()
    else:
        effective_reasoning_effort = (reasoning_effort or "").strip()

    effort_norm = effective_reasoning_effort.lower()
    if effort_norm in ("", "off", "false", "0", "disabled"):
        payload["thinking"] = {"type": "disabled"}
        effective_effort_for_log = "off"
    else:
        # Z.AI docs hỗ trợ max/xhigh/high/medium/low/minimal/none cho GLM-5.2.
        payload["thinking"] = {"type": "enabled"}
        payload["reasoning_effort"] = effort_norm
        effective_effort_for_log = effort_norm

    r = requests.post(
        f"{ZAI_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    try:
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Z.AI API error: {r.status_code} - {r.text[:1000]}") from exc

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
        "effort": effective_effort_for_log,
    }


def llm_create_once(
    system: str | None,
    messages: list,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    ensure_ai_config()
    if _is_openrouter_provider():
        return _openrouter_create_once(system, messages, max_tokens, timeout, reasoning_effort=reasoning_effort)
    if _is_zai_provider():
        return _zai_create_once(system, messages, max_tokens, timeout, reasoning_effort=reasoning_effort)
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
                f"[LLM_RESPONSE] call_type={call_type} provider={get_ai_provider_label()} model={get_ai_model_name()} "
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
        if _is_openrouter_provider():
            summary_effort = OPENROUTER_SUMMARY_REASONING_EFFORT
        elif _is_zai_provider():
            summary_effort = ZAI_SUMMARY_REASONING_EFFORT
        else:
            summary_effort = ANTHROPIC_SUMMARY_EFFORT
        text = create_with_continuation(
            system=None,
            messages=[{
                "role": "user",
                "content": (
                    "Tóm tắt trong 1-2 câu (tối đa 60 từ) lý do kỹ thuật chính "
                    "dẫn đến quyết định LONG/SHORT/NO TRADE trong phân tích sau. "
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
    m = re.search(r"QUYẾT ĐỊNH[:\s]+(LONG|SHORT|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)", output, re.IGNORECASE)
    if m:
        raw_direction = m.group(1).upper().replace("-", "_").replace(" ", "_")
        direction = "NO_TRADE" if raw_direction in ("NO_TRADE", "NO__TRADE", "KHÔNG_VÀO_LỆNH", "KHONG_VAO_LENH") else raw_direction
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

    confidence = None
    # Bắt phần trăm tin cậy từ dòng quyết định hoặc dòng LONG/SHORT.
    # Ví dụ: "🏆 QUYẾT ĐỊNH: LONG — 55%" hoặc "📈 LONG — 62%".
    conf_patterns = [
        r"QUYẾT\s+ĐỊNH[:\s]+(?:LONG|SHORT|NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)\s*[—\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        r"(?:📈|📉)?\s*(?:LONG|SHORT|NO[_\s-]?TRADE)\s*[—\-]\s*([0-9]+(?:\.[0-9]+)?)\s*%",
    ]
    for pat in conf_patterns:
        cm = re.search(pat, output, flags=re.IGNORECASE)
        if cm:
            try:
                confidence = float(cm.group(1))
                break
            except Exception:
                pass

    return {
        "direction":  direction,
        "confidence": confidence,
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
    }


# ─── Guard lệnh giao dịch trước khi lưu auto-check ────────────────────────────

def _entry_contains_price(pred: dict, current_price: float | None) -> bool:
    if current_price is None:
        return False
    entry_low = pred.get("entry_low")
    entry_high = pred.get("entry_high")
    if entry_low is None or entry_high is None:
        return False
    return float(entry_low) <= float(current_price) <= float(entry_high)


def _distance_price_to_entry(pred: dict, current_price: float | None) -> float | None:
    if current_price is None:
        return None
    entry_low = pred.get("entry_low")
    entry_high = pred.get("entry_high")
    if entry_low is None or entry_high is None:
        return None
    entry_low = float(entry_low)
    entry_high = float(entry_high)
    price = float(current_price)
    if entry_low <= price <= entry_high:
        return 0.0
    if price < entry_low:
        return entry_low - price
    return price - entry_high



def _output_claims_immediate_entry(output: str | None) -> bool:
    """True nếu lời giải thích nói đang/có thể vào ngay."""
    if not output:
        return False
    text = str(output).lower()
    return any(k in text for k in (
        "vào ngay",
        "có thể vào ngay",
        "có thể vào lệnh ngay",
        "giá hiện tại đang nằm",
        "đang nằm trong vùng entry",
        "đang trong vùng entry",
    ))


def _normalize_pending_entry_activation(
    output: str,
    pred: dict,
    current_price: float | None,
) -> str:
    """Nếu Entry chưa tới nhưng model ghi 'vào ngay', đổi thành lệnh chờ.

    Đây là bản sửa V29: Entry xa giá hiện tại không phải lý do NO TRADE.
    Kế hoạch vẫn có thể là LONG/SHORT dạng PENDING_ENTRY; chỉ cần sửa wording
    kích hoạt để user hiểu là phải chờ giá về Entry, không vào market ngay.
    """
    if not output or current_price is None:
        return output
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return output
    entry_low = pred.get("entry_low")
    entry_high = pred.get("entry_high")
    if entry_low is None or entry_high is None:
        return output

    try:
        low = float(entry_low)
        high = float(entry_high)
        price = float(current_price)
    except Exception:
        return output
    if low > high:
        low, high = high, low
    if low <= price <= high:
        return output
    if not _output_claims_immediate_entry(output):
        return output

    if price < low:
        relation = "giá hiện tại còn thấp hơn vùng Entry"
    else:
        relation = "giá hiện tại đã vượt khỏi vùng Entry"
    wait_line = (
        f"Kích hoạt: Chưa vào ngay. Chờ giá về vùng Entry {fmt(low)}–{fmt(high)} "
        f"và có nến xác nhận đúng hướng; {relation} tại {fmt(price)}, không đuổi giá."
    )

    # Thay toàn bộ dòng Kích hoạt cũ nếu có.
    if re.search(r"^\s*Kích hoạt:\s*.*$", output, flags=re.IGNORECASE | re.MULTILINE):
        output = re.sub(
            r"^\s*Kích hoạt:\s*.*$",
            wait_line,
            output,
            count=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    else:
        # Nếu model thiếu dòng Kích hoạt, chèn sau Rủi ro mỗi lệnh hoặc sau TP2.
        if re.search(r"^\s*Rủi ro mỗi lệnh:\s*.*$", output, flags=re.IGNORECASE | re.MULTILINE):
            output = re.sub(
                r"(^\s*Rủi ro mỗi lệnh:\s*.*$)",
                r"\1\n" + wait_line,
                output,
                count=1,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        else:
            output = re.sub(
                r"(^\s*TP2:\s*.*$)",
                r"\1\n" + wait_line,
                output,
                count=1,
                flags=re.IGNORECASE | re.MULTILINE,
            )
    return output



def _output_mentions_reversal_entry(output: str | None, direction: str) -> bool:
    """Nhận diện setup scalp đảo chiều/bắt đáy-bắt đỉnh từ lời giải thích của model.

    Guard này phải đối xứng:
    - LONG bắt đáy / quét đáy / râu dưới / quá bán.
    - SHORT bắt đỉnh / quét đỉnh / râu trên / quá mua / bị từ chối vùng trên.

    Dùng nhiều biến thể từ ngữ vì model có thể diễn đạt khác nhau sau mỗi lần gọi.
    """
    if not output:
        return False
    text = str(output).lower()
    if direction == "LONG":
        keys = (
            "quét đáy", "đáy quét", "quét vùng dưới", "quét hỗ trợ",
            "rút râu dưới", "râu dưới", "wick dưới", "lower wick",
            "bắt đáy", "quá bán", "oversold", "hồi phục", "bật lên",
            "lấy lại vùng", "đóng lại trên", "reclaim",
        )
    elif direction == "SHORT":
        keys = (
            "quét đỉnh", "đỉnh quét", "quét vùng trên", "quét kháng cự",
            "rút râu trên", "râu trên", "wick trên", "upper wick",
            "bắt đỉnh", "quá mua", "overbought", "đảo chiều giảm",
            "bị từ chối", "từ chối vùng trên", "từ chối kháng cự",
            "đóng lại dưới", "mất vùng", "reject", "rejection",
        )
    else:
        return False
    return any(k in text for k in keys)


def _closed_row_momentum_flags(timeframe_data: dict[str, pd.DataFrame | None], direction: str) -> dict:
    """Đọc động lượng từ nến đã đóng để guard các lệnh scalp đảo chiều quá sớm."""
    flags = {
        "m15_against": False,
        "h1_against": False,
        "m15_ema_against": False,
        "h1_ema_against": False,
        "m15_vol": None,
        "h1_vol": None,
    }
    for label, prefix in (("15M", "m15"), ("1H", "h1")):
        row = _analysis_row(timeframe_data.get(label))
        if row is None:
            continue
        macd = _safe_float(row.get("macd_hist"), 0.0) or 0.0
        e7 = _safe_float(row.get("ema_7")); e25 = _safe_float(row.get("ema_25")); e50 = _safe_float(row.get("ema_50"))
        vol = _safe_float(row.get("vol_ratio"), None)
        flags[f"{prefix}_vol"] = vol
        if direction == "LONG":
            flags[f"{prefix}_against"] = macd < 0
            flags[f"{prefix}_ema_against"] = e7 is not None and e25 is not None and e50 is not None and e7 < e25 < e50
        elif direction == "SHORT":
            flags[f"{prefix}_against"] = macd > 0
            flags[f"{prefix}_ema_against"] = e7 is not None and e25 is not None and e50 is not None and e7 > e25 > e50
    return flags


def _validate_scalp_reversal_quality(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    output: str | None,
) -> list[str]:
    """Guard riêng cho setup SCALP đảo chiều.

    V23 tuning:
    - Không reject LONG/SHORT chỉ vì confidence 51-55 nữa; user đã tự bấm xác nhận trade
      trước khi bot lưu history, nên output được phép là kế hoạch tham khảo nếu SL/RR đạt.
    - Chỉ reject cứng khi confidence quá thấp, hoặc setup bắt đáy/bắt đỉnh còn bị 15M/1H
      chống lại rõ ràng và volume xác nhận yếu.
    - Guard SL cấu trúc và RR vẫn nằm ở _validate_actionable_trade_plan, không nới bừa.
    """
    errors: list[str] = []
    if _guard_is_off():
        return errors
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return errors

    conf = pred.get("confidence")
    try:
        conf_val = float(conf) if conf is not None else None
    except Exception:
        conf_val = None

    is_reversal = _output_mentions_reversal_entry(output, direction)

    # Chỉ confidence rất thấp mới bị chặn toàn cục. V17 dùng 58 làm hard floor nên bot quá NO TRADE.
    if conf_val is not None and conf_val < MIN_ACTION_CONFIDENCE_SCALP:
        errors.append(
            f"Tín hiệu SCALP chỉ {conf_val:.1f}%, dưới ngưỡng tối thiểu {MIN_ACTION_CONFIDENCE_SCALP:.1f}% để lưu thành lệnh thật."
        )
        return errors

    if not is_reversal:
        return errors

    flags = _closed_row_momentum_flags(timeframe_data, direction)
    against_count = sum(bool(flags.get(k)) for k in ("m15_against", "h1_against", "m15_ema_against"))
    vol_values = [v for v in (flags.get("m15_vol"), flags.get("h1_vol")) if v is not None and np.isfinite(v)]
    max_vol = max(vol_values) if vol_values else None
    weak_confirm_volume = max_vol is None or max_vol < _env_float("TEOPARD_WEAK_CONFIRM_VOLUME", 0.45)

    # V26: đảo chiều scalp không bị chặn chỉ vì volume/MACD chưa đẹp.
    # Chỉ chặn khi confidence rất thấp và dữ liệu 15M/1H chống lại cực rõ.
    if (
        conf_val is not None
        and conf_val < MIN_REVERSAL_CONFIDENCE_SCALP
        and against_count >= 3
        and weak_confirm_volume
    ):
        errors.append(
            f"Setup SCALP đảo chiều chỉ {conf_val:.1f}% và còn bị 15M/1H chống lại rõ với volume xác nhận rất yếu."
        )

    if (
        conf_val is not None
        and conf_val < MIN_REVERSAL_CONFIDENCE_WITH_BAD_MOMENTUM
        and against_count >= 4
    ):
        errors.append(
            "Setup đảo chiều bị cả MACD và EMA 15M/1H chống lại quá rõ."
        )

    return errors

def _validate_actionable_trade_plan(
    pred: dict,
    timeframe_data: dict[str, pd.DataFrame | None],
    mode: str,
    current_price: float | None,
    output: str | None = None,
) -> list[str]:
    """
    Python guard để tránh lưu/auto-check các kế hoạch dễ bị LOSS do chỉ cần chạm Entry.

    Lý do thêm guard:
    - Auto-check hiện tại chỉ biết giá chạm Entry là ENTRY_FILLED.
    - Với LONG tại liquidity pool dưới hoặc SHORT tại liquidity pool trên, setup đúng phải chờ
      quét thanh khoản + rút râu/đóng nến xác nhận. Nếu lưu như limit order chạm-là-fill thì rất dễ
      bắt dao rơi/bắt đỉnh và SL bị quét ngay.
    - Vì vậy plan nào thiếu khoảng SL tối thiểu hoặc là scalp pending entry quá xa giá hiện tại
      sẽ bị lưu REJECTED_PLAN và trả NO_TRADE an toàn cho user.
    """
    errors: list[str] = []
    direction = (pred.get("direction") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return errors

    if mode == "short":
        errors.extend(_validate_scalp_reversal_quality(pred, timeframe_data, output))

    required = ("entry_low", "entry_high", "sl", "tp1", "tp2")
    missing = [name for name in required if pred.get(name) is None]
    if missing:
        return ["Thiếu số Entry/SL/TP nên không được lưu auto-check: " + ", ".join(missing)]

    entry_low = float(pred["entry_low"])
    entry_high = float(pred["entry_high"])
    if entry_low > entry_high:
        entry_low, entry_high = entry_high, entry_low
    sl = float(pred["sl"])
    tp1 = float(pred["tp1"])
    tp2 = float(pred["tp2"])
    price = float(current_price) if current_price is not None else None

    if pred.get("_structural_sl_error"):
        errors.append(str(pred.get("_structural_sl_error")))

    risk_ref = _risk_floor(timeframe_data, mode, price or ((entry_low + entry_high) / 2.0))
    min_stop = _minimum_stop_distance(timeframe_data, mode, price or ((entry_low + entry_high) / 2.0))
    pred["_risk_reference"] = risk_ref
    pred["_min_stop_distance"] = min_stop
    # Hình học dùng SL cuối cùng xuất cho user/tracking. RR guard mặc định dùng SL cấu trúc
    # gốc trước lớp đệm % thêm, để TEOPARD_EXTRA_SL_BUFFER_PCT không làm kèo bị NO TRADE.
    if direction == "LONG":
        if sl >= entry_low:
            errors.append("LONG sai hình học: SL phải nằm dưới toàn bộ vùng Entry.")
        if tp1 <= entry_high:
            errors.append("LONG sai hình học: TP1 phải nằm trên toàn bộ vùng Entry.")
        if tp2 < tp1:
            errors.append("LONG sai hình học: TP2 không được thấp hơn TP1.")
        final_risk = max(entry_high - sl, 0.0)
    else:
        if sl <= entry_high:
            errors.append("SHORT sai hình học: SL phải nằm trên toàn bộ vùng Entry.")
        if tp1 >= entry_low:
            errors.append("SHORT sai hình học: TP1 phải nằm dưới toàn bộ vùng Entry.")
        if tp2 > tp1:
            errors.append("SHORT sai hình học: TP2 không được cao hơn TP1.")
        final_risk = max(sl - entry_low, 0.0)

    rr_guard = _plan_worst_case_risk_reward(pred, use_rr_guard_sl=True, use_rr_guard_tp=True)
    risk = float(rr_guard.get("risk") or 0.0)
    reward1 = float(rr_guard.get("reward1") or 0.0)
    reward2 = float(rr_guard.get("reward2") or 0.0)

    min_risk = min_stop
    if final_risk <= 0:
        errors.append("Risk Entry-SL không hợp lệ.")
    elif not _guard_is_off() and final_risk < min_risk:
        errors.append(
            f"SL cuối vẫn quá sát Entry: risk bất lợi {fmt(final_risk)} USDT < ngưỡng chống nhiễu {fmt(min_risk)} USDT."
        )

    # RR guard không tính phần nới SL thêm theo %. Muốn ép RR theo SL cuối cùng thì set
    # TEOPARD_RR_USE_EXTRA_SL_BUFFER=1. Mặc định: buffer SL là phong cách quản trị rủi ro
    # của user, không được tự làm Python đổi LONG/SHORT thành NO TRADE.
    if risk > 0 and not _guard_is_off():
        rr1 = reward1 / risk
        rr2 = reward2 / risk
        if rr1 < MIN_TP1_R:
            errors.append(f"TP1 không đủ bù rủi ro sau khi thử target cấu trúc: RR1 khoảng {fmt(rr1, 2)}R < {fmt(MIN_TP1_R, 2)}R.")
        if rr2 < MIN_TP2_R:
            errors.append(f"TP2 không hợp lý sau khi thử target cấu trúc: RR2 khoảng {fmt(rr2, 2)}R < {fmt(MIN_TP2_R, 2)}R.")
        if TP2_MIN_SEPARATION_MULT > 1.0 and reward2 <= reward1 * TP2_MIN_SEPARATION_MULT:
            errors.append("TP2 quá sát TP1 sau khi thử target cấu trúc, không đáng là mục tiêu mở rộng riêng.")

    # V29: Entry xa giá hiện tại nhưng model ghi "vào ngay" không còn bị ép NO TRADE.
    # Output sẽ được _normalize_pending_entry_activation() đổi thành lệnh chờ PENDING_ENTRY.
    # Validator chỉ giữ các lỗi cứng về hình học, SL/TP và confidence.

    return errors


def _guarded_no_trade_output(symbol: str, mode: str, current_price: float | None, errors: list[str]) -> str:
    mode_label = "SCALP" if mode == "short" else "SWING"
    price_text = f" Giá hiện tại {fmt(current_price)} USDT." if current_price is not None else ""
    reason = errors[0] if errors else "Kế hoạch LONG/SHORT bị bộ lọc rủi ro từ chối."
    return sanitize_user_output(
        f"🎯 {symbol} — {mode_label}\n"
        f"🏆 QUYẾT ĐỊNH: NO TRADE — 65%\n"
        f"Lý do: {reason}{price_text} Bot không lưu tín hiệu này để tránh trường hợp vừa chạm vùng vào lệnh đã bị quét SL.\n"
        f"⚠️ Rủi ro: Nếu cố vào lệnh, xác suất bị nhiễu hoặc quét SL ngắn hạn còn cao."
    )


# ─── Hybrid AI validator ─────────────────────────────────────────────────────









def _remove_hidden_liquidity_sections(text: str) -> str:
    """Ẩn các section/vùng thanh khoản khỏi output user; dữ liệu này chỉ dùng nội bộ."""
    if not text:
        return text

    # Xóa block bắt đầu bằng emoji/mục thanh khoản cho tới section kế tiếp.
    text = re.sub(
        r"\n?💧\s*(?:Thanh khoản|Vùng thanh khoản|Heatmap|Vùng thanh lý)[\s\S]*?(?=\n\s*(?:🏆|📈|📉|Entry:|Lý do:|📊|⚠️)|\Z)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Xóa các dòng liệt kê vùng thanh khoản/thanh lý nếu model vẫn lỡ in rải rác.
    text = re.sub(
        r"^\s*(?:Vùng\s+)?(?:thanh khoản|thanh lý|heatmap|vùng quét)[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r"^\s*Vùng\s+thanh\s+khoản\s+(?:dưới|trên)[^\n]*\n?",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Dọn khoảng trắng thừa sau khi xóa block.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

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
    # Dọn lỗi gõ/nhãn tiếng Anh thường bị model chèn vào output user.
    text = re.sub(r"\bNO[_\s-]?TRADE\b", "NO TRADE", text, flags=re.IGNORECASE)
    text = re.sub(r"\bREJECTED[_\s-]?PLAN\b", "kế hoạch bị từ chối", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsweep\b", "quét thanh khoản", text, flags=re.IGNORECASE)
    text = re.sub(r"\breclaim\b", "lấy lại vùng", text, flags=re.IGNORECASE)
    text = re.sub(r"\brisk\s*/\s*reward\b", "tỷ lệ lời/lỗ", text, flags=re.IGNORECASE)
    text = re.sub(r"\brisk\s*-\s*reward\b", "tỷ lệ lời/lỗ", text, flags=re.IGNORECASE)
    text = re.sub(r"\brisk\b", "rủi ro", text, flags=re.IGNORECASE)
    text = re.sub(r"\breward\b", "lợi nhuận kỳ vọng", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNếuu\b", "Nếu", text, flags=re.IGNORECASE)
    text = re.sub(r"\bNếuuu+\b", "Nếu", text, flags=re.IGNORECASE)
    # Replace longer internal labels first so overlapping terms do not leave fragments.
    for old in sorted(replacements, key=len, reverse=True):
        text = text.replace(old, replacements[old])

    # Dọn riêng các nhãn MACD histogram bằng regex để không làm hỏng chữ như "history".
    text = re.sub(r"\bMACD[_\s-]*hist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhist(?:ogram)?\b", "động lượng MACD", text, flags=re.IGNORECASE)

    # Output public tối giản: không show metadata thừa hoặc section cũ.
    text = re.sub(r"^\s*Xu hướng:[^\n]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*Giá:[^\n]*\n?", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*📊\s*Kịch bản chính\s*:", "Lý do:", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*Kịch bản chính\s*:", "Lý do:", text, flags=re.IGNORECASE | re.MULTILINE)
    text = _remove_hidden_liquidity_sections(text)
    return text








def build_no_trade_summary(output: str) -> str:
    text = (output or "").strip().replace("\n", " ")
    if not text:
        return "Claude chọn NO TRADE nhưng không có lý do rõ."
    return "NO TRADE: " + text[:600]


def log_hidden_rejection(symbol: str, mode: str, pred: dict, validation_errors: list[str], output: str) -> None:
    """Log nội bộ để debug trên Railway."""
    try:
        print("[TEOPARD_REJECTED]", flush=True)
        print(f"symbol={symbol} mode={mode} direction={pred.get('direction')}", flush=True)
        print("errors=" + " | ".join(str(e) for e in (validation_errors or [])), flush=True)
        try:
            rr = _plan_worst_case_risk_reward(pred)
            print(
                "metrics="
                + f"risk={fmt(rr.get('risk'))} reward1={fmt(rr.get('reward1'))} reward2={fmt(rr.get('reward2'))} "
                + f"rr1={fmt(rr.get('rr1'),2)} rr2={fmt(rr.get('rr2'),2)} "
                + f"risk_ref={fmt(pred.get('_risk_reference'))} min_stop={fmt(pred.get('_min_stop_distance'))} "
                + f"sl_src={pred.get('_structural_sl_source')} inv={fmt(pred.get('_structural_invalidation_level'))} "
                + f"buf={fmt(pred.get('_structural_sl_buffer'))} extra_sl_pct={fmt((pred.get('_extra_sl_buffer_pct') or 0) * 100)}% before_extra_sl={fmt(pred.get('_sl_before_extra_buffer'))}",
                flush=True,
            )
        except Exception:
            pass
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

    # V32 model-authoritative flow:
    # - Model tự chọn Entry/SL/TP từ dữ liệu Binance + level map Python cung cấp.
    # - Python mặc định KHÔNG tự nhảy SL/TP sang mức khác, để output cuối giữ đúng phân tích model.
    # - Python chỉ validate lỗi cứng và áp buffer SL/TP theo sở thích user nếu user set biến Railway.
    # - Có thể bật lại auto-adjust bằng TEOPARD_PYTHON_ADJUST_SL=1 hoặc TEOPARD_PYTHON_ADJUST_TP=1.
    direction = (pred.get("direction") or "").upper()

    if direction in ("LONG", "SHORT"):
        if _python_adjusts_model_sl():
            pred, output = _normalize_trade_plan_structural_sl(pred, timeframe_data, mode, current_price, output)
        if _python_adjusts_model_tp():
            pred, output = _normalize_trade_plan_structural_tps(pred, timeframe_data, mode, current_price, output)
        pred, output = _apply_extra_sl_buffer_to_plan(pred, output)
        pred, output = _apply_extra_tp_buffers_to_plan(pred, output)
        output = _normalize_pending_entry_activation(output, pred, current_price)

    if direction == "NO_TRADE":
        # V19: chỉ lệnh user xác nhận đã trade mới được lưu vào predictions/history.
        return output

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        guarded_output = _guarded_no_trade_output(binance_symbol, mode, current_price, guard_errors)
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        # V19: không lưu rejected vào predictions nữa để history chỉ gồm lệnh user thật sự trade.
        return guarded_output

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
        save_trade_candidate(
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
        missing = []
        if direction not in ("LONG", "SHORT"):
            missing.append("Không parse được QUYẾT ĐỊNH LONG/SHORT/NO TRADE.")
        for field in ("entry_low", "entry_high", "sl", "tp1", "tp2"):
            if pred.get(field) is None:
                missing.append(f"Không parse được {field}.")
        log_hidden_rejection(binance_symbol, mode, pred, missing, output)

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


async def analyze_symbol(symbol: str, mode: str, user_id: int | None = None, chat_id: int | None = None) -> dict:
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

    # V32 model-authoritative flow:
    # - Model tự chọn Entry/SL/TP từ dữ liệu Binance + level map Python cung cấp.
    # - Python mặc định KHÔNG tự nhảy SL/TP sang mức khác, để output cuối giữ đúng phân tích model.
    # - Python chỉ validate lỗi cứng và áp buffer SL/TP theo sở thích user nếu user set biến Railway.
    # - Có thể bật lại auto-adjust bằng TEOPARD_PYTHON_ADJUST_SL=1 hoặc TEOPARD_PYTHON_ADJUST_TP=1.
    direction = (pred.get("direction") or "").upper()

    if direction in ("LONG", "SHORT"):
        if _python_adjusts_model_sl():
            pred, output = _normalize_trade_plan_structural_sl(pred, timeframe_data, mode, current_price, output)
        if _python_adjusts_model_tp():
            pred, output = _normalize_trade_plan_structural_tps(pred, timeframe_data, mode, current_price, output)
        pred, output = _apply_extra_sl_buffer_to_plan(pred, output)
        pred, output = _apply_extra_tp_buffers_to_plan(pred, output)
        output = _normalize_pending_entry_activation(output, pred, current_price)

    if direction == "NO_TRADE":
        # V19: NO TRADE không lưu vào predictions/history; chỉ lệnh user xác nhận mới được theo dõi.
        return {"text": output, "candidate_id": None}

    guard_errors = _validate_actionable_trade_plan(pred, timeframe_data, mode, current_price, output)
    if guard_errors:
        guarded_output = _guarded_no_trade_output(binance_symbol, mode, current_price, guard_errors)
        log_hidden_rejection(binance_symbol, mode, pred, guard_errors, output)
        # V19: không lưu rejected vào predictions/history nữa.
        return {"text": guarded_output, "candidate_id": None}

    can_track = (
        direction in ("LONG", "SHORT")
        and pred.get("entry_low") is not None
        and pred.get("entry_high") is not None
        and pred.get("sl") is not None
        and pred.get("tp1") is not None
        and pred.get("tp2") is not None
    )

    candidate_id = None
    if can_track:
        reasoning_summary = await asyncio.to_thread(summarize_reasoning, output)
        candidate_id = await asyncio.to_thread(
            save_trade_candidate,
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
        missing = []
        if direction not in ("LONG", "SHORT"):
            missing.append("Không parse được QUYẾT ĐỊNH LONG/SHORT/NO TRADE.")
        for field in ("entry_low", "entry_high", "sl", "tp1", "tp2"):
            if pred.get(field) is None:
                missing.append(f"Không parse được {field}.")
        log_hidden_rejection(binance_symbol, mode, pred, missing, output)

    return {"text": output, "candidate_id": candidate_id}
