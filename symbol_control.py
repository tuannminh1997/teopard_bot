import sqlite3
import os
import asyncio

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

DB_PATH = os.getenv("DB_PATH", "bot.db")
ANALYZE_SHORT_CALLBACK_PREFIX = "analyze_short"
ANALYZE_LONG_CALLBACK_PREFIX  = "analyze_long"
CONFIRM_TRADE_CALLBACK_PREFIX = "confirm_trade"
DISCARD_TRADE_CALLBACK_PREFIX = "discard_trade"


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().lstrip("/").upper()


def init_symbol_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allowed_symbols (
                symbol TEXT PRIMARY KEY
            )
        """)
        conn.commit()


def add_allowed_symbol(symbol: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO allowed_symbols (symbol) VALUES (?)",
            (normalize_symbol(symbol),),
        )
        conn.commit()


def remove_allowed_symbol(symbol: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM allowed_symbols WHERE symbol = ?",
            (normalize_symbol(symbol),),
        )
        conn.commit()


def is_allowed_symbol(symbol: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM allowed_symbols WHERE symbol = ?",
            (normalize_symbol(symbol),),
        ).fetchone()
    return row is not None


def get_allowed_symbols() -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT symbol FROM allowed_symbols ORDER BY symbol"
        ).fetchall()
    return [r[0] for r in rows]


def symbol_analysis_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("Scalp (15m/1H/4H)", callback_data=f"{ANALYZE_SHORT_CALLBACK_PREFIX}:{symbol}"),
        InlineKeyboardButton("Swing (4H/1D/1W)",  callback_data=f"{ANALYZE_LONG_CALLBACK_PREFIX}:{symbol}"),
    ]])




def is_actionable_trade_response(text: str | None) -> bool:
    """Return True only for user-visible LONG/SHORT trade signals.

    V22 guard: NO TRADE must never show the confirm-trade keyboard, even if
    a stale/buggy candidate_id is accidentally returned by analyze_symbol().
    The DB still uses NO_TRADE internally, but user output may be NO TRADE.
    """
    import re

    if not text:
        return False

    # Any explicit NO TRADE decision wins over later wording.
    if re.search(r"QUYẾT\s+ĐỊNH[:\s]+(?:NO[_\s-]?TRADE|KHÔNG\s+VÀO\s+LỆNH|KHONG\s+VAO\s+LENH)", text, re.IGNORECASE):
        return False

    # Only explicit LONG/SHORT decisions are actionable.
    return bool(re.search(r"QUYẾT\s+ĐỊNH[:\s]+(?:LONG|SHORT)\b", text, re.IGNORECASE))


def trade_candidate_keyboard(candidate_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Tôi đã trade theo lệnh này", callback_data=f"{CONFIRM_TRADE_CALLBACK_PREFIX}:{candidate_id}")],
        [InlineKeyboardButton("Bỏ qua, không lưu history", callback_data=f"{DISCARD_TRADE_CALLBACK_PREFIX}:{candidate_id}")],
    ])


def split_telegram_message(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= limit:
            current += line
            continue
        if current:
            chunks.append(current.strip())
            current = ""
        while len(line) > limit:
            chunks.append(line[:limit].strip())
            line = line[limit:]
        current = line
    if current:
        chunks.append(current.strip())
    return chunks


# ─── Command handlers ─────────────────────────────────────────────────────────

async def add_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp đúng: /addsymbol BTC")
        return
    symbol = normalize_symbol(context.args[0])
    add_allowed_symbol(symbol)
    await update.effective_message.reply_text(f"Đã thêm symbol {symbol}.")


async def remove_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp đúng: /removesymbol BTC")
        return
    symbol = normalize_symbol(context.args[0])
    remove_allowed_symbol(symbol)
    await update.effective_message.reply_text(f"Đã xóa symbol {symbol}.")


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbols = get_allowed_symbols()
    if not symbols:
        await update.effective_message.reply_text("Danh sách symbol hiện đang trống.")
        return
    await update.effective_message.reply_text(
        "Danh sách symbol được phép:\n" + "\n".join(f"• {s}" for s in symbols)
    )


# ─── Message handler: user nhập tên coin ─────────────────────────────────────

async def handle_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    from auth import is_account_activated, show_start_menu, verified_users

    user = update.effective_user
    message = update.effective_message
    if not user or not message or not message.text:
        return False

    symbol = normalize_symbol(message.text)
    if not is_allowed_symbol(symbol):
        return False

    if not is_account_activated(user.id):
        verified_users.discard(user.id)
        await show_start_menu(update)
        return True

    await message.reply_text(
        f"Bạn muốn phân tích {symbol}/USDT theo kiểu nào?",
        reply_markup=symbol_analysis_keyboard(symbol),
    )
    return True


async def symbol_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    handled = await handle_symbol(update, context)
    if handled:
        raise ApplicationHandlerStop


# ─── Callback: user chọn Scalp/Swing ────────────────────────────────────────

async def analyze_symbol_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import analyze_symbol
    from auth import get_user_usage, increment_user_usage, is_account_activated, show_start_menu, verified_users

    query = update.callback_query
    user  = update.effective_user
    if not query or not user or not query.data:
        return

    await query.answer()

    if not is_account_activated(user.id):
        verified_users.discard(user.id)
        await show_start_menu(update)
        return

    action, symbol = query.data.split(":", 1)
    mode = "short" if action == ANALYZE_SHORT_CALLBACK_PREFIX else "long"
    mode_label = "Scalp (15m/1H/4H)" if mode == "short" else "Swing (4H/1D/1W)"

    daily_limit, used_today = get_user_usage(user.id)
    remaining = daily_limit - used_today

    if remaining <= 0:
        await query.message.reply_text(
            f"Bạn đã hết {daily_limit} lượt hôm nay. "
            "Vui lòng chờ sang ngày mới hoặc liên hệ admin."
        )
        return

    await query.message.reply_text(
        f"Đang phân tích {symbol}/USDT — {mode_label}. "
        f"Vui lòng chờ... (còn {remaining - 1} lượt hôm nay)"
    )

    try:
        result_payload = await analyze_symbol(symbol, mode, user_id=user.id, chat_id=query.message.chat_id)
    except Exception as exc:
        await query.message.reply_text(f"Phân tích thất bại: {exc}")
        return

    increment_user_usage(user.id)

    if isinstance(result_payload, dict):
        result_text = result_payload.get("text", "")
        candidate_id = result_payload.get("candidate_id")
    else:
        result_text = str(result_payload)
        candidate_id = None

    # V22: Chỉ LONG/SHORT hợp lệ mới được hiện nút xác nhận trade.
    # NO TRADE không bao giờ có nút, kể cả khi analyze_symbol() lỡ trả candidate_id do lỗi/stale state.
    show_trade_confirm_button = bool(candidate_id) and is_actionable_trade_response(result_text)

    chunks = split_telegram_message(result_text)
    for idx, chunk in enumerate(chunks):
        is_last = idx == len(chunks) - 1
        if is_last and show_trade_confirm_button:
            await query.message.reply_text(
                chunk + "\n\nNếu bạn thật sự vào lệnh theo phân tích này, bấm nút bên dưới để bot lưu vào history và theo dõi WIN/LOSS.",
                reply_markup=trade_candidate_keyboard(int(candidate_id)),
            )
        else:
            await query.message.reply_text(chunk)



async def confirm_trade_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import confirm_trade_candidate

    query = update.callback_query
    user = update.effective_user
    if not query or not user or not query.data:
        return
    await query.answer()
    try:
        _, raw_id = query.data.split(":", 1)
        candidate_id = int(raw_id)
    except Exception:
        await query.message.reply_text("Không đọc được mã lệnh nháp.")
        return

    result = await asyncio.to_thread(confirm_trade_candidate, candidate_id, user.id)
    await query.message.reply_text(result.get("message", "Đã xử lý."))
    # Dù là lần bấm đầu, bấm lặp, hay đang xử lý, nút này không nên còn clickable trên UI.
    if result.get("ok") or result.get("already_confirmed"):
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass


async def discard_trade_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import discard_trade_candidate

    query = update.callback_query
    user = update.effective_user
    if not query or not user or not query.data:
        return
    await query.answer()
    try:
        _, raw_id = query.data.split(":", 1)
        candidate_id = int(raw_id)
    except Exception:
        await query.message.reply_text("Không đọc được mã lệnh nháp.")
        return

    result = await asyncio.to_thread(discard_trade_candidate, candidate_id, user.id)
    await query.message.reply_text(result.get("message", "Đã xử lý."))
    if result.get("ok"):
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass


async def confirmtrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import confirm_trade_candidate

    user = update.effective_user
    if not user:
        return
    if not context.args:
        await update.effective_message.reply_text("Cú pháp: /confirmtrade <mã_lệnh_nháp>. Thường bạn chỉ cần bấm nút dưới phân tích.")
        return
    try:
        candidate_id = int(context.args[0])
    except ValueError:
        await update.effective_message.reply_text("Mã lệnh nháp phải là số.")
        return
    result = await asyncio.to_thread(confirm_trade_candidate, candidate_id, user.id)
    await update.effective_message.reply_text(result.get("message", "Đã xử lý."))


# ─── Background job: auto check WIN/LOSS ─────────────────────────────────────

async def job_check_predictions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chạy định kỳ, tự check prediction đến hạn và chỉ cập nhật DB, không gửi tin nhắn tự động."""
    from datetime import datetime
    from analyze import auto_check_pending_predictions

    print(f"[AUTO_CHECK] Job chạy lúc {datetime.now().isoformat()}", flush=True)
    payload = await auto_check_pending_predictions()

    if isinstance(payload, dict):
        print(
            "[AUTO_CHECK] Done: "
            f"due={payload.get('due_count', 0)}, "
            f"entry_filled={payload.get('entry_filled_count', 0)}, "
            f"closed={payload.get('closed_count', 0)}, "
            f"rescheduled={payload.get('rescheduled_count', 0)}",
            flush=True,
        )

    # Không gửi thông báo tự động cho user/admin.
    # User muốn xem kết quả thì dùng /history, /stats hoặc /dashboard.


def command_scope_user_id(update: Update) -> int | None:
    user = update.effective_user
    if not user:
        return None
    # Mặc định mọi người, kể cả admin, xem dữ liệu của chính mình.
    # Admin muốn xem toàn hệ thống dùng các lệnh riêng: /statsall, /historyall, /dashboardall.
    return user.id


def is_current_user_admin(update: Update) -> bool:
    from auth import is_admin

    user = update.effective_user
    return bool(user and is_admin(user.id))


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_stats(symbol, user_id=command_scope_user_id(update)))


async def statsall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_stats(symbol, user_id=None))


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_history
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_history(symbol, user_id=command_scope_user_id(update)))


async def historyall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_history

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    symbol = context.args[0] if context.args else None
    await update.effective_message.reply_text(format_history(symbol, user_id=None))


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats
    await update.effective_message.reply_text(format_stats(user_id=command_scope_user_id(update)))


async def dashboardall_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from analyze import format_stats

    if not is_current_user_admin(update):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return
    await update.effective_message.reply_text(format_stats(user_id=None))


async def clearhistory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    from analyze import clear_prediction_history

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    if not context.args or context.args[0].upper() != "CONFIRM":
        await update.effective_message.reply_text(
            "Lệnh này sẽ xóa toàn bộ lịch sử phân tích/prediction nhưng vẫn giữ whitelist và danh sách symbol.\n"
            "Gõ: /clearhistory CONFIRM"
        )
        return

    payload = clear_prediction_history()
    if isinstance(payload, dict):
        await update.effective_message.reply_text(
            "Đã xóa lịch sử theo dõi. Whitelist và danh sách symbol vẫn được giữ.\n"
            f"Lệnh đã trade/đang theo dõi: {payload.get('visible_count', 0)}\n"
            f"Tổng dòng predictions cũ đã xóa: {payload.get('total_prediction_count', 0)}\n"
            f"Lệnh nháp chưa xác nhận đã xóa: {payload.get('draft_count', 0)}"
        )
    else:
        await update.effective_message.reply_text(
            f"Đã xóa {payload} prediction khỏi lịch sử. Whitelist và danh sách symbol vẫn được giữ."
        )


async def cleardrafts_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xóa riêng lệnh nháp/candidate, giữ nguyên history đã trade theo bot."""
    from auth import is_admin
    from analyze import clear_trade_candidates

    user = update.effective_user
    if not user:
        return

    args_upper = [arg.upper() for arg in (context.args or [])]
    clear_all = "ALL" in args_upper

    if clear_all and not is_admin(user.id):
        await update.effective_message.reply_text("Bạn không có quyền xóa lệnh nháp của toàn hệ thống.")
        return

    if "CONFIRM" not in args_upper:
        if clear_all:
            await update.effective_message.reply_text(
                "Lệnh này chỉ xóa lệnh nháp/candidate của toàn hệ thống, KHÔNG xóa /history.\n"
                "Gõ: /cleardrafts ALL CONFIRM"
            )
        else:
            await update.effective_message.reply_text(
                "Lệnh này chỉ xóa lệnh nháp/candidate của bạn, KHÔNG xóa /history.\n"
                "Gõ: /cleardrafts CONFIRM\n"
                "Admin muốn xóa toàn hệ thống: /cleardrafts ALL CONFIRM"
            )
        return

    payload = await asyncio.to_thread(clear_trade_candidates, None if clear_all else user.id)
    scope_text = "toàn hệ thống" if clear_all else "của bạn"
    reset_text = "Có" if payload.get("sequence_reset") else "Không, vì vẫn còn candidate của user khác"
    await update.effective_message.reply_text(
        f"Đã xóa lệnh nháp {scope_text}. History đã trade vẫn được giữ nguyên.\n"
        f"Tổng candidate đã xóa: {payload.get('deleted_count', 0)}\n"
        f"- Nháp còn chờ: {payload.get('draft_count', 0)}\n"
        f"- Đã hết hạn: {payload.get('expired_count', 0)}\n"
        f"- Đã bỏ qua: {payload.get('discarded_count', 0)}\n"
        f"- Đang xác nhận: {payload.get('confirming_count', 0)}\n"
        f"- Đã xác nhận/copy sang history: {payload.get('confirmed_count', 0)}\n"
        f"Reset ID lệnh nháp: {reset_text}"
    )


async def checknow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    from auth import is_admin
    from analyze import auto_check_pending_predictions

    admin = update.effective_user
    if not admin or not is_admin(admin.id):
        await update.effective_message.reply_text("Bạn không có quyền dùng lệnh này.")
        return

    await update.effective_message.reply_text("Đang ép kiểm tra toàn bộ prediction đang mở ngay bây giờ...")
    payload = await auto_check_pending_predictions(force=True)

    if not isinstance(payload, dict):
        await update.effective_message.reply_text("Đã kiểm tra xong.")
        return

    closed_count = int(payload.get("closed_count", 0))
    entry_filled_count = int(payload.get("entry_filled_count", 0))
    rescheduled_count = int(payload.get("rescheduled_count", 0))
    due_count = int(payload.get("due_count", 0))

    if due_count == 0:
        await update.effective_message.reply_text("Không có prediction đang mở để kiểm tra.")
        return

    await update.effective_message.reply_text(
        "Đã kiểm tra xong và cập nhật DB.\n"
        f"Prediction đang mở đã kiểm tra: {due_count}\n"
        f"Mới khớp Entry: {entry_filled_count}\n"
        f"Có kết quả cuối: {closed_count}\n"
        f"Tiếp tục chờ: {rescheduled_count}\n\n"
        "Bot không gửi thông báo tự động cho user/admin nữa. "
        "Cần xem chi tiết thì dùng /history, /stats, /dashboard hoặc /historyall."
    )


# ─── Register ────────────────────────────────────────────────────────────────

def register_symbol_handlers(app: Application) -> None:
    init_symbol_db()

    app.add_handler(CommandHandler("addsymbol",    add_symbol))
    app.add_handler(CommandHandler("removesymbol", remove_symbol))
    app.add_handler(CommandHandler("listsymbols",  list_symbols))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("statsall", statsall_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("historyall", historyall_command))
    app.add_handler(CommandHandler("dashboard", dashboard_command))
    app.add_handler(CommandHandler("dashboardall", dashboardall_command))
    app.add_handler(CommandHandler("clearhistory", clearhistory_command))
    app.add_handler(CommandHandler("cleardrafts", cleardrafts_command))
    app.add_handler(CommandHandler("checknow", checknow_command))
    app.add_handler(CommandHandler("confirmtrade", confirmtrade_command))
    app.add_handler(CallbackQueryHandler(
        confirm_trade_callback,
        pattern=f"^{CONFIRM_TRADE_CALLBACK_PREFIX}:",
    ))
    app.add_handler(CallbackQueryHandler(
        discard_trade_callback,
        pattern=f"^{DISCARD_TRADE_CALLBACK_PREFIX}:",
    ))
    app.add_handler(CallbackQueryHandler(
        analyze_symbol_callback,
        pattern=f"^({ANALYZE_SHORT_CALLBACK_PREFIX}|{ANALYZE_LONG_CALLBACK_PREFIX}):",
    ))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, symbol_message_handler), group=1)
    app.add_handler(MessageHandler(filters.COMMAND, symbol_message_handler), group=2)

    # Background job: check pending predictions mỗi 60 phút
    if app.job_queue is None:
        print("JobQueue is not available. Install python-telegram-bot[job-queue].")
    else:
        app.job_queue.run_repeating(job_check_predictions, interval=3600, first=300)


def symbol_control_commands() -> list[BotCommand]:
    return [
        BotCommand("listsymbols", "Xem danh sách coin hỗ trợ"),
        BotCommand("stats", "Xem thống kê win/loss, có thể gõ /stats BTC"),
        BotCommand("history", "Xem lịch sử, có thể gõ /history BTC"),
        BotCommand("dashboard", "Xem dashboard nhanh"),
        BotCommand("confirmtrade", "Lưu lệnh nháp để theo dõi"),
        BotCommand("cleardrafts", "Xóa lệnh nháp, giữ history"),
    ]
