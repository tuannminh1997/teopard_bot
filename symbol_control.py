import sqlite3
import os

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
        result = await analyze_symbol(symbol, mode)
    except Exception as exc:
        await query.message.reply_text(f"Phân tích thất bại: {exc}")
        return

    increment_user_usage(user.id)

    for chunk in split_telegram_message(result):
        await query.message.reply_text(chunk)


# ─── Background job: auto check WIN/LOSS ─────────────────────────────────────

async def job_check_predictions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chạy mỗi giờ, tự check prediction PENDING đến hạn và thông báo admin."""
    from datetime import datetime
    print(f"[AUTO_CHECK] Job chạy lúc {datetime.now().isoformat()}", flush=True)
    from analyze import auto_check_pending_predictions
    from auth import ADMIN_USER_IDS

    messages = await auto_check_pending_predictions()

    if not messages or not ADMIN_USER_IDS:
        return

    text = "📋 Kết quả tự động kiểm tra dự đoán:\n\n" + "\n".join(messages)
    for admin_id in ADMIN_USER_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=text)
        except Exception as exc:
            print(f"Không gửi được cho admin {admin_id}: {exc}")


# ─── Register ────────────────────────────────────────────────────────────────

def register_symbol_handlers(app: Application) -> None:
    init_symbol_db()

    app.add_handler(CommandHandler("addsymbol",    add_symbol))
    app.add_handler(CommandHandler("removesymbol", remove_symbol))
    app.add_handler(CommandHandler("listsymbols",  list_symbols))
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
        app.job_queue.run_repeating(job_check_predictions, interval=60, first=30)


def symbol_control_commands() -> list[BotCommand]:
    return [
        BotCommand("listsymbols", "Xem danh sách symbol"),
    ]
