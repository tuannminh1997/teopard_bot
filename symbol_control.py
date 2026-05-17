import sqlite3

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

DB_PATH = "bot.db"
ANALYZE_SHORT_CALLBACK_PREFIX = "analyze_short"
ANALYZE_LONG_CALLBACK_PREFIX = "analyze_long"


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
    normalized_symbol = normalize_symbol(symbol)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO allowed_symbols (symbol) VALUES (?)",
            (normalized_symbol,),
        )
        conn.commit()


def remove_allowed_symbol(symbol: str) -> None:
    normalized_symbol = normalize_symbol(symbol)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM allowed_symbols WHERE symbol = ?",
            (normalized_symbol,),
        )
        conn.commit()


def is_allowed_symbol(symbol: str) -> bool:
    normalized_symbol = normalize_symbol(symbol)

    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM allowed_symbols WHERE symbol = ?",
            (normalized_symbol,),
        ).fetchone()

    return row is not None


def get_allowed_symbols() -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT symbol FROM allowed_symbols ORDER BY symbol"
        ).fetchall()

    return [row[0] for row in rows]


def symbol_analysis_keyboard(symbol: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "Ngắn hạn",
                callback_data=f"{ANALYZE_SHORT_CALLBACK_PREFIX}:{symbol}",
            ),
            InlineKeyboardButton(
                "Dài hạn",
                callback_data=f"{ANALYZE_LONG_CALLBACK_PREFIX}:{symbol}",
            ),
        ],
    ])


def split_telegram_message(text: str, limit: int = 3900) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    current = ""

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

    await update.effective_message.reply_text(
        f"Đã thêm symbol {symbol} vào danh sách được phép."
    )


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

    await update.effective_message.reply_text(
        f"Đã xóa symbol {symbol} khỏi danh sách được phép."
    )


async def list_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbols = get_allowed_symbols()

    if not symbols:
        await update.effective_message.reply_text("Danh sách symbol hiện đang trống.")
        return

    text = "Danh sách symbol được phép:\n" + "\n".join(symbols)
    await update.effective_message.reply_text(text)


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
        f"Bạn muốn phân tích {symbol}/USDT theo khung nào?",
        reply_markup=symbol_analysis_keyboard(symbol),
    )
    return True


async def symbol_message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    handled = await handle_symbol(update, context)

    if handled:
        raise ApplicationHandlerStop


async def analyze_symbol_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    from analyze import analyze_symbol
    from auth import get_user_usage, increment_user_usage, is_account_activated, show_start_menu, verified_users

    query = update.callback_query
    user = update.effective_user

    if not query or not user or not query.data:
        return

    await query.answer()

    if not is_account_activated(user.id):
        verified_users.discard(user.id)
        await show_start_menu(update)
        return

    action, symbol = query.data.split(":", 1)
    mode = "short" if action == ANALYZE_SHORT_CALLBACK_PREFIX else "long"
    mode_label = "ngắn hạn" if mode == "short" else "dài hạn"

    daily_limit, used_today = get_user_usage(user.id)
    remaining = daily_limit - used_today

    if remaining <= 0:
        await query.message.reply_text(
            f"Bạn đã hết lượt sử dụng hôm nay ({daily_limit}/{daily_limit} lượt). "
            "Vui lòng chờ sang ngày mới hoặc liên hệ admin để được cấp thêm lượt."
        )
        return

    increment_user_usage(user.id)

    await query.message.reply_text(
        f"Đang phân tích {symbol}/USDT theo khung {mode_label}. "
        f"Vui lòng chờ... (còn {remaining - 1} lượt hôm nay)"
    )

    try:
        result = await analyze_symbol(symbol, mode)
    except Exception as exc:
        await query.message.reply_text(f"Phân tích thất bại: {exc}")
        return

    for chunk in split_telegram_message(result):
        await query.message.reply_text(chunk)


def register_symbol_handlers(app: Application) -> None:
    init_symbol_db()

    app.add_handler(CommandHandler("addsymbol", add_symbol))
    app.add_handler(CommandHandler("removesymbol", remove_symbol))
    app.add_handler(CommandHandler("listsymbols", list_symbols))
    app.add_handler(CallbackQueryHandler(
        analyze_symbol_callback,
        pattern=f"^({ANALYZE_SHORT_CALLBACK_PREFIX}|{ANALYZE_LONG_CALLBACK_PREFIX}):",
    ))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, symbol_message_handler), group=1)
    app.add_handler(MessageHandler(filters.COMMAND, symbol_message_handler), group=2)


def symbol_control_commands() -> list[BotCommand]:
    return [
        BotCommand("listsymbols", "Xem danh sách symbol"),
    ]



