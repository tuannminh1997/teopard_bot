import asyncio
import os
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-4-sonnet-20250514")

SHORT_TERM_TIMEFRAMES = {
    "4H": ("4h", 500),
    "12H": ("12h", 500),
    "24H": ("1d", 500),
}

LONG_TERM_TIMEFRAMES = {
    "24H": ("1d", 500),
    "48H": ("48h_resampled", 500),
    "1W": ("1w", 300),
}


def load_analysis_system_prompt() -> str:
    prompt_paths = [
        Path("analyze_system_prompt.txt"),
        Path("analysis_system_prompt.txt"),
    ]

    for prompt_path in prompt_paths:
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()

    raise FileNotFoundError(
        "Không tìm thấy file analyze_system_prompt.txt hoặc analysis_system_prompt.txt."
    )


def get_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame | None:
    try:
        response = requests.get(
            BINANCE_API_URL,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        df = pd.DataFrame(
            data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        df[numeric_cols] = df[numeric_cols].astype(float)

        return df
    except Exception as exc:
        print(f"Không lấy được dữ liệu Binance cho {symbol} {interval}: {exc}")
        return None


def resample_to_48h(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None

    return (
        df.set_index("timestamp")
        .resample("48h")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "quote_volume": "sum",
                "taker_buy_volume": "sum",
                "taker_buy_quote_volume": "sum",
                "count": "sum",
            }
        )
        .dropna()
        .reset_index()
    )


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = pd.Series(np.nan, index=data.index, dtype="float64")
    avg_loss = pd.Series(np.nan, index=data.index, dtype="float64")

    if len(data) <= period:
        return pd.Series(np.nan, index=data.index, dtype="float64")

    avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()

    for i in range(period + 1, len(data)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (period - 1)) + gain.iloc[i]) / period
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)

    return rsi


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(
        span=signal_period,
        adjust=False,
        min_periods=signal_period,
    ).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_technical_indicators(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) < 80:
        return None

    result = df.copy()

    result["ema_7"] = calculate_ema(result["close"], 7)
    result["ema_25"] = calculate_ema(result["close"], 25)
    result["ema_50"] = calculate_ema(result["close"], 50)

    result["rsi_6"] = calculate_rsi(result["close"], 6)
    result["rsi_12"] = calculate_rsi(result["close"], 12)
    result["rsi_24"] = calculate_rsi(result["close"], 24)

    macd_line, signal_line, histogram = calculate_macd(result["close"])
    result["macd_line"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram

    result["volume_ma_20"] = result["volume"].rolling(window=20).mean()
    result["volume_ratio"] = result["volume"] / result["volume_ma_20"]

    return result.dropna().reset_index(drop=True)


def get_fear_greed_index() -> str:
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("metadata", {}).get("error"):
            raise ValueError(payload["metadata"]["error"])

        value = int(payload["data"][0]["value"])
        return f"Chỉ số sợ hãi và tham lam: {value}/100"
    except Exception as exc:
        print(f"Không lấy được chỉ số sợ hãi và tham lam: {exc}")
        return "Chỉ số sợ hãi và tham lam: không có dữ liệu"


def get_current_price(symbol: str) -> str:
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=30,
        )
        response.raise_for_status()
        price = float(response.json()["price"])
        return f"Giá hiện tại: {format_number(price)} USDT"
    except Exception as exc:
        print(f"Không lấy được giá hiện tại cho {symbol}: {exc}")
        return "Giá hiện tại: không có dữ liệu"


def get_timeframe_data(symbol: str, mode: str) -> dict[str, pd.DataFrame | None]:
    configs = SHORT_TERM_TIMEFRAMES if mode == "short" else LONG_TERM_TIMEFRAMES
    result: dict[str, pd.DataFrame | None] = {}

    for label, (interval, limit) in configs.items():
        if interval == "48h_resampled":
            base_df = get_binance_klines(symbol, "12h", 1000)
            raw_df = resample_to_48h(base_df)
        else:
            raw_df = get_binance_klines(symbol, interval, limit)

        result[label] = calculate_technical_indicators(raw_df)

    return result


def format_number(value: float) -> str:
    if pd.isna(value):
        return "không có"

    if abs(value) >= 100:
        return f"{value:,.2f}"

    if abs(value) >= 1:
        return f"{value:,.4f}"

    return f"{value:,.8f}"


def format_timeframe_rows(df: pd.DataFrame, row_count: int = 30) -> str:
    rows = []

    for _, row in df.tail(row_count).iterrows():
        rows.append(
            " | ".join(
                [
                    str(row["timestamp"]),
                    format_number(row["open"]),
                    format_number(row["high"]),
                    format_number(row["low"]),
                    format_number(row["close"]),
                    format_number(row["ema_7"]),
                    format_number(row["ema_25"]),
                    format_number(row["ema_50"]),
                    format_number(row["rsi_6"]),
                    format_number(row["rsi_12"]),
                    format_number(row["rsi_24"]),
                    format_number(row["macd_line"]),
                    format_number(row["macd_signal"]),
                    format_number(row["macd_histogram"]),
                    format_number(row["volume"]),
                    format_number(row["volume_ma_20"]),
                    format_number(row["volume_ratio"]),
                ]
            )
        )

    return "\n".join(rows)


def format_analysis_prompt(
    symbol: str,
    mode: str,
    timeframe_data: dict[str, pd.DataFrame | None],
    fear_greed_info: str,
    current_price_info: str,
) -> str:
    mode_label = "ngắn hạn" if mode == "short" else "dài hạn"
    timeframe_label = "4H, 12H, 24H" if mode == "short" else "24H, 48H, 1W"
    focus_text = (
        "Ưu tiên 4H và 12H để ra quyết định, dùng 24H để xác nhận xu hướng lớn hơn."
        if mode == "short"
        else "Ưu tiên 24H và 48H để ra quyết định, dùng 1W để xác nhận xu hướng lớn hơn."
    )

    prompt = f"""
YÊU CẦU PHÂN TÍCH {mode_label.upper()} CHO {symbol}

{current_price_info}
Khung thời gian được cung cấp: {timeframe_label}
Cách đọc dữ liệu: {focus_text}
{fear_greed_info}

Chỉ trả về 2 phần sau:
1. Chiến lược giao dịch
2. Kịch bản & xác suất

Không trả về các phần khác như xu hướng tổng quan, hỗ trợ kháng cự riêng lẻ, phân tích chỉ báo riêng lẻ hoặc kết luận dài.
Hãy dùng số liệu EMA, RSI, MACD, khối lượng trong dữ liệu bên dưới để giải thích ngắn gọn cho chiến lược và kịch bản.

Định dạng mỗi dòng dữ liệu:
Thời gian | Mở cửa | Cao nhất | Thấp nhất | Đóng cửa | EMA7 | EMA25 | EMA50 | RSI6 | RSI12 | RSI24 | MACD | Tín hiệu MACD | Cột MACD | Khối lượng | Khối lượng trung bình 20 kỳ | Tỷ lệ khối lượng
"""

    for timeframe, df in timeframe_data.items():
        prompt += f"\n\nKHUNG {timeframe}\n"

        if df is None or df.empty:
            prompt += "Không đủ dữ liệu.\n"
            continue

        prompt += format_timeframe_rows(df)

    return prompt

def call_claude_analysis(symbol: str, mode: str) -> str:
    if not ANTHROPIC_API_KEY:
        return "Thiếu ANTHROPIC_API_KEY trong file .env."

    binance_symbol = f"{symbol.upper()}USDT"
    timeframe_data = get_timeframe_data(binance_symbol, mode)

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        return f"Không lấy được dữ liệu Binance cho {binance_symbol}."

    system_prompt = load_analysis_system_prompt()
    fear_greed_info = get_fear_greed_index()
    current_price_info = get_current_price(binance_symbol)
    user_prompt = format_analysis_prompt(
        symbol=binance_symbol,
        mode=mode,
        timeframe_data=timeframe_data,
        fear_greed_info=fear_greed_info,
        current_price_info=current_price_info,
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        timeout=300,
    )

    return "".join(
        block.text
        for block in response.content
        if hasattr(block, "text")
    )


async def analyze_symbol(symbol: str, mode: str) -> str:
    return await asyncio.to_thread(call_claude_analysis, symbol, mode)