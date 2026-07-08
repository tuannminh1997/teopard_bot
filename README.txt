# Teopard Bot — GLM Native / Z.AI build

Bản này chốt dùng GLM native qua Z.AI.

## Flow phân tích

- Bot lấy dữ liệu nến từ Binance.
- Python tính dữ liệu cứng: EMA, RSI, MACD, ATR, Fibonacci, cấu trúc, vùng stop/liquidity ước lượng.
- Model GLM tự phân tích và tự chọn Entry / SL / TP.
- Python mặc định không tự nhảy SL/TP sang số khác.
- Python chỉ validate lỗi hình học/risk tối thiểu, xử lý lệnh chờ, và chỉ lưu history khi user bấm xác nhận đã trade theo bot.

## Timeframe roles V33

SCALP:
- 15M = trigger/timing, sweep/wick, chỉ tham khảo để vào lệnh.
- 1H = khung setup/chính.
- 4H = xác nhận xu hướng.
- 1D = bối cảnh lớn.

SWING:
- 1H = trigger/timing phụ.
- 4H = setup/vùng vào.
- 1D = khung xu hướng/chính.
- 1W = bối cảnh lớn.

## Nến đã đóng vs nến đang chạy

- Indicator, structure, Fibonacci, raw candle chính và market regime dùng nến đã đóng.
- Nến đang chạy được tách riêng trong LIVE_CANDLE_CONTEXT và chỉ dùng tham khảo.
- Model không được dùng nến live để xác nhận Entry/đảo chiều.

## Railway variables cần thiết

```env
BOT_TOKEN=...
ADMIN_USER_IDS=5920124635
DB_PATH=/data/bot.db

AI_PROVIDER=zai
ZAI_API_KEY=...
ZAI_MODEL=glm-5.2
ZAI_BASE_URL=https://api.z.ai/api/paas/v4
ZAI_REASONING_EFFORT=high
ZAI_SUMMARY_REASONING_EFFORT=none
ZAI_APP_NAME=Teopard Bot

LLM_MAX_OUTPUT_TOKENS=8000
LLM_MAX_CONTINUATIONS=2
LLM_SUMMARY_MAX_OUTPUT_TOKENS=600

TEOPARD_PYTHON_ADJUST_SL=0
TEOPARD_PYTHON_ADJUST_TP=0

TEOPARD_EXTRA_SL_BUFFER_PCT=0
TEOPARD_EXTRA_TP1_BUFFER_PCT=0
TEOPARD_EXTRA_TP2_BUFFER_PCT=0
TEOPARD_EXTRA_TP_BUFFER_PCT=0
TEOPARD_RR_USE_EXTRA_SL_BUFFER=0
TEOPARD_RR_USE_EXTRA_TP_BUFFER=0

TEOPARD_GUARD_PROFILE=loose
TEOPARD_MIN_TP1_R=0.40
TEOPARD_MIN_TP2_R=0.50
TEOPARD_MIN_SCALP_CONFIDENCE=48
TEOPARD_MIN_REVERSAL_CONFIDENCE=50
TEOPARD_MIN_REVERSAL_BAD_MOMENTUM_CONFIDENCE=52
TEOPARD_WEAK_CONFIRM_VOLUME=0.45
```

## Có thể xóa khỏi Railway

Nếu đã chốt chỉ dùng Z.AI native thì có thể xóa:

```env
OPENROUTER_API_KEY
OPENROUTER_MODEL
OPENROUTER_BASE_URL
OPENROUTER_REASONING_EFFORT
OPENROUTER_SUMMARY_REASONING_EFFORT
OPENROUTER_SITE_URL
OPENROUTER_APP_NAME

ANTHROPIC_API_KEY
CLAUDE_MODEL
ANTHROPIC_EFFORT
ANTHROPIC_SUMMARY_EFFORT
CLAUDE_MAX_TOKENS

GLM_MODEL
GLM_REASONING_EFFORT
Z_AI_API_KEY
TEOPARD_SL_EXTRA_BUFFER_PCT
TEOPARD_TP1_EXTRA_BUFFER_PCT
TEOPARD_TP2_EXTRA_BUFFER_PCT
```

## Lệnh chính

- `/start`
- `/help`
- `/listsymbols`
- `/history`
- `/stats`
- `/dashboard`
- `/clearhistory CONFIRM`
- `/cleardrafts CONFIRM`

History chỉ lưu lệnh khi user bấm xác nhận đã trade theo bot. `/cleardrafts CONFIRM` chỉ xóa lệnh nháp/candidate và giữ nguyên history.
