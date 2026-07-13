# Teopard Bot — GLM Native / Z.AI build

Bản này chốt dùng GLM native qua Z.AI.

## Flow phân tích

- Bot lấy dữ liệu nến từ Binance.
- Python tính dữ liệu cứng: EMA, RSI, MACD, ATR, Fibonacci, cấu trúc, vùng stop/liquidity ước lượng.
- Model GLM tự phân tích và tự chọn Entry / SL / TP.
- Python mặc định không tự nhảy SL/TP sang số khác.
- Python validate hình học, nguồn level, RR theo mode và khoảng chống nhiễu ATR; xử lý lệnh chờ, và chỉ lưu history khi user bấm xác nhận đã trade theo bot.

## Cơ chế Entry / SL / TP bản structural-final

- Python gộp Fibonacci, EMA, đỉnh/đáy và vùng quét gần nhau thành tối đa 6 level mỗi phía.
- Model phải khai báo nguồn nội bộ cho Entry, SL, TP1, TP2 bằng ID level; block này được xóa trước khi gửi Telegram.
- Python không tự bẻ số mặc định, nhưng từ chối kế hoạch nếu SL chưa nằm ngoài invalidation, TP không bám level, RR theo mode quá thấp, TP2 quá sát TP1 hoặc TP quá gần so với ATR của khung vào lệnh.
- Các biến cũ `TEOPARD_MIN_TP1_R`, `TEOPARD_MIN_TP2_R`, `TEOPARD_TP2_SEPARATION_MULT` không còn được dùng; dùng biến SCALP/SWING riêng trong phần Railway variables.

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
ZAI_RETRY_REASONING_EFFORT=medium
ZAI_SUMMARY_REASONING_EFFORT=none
ZAI_APP_NAME=Teopard Bot

LLM_MAX_OUTPUT_TOKENS=2600
LLM_MAIN_OUTPUT_TOKEN_CAP=2600
LLM_MAX_CONTINUATIONS=0
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
TEOPARD_MIN_TP1_R_SCALP=0.70
TEOPARD_MIN_TP2_R_SCALP=1.20
TEOPARD_TP2_SEPARATION_MULT_SCALP=1.35
TEOPARD_MIN_TP1_ATR_MULT_SCALP=0.50
TEOPARD_MIN_TP2_ATR_MULT_SCALP=1.00
TEOPARD_MIN_TP1_R_SWING=0.80
TEOPARD_MIN_TP2_R_SWING=1.50
TEOPARD_TP2_SEPARATION_MULT_SWING=1.40
TEOPARD_MIN_TP1_ATR_MULT_SWING=0.50
TEOPARD_MIN_TP2_ATR_MULT_SWING=1.00
TEOPARD_MIN_SCALP_CONFIDENCE=62
TEOPARD_MIN_SWING_CONFIDENCE=62
TEOPARD_MIN_SETUP_STRENGTH=62
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

## Hotfix timeout provider AI
Nếu gặp lỗi `Read timed out` từ Z.AI/OpenRouter, đây là provider không trả response trong thời gian đọc. Bản này chỉ retry một lần; lần đầu dùng reasoning đã cấu hình, retry tự hạ xuống `ZAI_RETRY_REASONING_EFFORT` để tránh treo lặp lại. Main analysis không continuation và được cap output token. Bot cũng không gọi LLM lần hai chỉ để tóm tắt history.
Có thể chỉnh Railway variables:

```env
LLM_MAIN_TIMEOUT_SECONDS=240
LLM_RETRY_TIMEOUT_SECONDS=150
LLM_SUMMARY_TIMEOUT_SECONDS=60
LLM_API_RETRIES=1
LLM_MAIN_RETRY_LIMIT=1
LLM_RETRY_SLEEP_SECONDS=2
```

## Auto Scan Mode

Auto Scan là mode riêng, không thay thế phân tích thủ công.

Flow:

- Bot quét theo nến đã đóng, mặc định mỗi 15 phút.
- DeepSeek v4 flash lọc nhanh bằng text prompt rút gọn để xem có đáng gọi GLM không.
- Nếu lọc nhanh đạt ngưỡng, bot gửi data text đầy đủ sang GLM/Z.AI giống mode thủ công.
- Nếu GLM/Z.AI trả LONG/SHORT đủ confidence, bot gửi tín hiệu cho user.
- Auto Scan không hiện nút "Tôi đã đặt lệnh theo phân tích này".
- Tín hiệu Auto Scan được lưu thẳng vào predictions để auto-check.

Lệnh user:

- `/autoscanon BTC` - bật Auto Scan cho BTC.
- `/autoscanoff` - tắt Auto Scan.
- `/autoscanstatus` - xem trạng thái, lần quét gần nhất và lần quét kế tiếp.
- `/autoscanlog` - xem log các lần quét gần nhất.

Railway variables Auto Scan:

```env
DEEPSEEK_API_KEY="..."
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-v4-flash"
DEEPSEEK_TIMEOUT_SECONDS="60"
DEEPSEEK_MAX_OUTPUT_TOKENS="700"
DEEPSEEK_TEMPERATURE="0.05"

AUTO_SCAN_INTERVAL_SECONDS="900"
AUTO_SCAN_MODES="short"
AUTO_SCAN_MIN_PREFILTER_CONFIDENCE="62"
AUTO_SCAN_MIN_FINAL_CONFIDENCE="62"
AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES="180"
AUTO_SCAN_SEND_NO_TRADE="0"
AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS="5"
AUTO_SCAN_LOG_LIMIT="20"
AUTO_SCAN_DEBUG="0"
AUTO_SCAN_SCHEDULER_TICK_SECONDS="60"
```

Không cần set `AUTO_SCAN_SYMBOLS`. User chọn symbol bằng `/autoscanon BTC`. Auto Scan chỉ cho 1 symbol/user để tiết kiệm tài nguyên.

## Text input/output

Bản này dùng lại text prompt/text output cho GLM như mode thủ công cũ. Không dùng JSON input/output cho phân tích chính. Python vẫn parse text output bằng regex để lưu candidate/history/auto-check.

## Auto Scan nghỉ đêm theo giờ Việt Nam

- Từ 00:00 đến trước 07:00, Auto Scan tự tắt tạm thời để không gọi Binance/DeepSeek/GLM.
- Lúc 07:00, bot chỉ tự bật lại những tài khoản đang bật trước khi bước vào giờ nghỉ.
- Nếu user chủ động dùng `/autoscanoff` trong giờ nghỉ, bot sẽ không tự bật lại vào buổi sáng.
- Mặc định không cần thêm Railway Variables. Có thể tùy chỉnh bằng `AUTO_SCAN_SLEEP_HOUR_VN` và `AUTO_SCAN_WAKE_HOUR_VN`.
