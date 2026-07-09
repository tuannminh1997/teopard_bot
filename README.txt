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


## Cập nhật JSON output nội bộ

- Model được yêu cầu trả về 1 JSON object nội bộ gồm decision, confidence, Entry, SL, TP1, TP2, activation, reason, main_scenario và risk_note.
- User vẫn thấy format text cũ trên Telegram; Python render JSON thành output cũ trước khi gửi.
- Nếu model lỡ trả text cũ hoặc JSON lỗi, code vẫn fallback sang parser regex cũ để bot không chết ngay.
- Candidate/history lưu theo field đã parse từ JSON thay vì phụ thuộc hoàn toàn vào regex text.

## Cập nhật JSON input + JSON output

Bản này đã chuyển cả dữ liệu đầu vào gửi vào model sang JSON nội bộ.

Flow mới:

- Python lấy nến Binance.
- Python tính indicator/cấu trúc/Fibonacci/vùng thanh khoản/risk/history/open plan.
- Python đóng gói toàn bộ thành JSON payload `teopard_model_input_v1_json`.
- Model đọc JSON input và trả JSON output theo contract.
- Python parse JSON output, validate, rồi render lại format Telegram cũ cho user.

User không thấy JSON input/output; giao diện Telegram vẫn giữ format cũ.

## Update current price display

Output Telegram hiện luôn in dòng `Giá hiện tại: ... USDT` ngay dưới dòng `QUYẾT ĐỊNH`, áp dụng cho cả LONG/SHORT và NO TRADE. Nếu model JSON không trả `current_price`, Python tự chèn giá hiện tại lấy từ Binance khi render.

## JSON input data contract V2

Bot hiện gửi dữ liệu cho model bằng JSON có contract rõ theo từng mode.

SCALP:
- 15M: 120 nến đã đóng, dùng cho Entry/timing/xác nhận nến.
- 1H: 200 nến đã đóng, dùng cho setup chính.
- 4H: 150 nến đã đóng, dùng cho bias/xu hướng lớn.
- 1D: 80 nến đã đóng, dùng cho bối cảnh lớn.

SWING:
- 1H: 120 nến đã đóng, dùng cho timing phụ.
- 4H: 220 nến đã đóng, dùng cho setup/vùng Entry chính.
- 1D: 220 nến đã đóng, dùng cho trend chính.
- 1W: 120 nến đã đóng, dùng cho macro context.

Model được nhắc phải tôn trọng contract này: SCALP không dùng 15M làm xu hướng chính; SWING không dùng 1H làm bias chính. Output gửi user vẫn được render theo format cũ.

## Hotfix timeout provider AI
Nếu gặp lỗi `Read timed out` từ Z.AI/OpenRouter, đây thường là lỗi tạm thời từ provider hoặc request quá lâu. Bản này có retry tự động.
Có thể chỉnh Railway variables:

```env
LLM_MAIN_TIMEOUT_SECONDS=180
LLM_SUMMARY_TIMEOUT_SECONDS=60
LLM_API_RETRIES=2
LLM_RETRY_SLEEP_SECONDS=2
```

## Auto Scan Mode

Auto Scan là mode riêng, không thay thế phân tích thủ công.

Flow:

- Mỗi 15 phút bot lấy data Binance.
- DeepSeek v4 flash lọc nhanh xem có tín hiệu đáng phân tích sâu không.
- Nếu lọc nhanh đạt ngưỡng, bot gửi data đầy đủ sang GLM/Z.AI như mode thủ công.
- Nếu GLM/Z.AI trả LONG/SHORT đủ confidence, bot gửi tín hiệu cho user.
- Auto Scan không hiện nút "Tôi đã đặt lệnh theo phân tích này".
- Tín hiệu Auto Scan được lưu thẳng vào history/predictions để auto-check.

Lệnh user:

- `/autoscanon BTC` - bật Auto Scan cho BTC
- `/autoscanoff` - tắt Auto Scan
- `/autoscanstatus` - xem trạng thái Auto Scan

Railway variables Auto Scan:

```env
DEEPSEEK_API_KEY="..."
DEEPSEEK_BASE_URL="https://openrouter.ai/api/v1"
DEEPSEEK_MODEL="deepseek/deepseek-v4-flash"
DEEPSEEK_TIMEOUT_SECONDS="60"
DEEPSEEK_MAX_OUTPUT_TOKENS="700"
DEEPSEEK_TEMPERATURE="0.05"

AUTO_SCAN_INTERVAL_SECONDS="900"
AUTO_SCAN_MODES="short"
AUTO_SCAN_MIN_PREFILTER_CONFIDENCE="62"
AUTO_SCAN_MIN_FINAL_CONFIDENCE="60"
AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES="180"
AUTO_SCAN_SEND_NO_TRADE="0"
```

Không cần set AUTO_SCAN_SYMBOLS nữa. User chọn symbol bằng lệnh /autoscanon BTC hoặc /autoscanon BTC.

## Auto Scan V2 - closed candle scheduler + logs

Auto Scan now checks every 60 seconds internally but only runs once per closed-candle slot based on `AUTO_SCAN_INTERVAL_SECONDS`.
For 15-minute scanning, it scans around `00/15/30/45` after `AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS` seconds.

Commands:
- `/autoscanon BTC` - enable Auto Scan for one symbol.
- `/autoscanoff` - disable Auto Scan.
- `/autoscanstatus` - show status, latest scan, next scan and latest pipeline result.
- `/autoscanlog` - show recent scan logs.

DeepSeek prefilter receives a compact JSON input for cheaper screening. GLM/Z.AI full analysis still receives the full JSON input contract, same as manual mode.

Railway optional variables:
```env
AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS="5"
AUTO_SCAN_LOG_LIMIT="20"
AUTO_SCAN_DEBUG="0"
```

## Auto Scan scheduler tick

Optional Railway variables:

```env
AUTO_SCAN_SCHEDULER_TICK_SECONDS="60"
AUTO_SCAN_CANDLE_CLOSE_DELAY_SECONDS="5"
AUTO_SCAN_DEBUG="0"
```

`AUTO_SCAN_SCHEDULER_TICK_SECONDS` controls how often the bot wakes up to check whether a new closed-candle slot should be scanned. It does not call Binance/DeepSeek/GLM unless the slot is actually due and has not been scanned yet.

## Compact JSON input token fix
Bản này giữ JSON input/output nhưng tối ưu token: nến gửi vào model dùng dạng compact `columns + rows`, không còn lặp key `timestamp/open/high/low/close/volume` ở từng nến. Các legacy text block dài trong payload cũng được bỏ/rút gọn để prompt không phình token quá lớn.
