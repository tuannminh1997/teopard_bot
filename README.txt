# Teopard Bot V4 Lifecycle

Bản này dùng lifecycle mới cho prediction:

```text
PENDING_ENTRY -> ENTRY_FILLED -> WIN / LOSS / EXPIRED / AMBIGUOUS
PENDING_ENTRY -> NOT_FILLED
```

## Logic kiểm tra

Scalp:
- Chờ Entry tối đa 12h.
- Sau khi khớp Entry, theo dõi tối đa 72h.
- Job check mỗi 1h.
- Chấm TP/SL bằng nến 15M.

Swing:
- Chờ Entry tối đa 24h.
- Sau khi khớp Entry, theo dõi tối đa 7 ngày.
- Job check mỗi 12h.
- Chấm TP/SL bằng nến 1H.

Auto-check chỉ cập nhật DB, không gửi kết quả tự động cho user/admin. User muốn xem kết quả thì dùng /history, /stats hoặc /dashboard. Admin muốn xem tổng hệ thống thì dùng /historyall, /statsall hoặc /dashboardall; muốn ép kiểm tra ngay dùng /checknow.

## Lệnh user

- `/start`
- `/whoami`
- `/help`
- `/listsymbols`
- `/stats` — thống kê của chính user đang dùng lệnh
- `/stats BTC` — thống kê BTCUSDT của chính user đang dùng lệnh
- `/history` — lịch sử của chính user đang dùng lệnh
- `/history BTC` — lịch sử BTCUSDT của chính user đang dùng lệnh
- `/dashboard` — dashboard của chính user đang dùng lệnh

Lưu ý: Telegram menu không hiển thị command kèm tham số, nên `/stats BTC` và `/history BTC` phải gõ tay.

## Lệnh admin

Admin có toàn bộ lệnh user. Từ bản này, `/stats`, `/history`, `/dashboard` của admin cũng chỉ xem dữ liệu của chính admin để tránh rối.

Admin muốn xem toàn hệ thống dùng lệnh riêng:

- `/statsall`
- `/statsall BTC`
- `/historyall`
- `/historyall BTC`
- `/dashboardall`

Lệnh quản trị:

- `/adduser 123456789`
- `/removeuser 123456789`
- `/listusers`
- `/setlimit 123456789 10`
- `/resetusage 123456789`
- `/addsymbol BTC`
- `/removesymbol BTC`
- `/checknow`
- `/clearhistory CONFIRM`

Menu riêng của admin đã được set bằng `BotCommandScopeChat`, nên admin sẽ thấy đủ các lệnh quản trị trong menu Telegram sau khi bot restart/redeploy.

## Feature engineering V4.2

Bot tính sẵn bằng Python trước khi gửi Claude:
- EMA7/25/50, RSI6/14, MACD, Volume ratio, ATR14.
- Chuỗi nến, wick/body của nến cuối.
- Market structure, đỉnh/đáy gần/biên lớn.
- Fibonacci 0.382/0.5/0.618 từ swing đã tính.
- Vùng quét Long/Short gần và sâu từ pivot/equal high/equal low.
- Rủi ro tối thiểu đề xuất theo ATR/giá.

Cấu trúc Hybrid AI Engine:
- Python chỉ tính dữ liệu cứng và bản đồ kỹ thuật.
- Claude tự phân tích, tự chọn LONG/SHORT và tự đặt Entry/SL/TP.
- Python không còn chặn risk/format trước khi gửi user. Claude trả phản hồi thế nào thì bot gửi user phản hồi đó. Python chỉ parse Entry/SL/TP để lưu auto-check nếu đủ số.
- Nếu kế hoạch chưa hợp lệ, bot KHÔNG tự sửa và KHÔNG gọi Claude sửa lại; bot ẩn tín hiệu đó và lưu hidden REJECTED_PLAN để học/debug.

Prompt đã chặn Claude tự bịa Fibonacci/vùng quét nếu Python không gửi dữ liệu.

## Railway env

- `BOT_TOKEN`
- `ANTHROPIC_API_KEY`
- `ADMIN_USER_IDS=5920124635`
- `DB_PATH=/data/bot.db`

Không commit `.env` hoặc `bot.db`.


V4.1 privacy/history reset update:
- User thường chỉ xem /stats, /history, /dashboard của chính mình.
- Admin xem được thống kê/lịch sử toàn hệ thống bằng /statsall, /historyall, /dashboardall.
- Thêm /clearhistory CONFIRM cho admin để xóa toàn bộ prediction/history nhưng giữ whitelist và allowed_symbols.

V4.3 Hybrid AI Engine update:
- Bỏ kế hoạch tham chiếu LONG/SHORT cứng khỏi prompt.
- Python cung cấp dữ liệu cứng: ATR/Fibonacci/structure/liquidity/risk floor.
- Claude tự ra chiến lược và tự đặt Entry/SL/TP.
- Python không kiểm tra logic/risk để ẩn tín hiệu nữa. Claude tự chịu trách nhiệm phân tích Entry/SL/TP; bot chỉ parse số để auto-check nếu có đủ Entry/SL/TP.
- Nếu output chưa hợp lệ, bot KHÔNG tự sửa và KHÔNG gọi Claude sửa lại; bot ẩn tín hiệu đó để tránh gửi plan thiếu/không an toàn cho user.

V4.4 per-user learning update:
- Claude learning history được lọc theo user đang phân tích.
- User A phân tích thì Claude chỉ nhận lịch sử của User A cho cùng symbol/mode.
- User B không bị ảnh hưởng bởi lịch sử của User A.
- Admin khi tự phân tích cũng chỉ dùng lịch sử của chính admin, không dùng lịch sử toàn hệ thống.

V4.5 admin self/global history update:
- `/history`, `/stats`, `/dashboard` luôn xem dữ liệu của chính người dùng lệnh, kể cả admin.
- Admin muốn xem toàn hệ thống dùng `/historyall`, `/statsall`, `/dashboardall`.
- `/historyall` và `/historyall BTC` hiện User ID / Chat ID để biết lệnh thuộc user nào.

V4.6 feature snapshot learning update:
- Thêm cột feature_snapshot trong bảng predictions.
- Mỗi prediction lưu thêm snapshot kỹ thuật ngắn gọn tại lúc phân tích: EMA/RSI/MACD/ATR/volume, cấu trúc, Fibonacci, vùng quét Long/Short, risk floor, chuỗi nến/wick.
- Khi Claude học từ 5 lịch sử gần nhất của chính user, prompt giờ có thêm Feature then để biết lệnh cũ WIN/LOSS trong bối cảnh cấu trúc/Fib/liquidity nào.
- market_snapshot được giữ gọn cho dữ liệu thị trường cơ bản, feature_snapshot tách riêng để tránh prompt history quá rối.

V4.8 rejected plan learning update:
- Nếu Claude trả LONG/SHORT, bot hiển thị trực tiếp cho user. Nếu parse đủ Entry/SL/TP thì bot lưu prediction để auto-check. Nếu không parse đủ số thì bot vẫn hiển thị phản hồi nhưng chỉ lưu hidden record để learning/debug, không auto-check.
- Bot vẫn lưu hidden record `REJECTED_PLAN` để Claude học/debug, nhưng không auto-check và không hiện trong /history, /stats, /dashboard.


Cập nhật format phản hồi:
- Bắt buộc có đủ các mục Thanh khoản, Quyết định, Entry/SL/TP, Kịch bản chính và Rủi ro.
- Bot không dùng cụm “swing gần/swing lớn” trong output cho user; thay bằng “đỉnh/đáy gần/biên lớn”.
- Nếu Claude trả thiếu format, Python không tự sửa nội dung; bot ẩn phản hồi đó, trả thông báo “chưa đủ dữ liệu hợp lệ để tạo phân tích” cho user và lưu hidden REJECTED_PLAN để học/debug.

V4.8 compact output guard update:
- Output cho user dùng format rút gọn: không hiện riêng “Bối cảnh” và “Cấu trúc”.
- Bot vẫn gửi dữ liệu EMA/RSI/MACD/ATR/Fibonacci/cấu trúc/vùng quét cho Claude để phân tích nội bộ.
- Prompt vẫn yêu cầu Claude trả đủ Thanh khoản, Quyết định, Entry/SL/TP, Kịch bản chính và Rủi ro, nhưng Python không ẩn phản hồi nếu Claude thiếu format.


Bản cập nhật thông báo lỗi:
- Bot không còn dùng Python validator để ẩn phản hồi. Claude trả thế nào thì user thấy thế đó. Chỉ khi không parse đủ Entry/SL/TP thì bot không đưa vào auto-check.
- Lỗi vẫn được lưu hidden dạng REJECTED_PLAN để phục vụ learning/debug, không xuất hiện trong history/stats/dashboard.

V4.9 Sonnet Analyst Mode update:
- Thêm quyền quyết định `NO_TRADE`: Claude không còn bị ép phải chọn LONG/SHORT khi setup xấu.
- Python gửi thêm `MARKET_REGIME_DO_PYTHON_PHAN_LOAI` để Claude biết thị trường đang trend, range/nhiễu, thanh khoản thấp hay biến động cao.
- Python gửi thêm `RAW_CANDLE_CONTEXT_CHON_LOC` gồm nến thô có body%, râu trên/dưới, volume và taker-buy ratio nếu có để Sonnet đọc hành vi giá tốt hơn.
- Claude phải so sánh nội bộ LONG / SHORT / NO_TRADE trước khi quyết định, nhưng không in bảng so sánh ra user.
- Nếu Claude chọn NO_TRADE, bot hiển thị phản hồi NO_TRADE của Claude cho user và lưu hidden learning record; NO_TRADE không auto-check và không hiện trong /history/stats/dashboard.


Ghi chú bản LLM/OpenRouter:
- Bot không dùng Python risk/format guard để sửa hoặc chặn lệnh AI.
- Phân tích chính vẫn có continuation nếu provider trả finish_reason/stop_reason=length.
- Call tóm tắt reasoning không continuation để tránh GLM lặp length do reasoning token.
- Có thể chỉnh LLM_MAX_OUTPUT_TOKENS trong Railway, mặc định 8000.


## Chuyển sang GLM 5.2 qua OpenRouter

Bản này hỗ trợ 2 provider AI:

### 1) Claude / Anthropic (mặc định)
```text
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-sonnet-5
```

### 2) GLM 5.2 qua OpenRouter
```text
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=z-ai/glm-5.2
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MAX_OUTPUT_TOKENS=8000
LLM_MAX_CONTINUATIONS=2
# Tuỳ chọn: reasoning effort cho phân tích chính. Nếu dùng GLM và không khai báo biến này, code mặc định xhigh.
OPENROUTER_REASONING_EFFORT=xhigh
# Summary mặc định tắt reasoning để không đốt token ẩn
LLM_SUMMARY_MAX_OUTPUT_TOKENS=600
OPENROUTER_SUMMARY_REASONING_EFFORT=off
```

Khi dùng OpenRouter/GLM thì không cần `ANTHROPIC_API_KEY`. Bot vẫn giữ cùng flow: Python tính dữ liệu kỹ thuật, model tự phân tích LONG/SHORT/NO_TRADE, Python chỉ parse Entry/SL/TP để auto-check nếu đủ số.


=== GLM 5.2 / OpenRouter setup ===
Bản này đã tương thích GLM 5.2 qua OpenRouter. Khi AI_PROVIDER=openrouter, bot dùng Chat Completions API với:
- messages system/user kiểu OpenAI-compatible
- max_completion_tokens cho output token
- finish_reason=length để gọi continuation nếu bị cắt

Railway Variables để chạy GLM 5.2:
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=z-ai/glm-5.2
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MAX_OUTPUT_TOKENS=8000
LLM_MAX_CONTINUATIONS=2
OPENROUTER_REASONING_EFFORT=xhigh
LLM_SUMMARY_MAX_OUTPUT_TOKENS=600
OPENROUTER_SUMMARY_REASONING_EFFORT=off

Ghi chú:
- OPENROUTER_REASONING_EFFORT chỉ dùng cho phân tích chính Entry/SL/TP.
- summarize_reasoning dùng LLM_SUMMARY_MAX_OUTPUT_TOKENS và mặc định tắt reasoning/continuation để tránh log length lặp lại.

Các biến Telegram/DB giữ nguyên:
BOT_TOKEN=...
ADMIN_USER_IDS=5920124635
DB_PATH=/data/bot.db

Muốn quay lại Anthropic-native:
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
CLAUDE_MODEL=claude-sonnet-5

Các biến provider không dùng có thể để dư trên Railway, code chỉ đọc provider tương ứng theo AI_PROVIDER.

V4.10 OpenRouter/GLM logging update:
- Khi AI_PROVIDER=openrouter và model có chữ "glm", nếu Railway chưa set OPENROUTER_REASONING_EFFORT thì bot mặc định dùng xhigh cho phân tích chính.
- Log LLM_RESPONSE có thêm call_type=main hoặc call_type=summary để dễ phân biệt chi phí phân tích chính và chi phí tóm tắt learning.
- PREDICTION_HISTORY_COUNT vẫn giữ 5 để Claude/GLM có đủ lịch sử học lại.
