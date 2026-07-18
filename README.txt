# Teopard Bot — DeepSeek V4 Flash → DeepSeek V4 Pro

Bản hiện tại dùng:

- `deepseek-v4-flash` làm prefilter mini-rubric cho Auto Scan.
- `deepseek-v4-pro` làm AI phân tích cuối cho manual và Auto Scan.
- Python tính chỉ báo/cấu trúc, kiểm tra Entry–SL–TP, RR, ATR và quản lý vòng đời lệnh.

## Flow phân tích

Manual:

1. Lấy nến Binance theo mode.
2. Python tính EMA, RSI, MACD, ATR, volume, Fibonacci, cấu trúc và đỉnh/đáy.
3. DeepSeek V4 Pro chọn LONG, SHORT hoặc NO TRADE và lập Entry/SL/TP.
4. Python kiểm tra rubric, nguồn level, hình học, RR và khoảng chống nhiễu ATR.
5. Lệnh manual chỉ vào history khi user bấm xác nhận theo dõi.

Auto Scan:

1. DeepSeek V4 Flash quét mini-rubric mỗi 15 phút.
2. Chỉ khi điểm hướng tốt nhất và độ chênh LONG/SHORT đạt ngưỡng mới gọi V4 Pro.
3. V4 Pro phân tích đầy đủ; Python guard kiểm tra lần cuối.
4. Tín hiệu gửi thành công được lưu đồng thời vào `/history` và `/autoscanlog`.
5. `/history` và `/autoscanlog` đều giữ 5 bản ghi gần nhất mỗi user.

## Không dùng pseudo-liquidation

Bản này không truyền vùng thanh lý/thanh khoản ước lượng từ OHLCV cho Flash hoặc Pro.

- BẢN ĐỒ LEVEL chỉ gồm cấu trúc, đỉnh/đáy, Fibonacci và EMA7/EMA25/EMA50.
- ATR, volume, RSI, MACD và nến đã đóng vẫn được gửi bình thường.
- Râu nến/cú quét đỉnh-đáy đã xảy ra được coi là hành động giá, không phải dữ liệu thanh lý thật.
- Auto-adjust TP, nếu bật, chỉ lấy target cấu trúc/Fibonacci; không dùng liquidity box ước lượng.
- Feature snapshot mới không chứa vùng dưới/trên ước lượng.
- Snapshot cũ trong DB được lọc khi đưa lại vào prompt; không cần xóa `bot.db`.
- Các hàm tính vùng cũ còn nằm trong `analyze.py` để tương thích mã nguồn, nhưng không còn được gọi trong prompt, level map, guard, manual hay Auto Scan.

## Timeframe

SCALP:

- 15M: timing và xác nhận nến.
- 1H: setup chính.
- 4H: xác nhận xu hướng.
- 1D: bối cảnh lớn.

SWING core:

- 4H: setup và vùng vào.
- 1D: xu hướng chính/quyết định.
- 1W: bối cảnh lớn.
- 1H: timing phụ, không tự quyết định bias.

Chỉ nến đã đóng được dùng làm xác nhận. Nến live 1H/4H/1D được tách riêng để mô tả tiến độ, volume theo tiến độ và tương tác EMA; đây không phải rule ép hướng.

Snapshot quyết định đồng bộ:

- Flash và AI cuối dùng cùng một snapshot được tạo một lần cho mỗi lượt phân tích.
- SCALP: chuỗi biến đổi 15M/1H/4H, live 1H/4H; 1D macro.
- SWING: chuỗi biến đổi 4H/1D/1W, live 4H/1D; 1H timing phụ.
- Snapshot gồm return 1/3/6 nến, EMA slope, RSI/MACD delta, pivot high/low, taker buy ratio và touch/retest/acceptance quanh EMA7/25/50.

## Railway

Dùng file `railway_deepseek_pro.env` hoặc `railway_variables.env.example` làm mẫu. Các biến chính:

```env
BOT_TOKEN=""
ADMIN_USER_IDS=""
DB_PATH="/data/bot.db"
AI_PROVIDER="deepseek"

DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-v4-flash"
DEEPSEEK_MAX_OUTPUT_TOKENS="3000"

DEEPSEEK_FINAL_MODEL="deepseek-v4-pro"
DEEPSEEK_FINAL_REASONING_EFFORT="high"
DEEPSEEK_FINAL_RETRY_REASONING_EFFORT="high"

AUTO_SCAN_INTERVAL_SECONDS="900"
AUTO_SCAN_MIN_PREFILTER_CONFIDENCE="62"
AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP="5"
AUTO_SCAN_MIN_FINAL_CONFIDENCE="62"
AUTO_SCAN_MIN_FINAL_SETUP_STRENGTH="62"
AUTO_SCAN_MAX_FINAL_AI_CALLS_PER_DAY="5"
AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES="180"
AUTO_SCAN_SLEEP_HOUR_VN="0"
AUTO_SCAN_WAKE_HOUR_VN="7"
```

`DEEPSEEK_FINAL_API_KEY` có thể để trống; code sẽ dùng chung `DEEPSEEK_API_KEY`.

## Quota và giờ nghỉ Auto Scan

- Quota mặc định: tối đa 5 lần bắt đầu gọi AI cuối trong ngày Auto Scan.
- Đủ quota: toàn bộ Auto Scan dừng, không gọi Binance, Flash hoặc Pro.
- 00:00–07:00 giờ Việt Nam: Auto Scan nghỉ.
- 07:00 hôm sau: reset quota và tự bật lại khi bị dừng do quota/giờ nghỉ.
- `/autoscanoff` thủ công không bị tự bật lại.

## Lệnh

User:

- `/start`
- `/help`
- `/listsymbols`
- `/history`
- `/stats`
- `/dashboard`
- `/autoscanon BTC`
- `/autoscanoff`
- `/autoscanstatus`
- `/autoscanlog`

Admin:

- `/adduser <id>`
- `/removeuser <id>`
- `/setlimit <id> <số lượt>`
- `/addsymbol BTC`
- `/removesymbol BTC`
- `/historyall`
- `/statsall`
- `/checknow`
- `/clearhistory CONFIRM`
- `/cleardrafts ALL CONFIRM`

## Scoring V39 — Python objective scores

Bản này đổi cách chấm điểm cuối:

- Model vẫn chọn LONG/SHORT/NO TRADE và lập Entry/SL/TP.
- Model vẫn trả rubric nội bộ để kiểm tra format/fallback.
- Python chấm lại Chất lượng kế hoạch và Điểm chắc chắn bằng dữ liệu cứng.

Chất lượng kế hoạch chỉ đo chất lượng kế hoạch:

- Entry đúng vùng kỹ thuật.
- SL nằm ngoài điểm vô hiệu.
- TP bám target thực tế.
- RR và room đủ đáng.
- Điều kiện kích hoạt rõ.
- Rủi ro nhiễu/thực thi.

Điểm chắc chắn chỉ đo hướng model chọn có được dữ liệu ủng hộ hay không:

- Đồng thuận hướng đa khung.
- Cấu trúc thị trường.
- Price action và EMA interaction.
- Diễn biến momentum.
- Volume và taker flow.
- Mức mâu thuẫn/kịch bản đối lập.

Output public đổi từ “Điểm chắc chắn: x%” sang “Điểm chắc chắn: x/100” để tránh hiểu nhầm đây là xác suất thắng đã được backtest.

## V41 — model-authoritative evidence flow

Bản này bỏ việc đưa preferred_direction, LONG support và SHORT support vào prompt AI cuối để tránh Python neo hướng model. DeepSeek Flash vẫn tự lọc nhanh từ snapshot kỹ thuật rút gọn để tiết kiệm chi phí, nhưng kết quả Flash không ép hướng AI cuối.

AI cuối nhận dữ liệu đầy đủ đã cải thiện: snapshot đồng bộ, EMA7/25/50 interaction, nến live 1H/4H chuẩn hóa theo tiến độ, chuỗi RSI/MACD/EMA/return, high/low cấu trúc và taker imbalance. Sau khi AI cuối tự chọn LONG/SHORT/NO TRADE, Python mới hậu kiểm Điểm chắc chắn và Chất lượng kế hoạch Entry/SL/TP.

Log vì vậy đọc theo thứ tự:
- DeepSeek Flash: lọc nhanh ứng viên LONG/SHORT từ snapshot rút gọn để tiết kiệm chi phí.
- AI cuối: tự quyết định LONG/SHORT/NO TRADE từ dữ liệu đầy đủ, không bị scorecard Python dẫn hướng.
- Python: hậu kiểm Điểm chắc chắn và Chất lượng kế hoạch.
