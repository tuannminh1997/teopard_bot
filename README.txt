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

- Python gửi OHLCV, nến đã đóng, nến live trung lập và các phép tính kỹ thuật khách quan như EMA, RSI, MACD, ATR, Fibonacci, swing và cấu trúc.
- Model tự suy luận Entry, SL, TP1 và TP2; không có Level Map, catalog hoặc ID giá bắt buộc.
- ATR, volume, RSI, MACD và nến đã đóng vẫn được gửi bình thường.
- Râu nến/cú quét đỉnh-đáy đã xảy ra được coi là hành động giá, không phải dữ liệu thanh lý thật.
- Mặc định Python không tự chỉnh Entry/SL/TP; các mức do model chọn được giữ nguyên.
- Feature snapshot mới không chứa vùng dưới/trên ước lượng.
- Snapshot cũ trong DB được lọc khi đưa lại vào prompt; không cần xóa `bot.db`.

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
AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE="62"
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

## Scoring V43 — model confidence + Python plan quality

Bản này đổi lại cách chấm điểm cuối để tránh Python over-filter:

- Model vẫn chọn LONG/SHORT/NO TRADE và lập Entry/SL/TP.
- Model vẫn trả rubric nội bộ để kiểm tra format/fallback.
- Python chỉ chấm lại một Điểm tín hiệu duy nhất do model cuối tự chấm. Python chỉ parse điểm này và guard lỗi cứng Entry/SL/TP/RR.

Rubric Điểm tín hiệu gồm 5 mục: hướng/bối cảnh, Entry/timing, SL/TP/RR, mâu thuẫn/rủi ro nhiễu và thực thi thực tế.

- Entry đúng vùng kỹ thuật.
- SL nằm ngoài điểm vô hiệu.
- TP bám target thực tế.
- RR và room đủ đáng.
- Điều kiện kích hoạt rõ.
- Rủi ro nhiễu/thực thi.

Điểm tín hiệu do model cuối tự đánh giá dựa trên dữ liệu được cung cấp:

- Đồng thuận hướng đa khung.
- Cấu trúc thị trường.
- Price action và EMA interaction.
- Diễn biến momentum.
- Volume và taker flow.
- Mức mâu thuẫn/kịch bản đối lập.

Output public chỉ dùng “Điểm tín hiệu: x/100”; không còn Chất lượng kế hoạch/Độ chắc chắn/Độ mạnh setup.

## V41 — model-authoritative evidence flow

Bản này bỏ việc đưa preferred_direction, LONG support và SHORT support vào prompt AI cuối để tránh Python neo hướng model. DeepSeek Flash vẫn tự lọc nhanh từ snapshot kỹ thuật rút gọn để tiết kiệm chi phí, nhưng kết quả Flash không ép hướng AI cuối.

AI cuối nhận dữ liệu đầy đủ đã cải thiện: snapshot đồng bộ, EMA7/25/50 interaction, nến live 1H/4H chuẩn hóa theo tiến độ, chuỗi RSI/MACD/EMA/return, high/low cấu trúc và taker imbalance. Sau khi AI cuối tự chọn LONG/SHORT/NO TRADE, Python chỉ parse Điểm tín hiệu và guard lỗi cứng Entry/SL/TP/RR.

Log vì vậy đọc theo thứ tự:
- DeepSeek Flash: lọc nhanh ứng viên LONG/SHORT từ snapshot rút gọn để tiết kiệm chi phí.
- AI cuối: tự quyết định LONG/SHORT/NO TRADE từ dữ liệu đầy đủ, không bị scorecard Python dẫn hướng.
- Python: chỉ parse Điểm tín hiệu và guard lỗi cứng Entry/SL/TP/RR.

Cập nhật V42 - Auto Scan log dễ đọc hơn:
- DeepSeek prefilter không còn hiển thị kiểu “NEUTRAL 43/100”.
- Khi LONG và SHORT gần cân bằng, /autoscanstatus và /autoscanlog hiển thị cả hai điểm: “LONG x/100 | SHORT y/100 (gần cân bằng, chênh z)”.
- Log mới lưu riêng điểm LONG, điểm SHORT và độ chênh để debug rõ ràng hơn; log cũ vẫn được đọc lại từ phần ghi chú nếu có.

## V42 Auto Scan prefilter
- DeepSeek Flash chấm điểm lọc nhanh LONG/SHORT. Python chỉ parse, cộng điểm, kiểm tra ngưỡng và độ chênh để quyết định có gọi AI cuối hay không.
- Nếu không parse được format của Flash, /autoscanlog sẽ ghi "Không parse được mini-rubric", không hiển thị LONG 0/SHORT 0 như điểm thật.
- AI cuối không nhận điểm LONG/SHORT của Flash để tránh bị neo hướng; AI cuối vẫn tự quyết định LONG / SHORT / NO TRADE từ snapshot đầy đủ.


Cập nhật V43 - không dùng Python confidence làm gate mặc định:
- AI cuối vẫn tự chọn LONG/SHORT/NO TRADE và tự chấm Điểm tín hiệu qua rubric nội bộ.
- Python không chấm rubric cuối; chỉ guard lỗi cứng Entry/SL/TP/RR.
- Điểm kiểm tra dữ liệu Python vẫn được tính để debug, nhưng không chặn Auto Scan mặc định vì dễ quá bảo thủ trong thị trường chuyển pha.
- Muốn bật lại gate dữ liệu Python thì set AUTO_SCAN_USE_PYTHON_CONFIDENCE_GATE=1, nhưng mặc định nên để 0.


Cập nhật V44 - một rubric cuối duy nhất:
- AI cuối tự chọn LONG/SHORT/NO TRADE và tự chấm Điểm tín hiệu /100.
- Python không tự chấm confidence/setup nữa; chỉ parse Điểm tín hiệu, lọc ngưỡng và guard lỗi cứng Entry/SL/TP/RR.
- Output user chỉ còn: QUYẾT ĐỊNH, Điểm tín hiệu, Entry/SL/TP, Kích hoạt và Rủi ro. Không hiện Kịch bản chính, Chất lượng kế hoạch hay Điểm tin cậy AI.
- Auto Scan gửi user khi Điểm tín hiệu >= AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE và kế hoạch không lỗi cứng.
