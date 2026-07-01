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
- Python validator kiểm tra lại Entry/SL/TP trước khi lưu prediction để auto-check.
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
- Python validator kiểm tra logic LONG/SHORT, khoảng cách SL, TP1/TP2 và entry quá xa trước khi lưu DB.
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
- Nếu Claude trả Entry/SL/TP không đạt validator, bot KHÔNG hiển thị plan lỗi cho user.
- Bot chỉ trả thông báo ngắn: nếu thiếu format/thiếu mục bắt buộc thì báo chưa đủ dữ liệu hợp lệ để tạo phân tích; nếu Entry/SL/TP không đạt risk thì báo chưa có setup hợp lệ.
- Bot vẫn lưu hidden record `REJECTED_PLAN` để Claude học/debug, nhưng không auto-check và không hiện trong /history, /stats, /dashboard.


Cập nhật format phản hồi:
- Bắt buộc có đủ các mục Thanh khoản, Quyết định, Entry/SL/TP, Kịch bản chính và Rủi ro.
- Bot không dùng cụm “swing gần/swing lớn” trong output cho user; thay bằng “đỉnh/đáy gần/biên lớn”.
- Nếu Claude trả thiếu format, Python không tự sửa nội dung; bot ẩn phản hồi đó, trả thông báo “chưa đủ dữ liệu hợp lệ để tạo phân tích” cho user và lưu hidden REJECTED_PLAN để học/debug.

V4.8 compact output guard update:
- Output cho user dùng format rút gọn: không hiện riêng “Bối cảnh” và “Cấu trúc”.
- Bot vẫn gửi dữ liệu EMA/RSI/MACD/ATR/Fibonacci/cấu trúc/vùng quét cho Claude để phân tích nội bộ.
- Python validator bắt buộc phản hồi có đủ Thanh khoản, Quyết định, Entry/SL/TP, Kịch bản chính và Rủi ro để tránh output bị cụt.


Bản cập nhật thông báo lỗi:
- Nếu Claude trả thiếu format hoặc kế hoạch Entry/SL/TP không đạt validator, bot không hiển thị plan lỗi.
- User chỉ thấy một thông báo chung: “⚠️ Teopard chưa tìm thấy setup hợp lệ để tạo tín hiệu.”
- Lỗi vẫn được lưu hidden dạng REJECTED_PLAN để phục vụ learning/debug, không xuất hiện trong history/stats/dashboard.

V4.9 Sonnet Analyst Mode update:
- Thêm quyền quyết định `NO_TRADE`: Claude không còn bị ép phải chọn LONG/SHORT khi setup xấu.
- Python gửi thêm `MARKET_REGIME_DO_PYTHON_PHAN_LOAI` để Claude biết thị trường đang trend, range/nhiễu, thanh khoản thấp hay biến động cao.
- Python gửi thêm `RAW_CANDLE_CONTEXT_CHON_LOC` gồm nến thô có body%, râu trên/dưới, volume và taker-buy ratio nếu có để Sonnet đọc hành vi giá tốt hơn.
- Claude phải so sánh nội bộ LONG / SHORT / NO_TRADE trước khi quyết định, nhưng không in bảng so sánh ra user.
- Nếu Claude chọn NO_TRADE, bot không hiển thị phân tích đó như tín hiệu, không auto-check, không hiện trong /history/stats/dashboard; bot chỉ lưu hidden learning record để lần sau học được lúc nào nên đứng ngoài.
- User vẫn chỉ thấy thông báo chung: “⚠️ Teopard chưa tìm thấy setup hợp lệ để tạo tín hiệu.”
