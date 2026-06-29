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

Kết quả tự động gửi cho user tạo prediction và admin. Format kết quả có thời gian phân tích theo giờ Việt Nam, Entry, SL, TP1, TP2, giá khớp Entry, giá check, thời gian giữ lệnh và ID prediction.

## Lệnh user

- `/start`
- `/whoami`
- `/help`
- `/listsymbols`
- `/stats`
- `/stats BTC`
- `/history`
- `/history BTC`
- `/dashboard`

Lưu ý: Telegram menu không hiển thị command kèm tham số, nên `/stats BTC` và `/history BTC` phải gõ tay.

## Lệnh admin

Admin có toàn bộ lệnh user, cộng thêm:

- `/adduser 123456789`
- `/removeuser 123456789`
- `/listusers`
- `/setlimit 123456789 10`
- `/resetusage 123456789`
- `/addsymbol BTC`
- `/removesymbol BTC`
- `/checknow`

Menu riêng của admin đã được set bằng `BotCommandScopeChat`, nên admin sẽ thấy đủ các lệnh quản trị trong menu Telegram sau khi bot restart/redeploy.

## Feature engineering V4.2

Bot tính sẵn bằng Python trước khi gửi Claude:
- EMA7/25/50, RSI6/14, MACD, Volume ratio, ATR14.
- Chuỗi nến, wick/body của nến cuối.
- Market structure, swing gần/swing lớn.
- Fibonacci 0.382/0.5/0.618 từ swing đã tính.
- Vùng quét Long/Short gần và sâu từ pivot/equal high/equal low.
- Rủi ro tối thiểu đề xuất theo ATR/giá.

Cấu trúc Hybrid AI Engine:
- Python chỉ tính dữ liệu cứng và bản đồ kỹ thuật.
- Claude tự phân tích, tự chọn LONG/SHORT và tự đặt Entry/SL/TP.
- Python validator kiểm tra lại Entry/SL/TP trước khi lưu prediction để auto-check.
- Nếu kế hoạch chưa hợp lệ, bot gọi Claude sửa lại một lần.

Prompt đã chặn Claude tự bịa Fibonacci/vùng quét nếu Python không gửi dữ liệu.

## Railway env

- `BOT_TOKEN`
- `ANTHROPIC_API_KEY`
- `ADMIN_USER_IDS=5920124635`
- `DB_PATH=/data/bot.db`

Không commit `.env` hoặc `bot.db`.


V4.1 privacy/history reset update:
- User thường chỉ xem /stats, /history, /dashboard của chính mình.
- Admin xem được thống kê/lịch sử toàn hệ thống.
- Thêm /clearhistory CONFIRM cho admin để xóa toàn bộ prediction/history nhưng giữ whitelist và allowed_symbols.

V4.3 Hybrid AI Engine update:
- Bỏ kế hoạch tham chiếu LONG/SHORT cứng khỏi prompt.
- Python cung cấp dữ liệu cứng: ATR/Fibonacci/structure/liquidity/risk floor.
- Claude tự ra chiến lược và tự đặt Entry/SL/TP.
- Python validator kiểm tra logic LONG/SHORT, khoảng cách SL, TP1/TP2 và entry quá xa trước khi lưu DB.
- Nếu output chưa hợp lệ, bot gọi Claude sửa lại một lần.
