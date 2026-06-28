# Teopard Telegram Bot

Teopard là Telegram bot phân tích kỹ thuật coin theo 2 mode:
- Scalp: 15m / 1H / 4H.
- Swing: 4H / 1D / 1W.

Bot lấy dữ liệu Binance, tính EMA/RSI/MACD/Volume, lấy Fear & Greed Index, gửi dữ liệu sang Claude/Anthropic để phân tích, sau đó lưu prediction vào SQLite để tự kiểm tra WIN/LOSS và dùng lịch sử cho lần phân tích sau.

## 1. Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. File .env cần có

Tạo file `.env` cùng cấp với `bot.py`:

```env
BOT_TOKEN=your_telegram_bot_token
ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-haiku-4-5-20251001
ADMIN_USER_IDS=123456789,987654321
DB_PATH=bot.db
```

## 3. Chạy bot

```powershell
python bot.py
```

## 4. Lệnh user

- `/start`: bắt đầu, lấy User ID và kích hoạt bot.
- `/whoami`: xem User ID.
- `/help`: xem hướng dẫn.
- `/listsymbols`: xem danh sách symbol được hỗ trợ.

## 5. Lệnh admin

- `/adduser 123456789`: thêm User ID vào whitelist.
- `/removeuser 123456789`: xóa User ID khỏi whitelist.
- `/listusers`: xem whitelist và số lượt đã dùng trong ngày.
- `/setlimit 123456789 10`: set giới hạn lượt/ngày cho user.
- `/resetusage 123456789`: reset lượt dùng hôm nay của user.
- `/addsymbol BTC`: thêm symbol được hỗ trợ.
- `/removesymbol BTC`: xóa symbol được hỗ trợ.

## 6. Flow sử dụng

1. User bấm `/start`.
2. User lấy User ID và gửi cho admin.
3. Admin cấp quyền bằng `/adduser user_id`.
4. Admin thêm coin bằng `/addsymbol BTC` nếu chưa có.
5. User nhập `BTC` vào chat.
6. Bot hỏi chọn Scalp hoặc Swing.
7. Bot phân tích và trả kết quả.
8. Bot lưu prediction vào SQLite.
9. Background job chạy mỗi giờ để kiểm tra prediction đến hạn.

## 7. Cơ chế tự check prediction

- Scalp: check sau 12 giờ.
- Swing: check sau 24 giờ.
- Bot dùng nến Binance 15m kể từ thời điểm tạo prediction.
- Nếu TP1 chạm trước SL: WIN.
- Nếu SL chạm trước TP1: LOSS.
- Nếu TP1 và SL cùng chạm trong một nến 15m: AMBIGUOUS.
- Nếu hết thời gian mà chưa chạm TP1/SL: EXPIRED.

## 8. Ghi chú

- Giới hạn mặc định là 10 lượt/ngày/người.
- Admin có thể đổi giới hạn bằng `/setlimit`.
- Kết quả auto-check hiện gửi cho admin, chưa gửi lại trực tiếp cho user tạo prediction.
