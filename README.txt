# Teopard Bot — V50 Objective Data + Independent Flash Reviewer

## Luồng phân tích

### Manual
1. Lấy dữ liệu Binance hiện tại, không gửi history hoặc kế hoạch đang mở.
2. Python chỉ chuẩn bị dữ liệu khách quan.
3. Model chính (thường là DeepSeek V4 Pro/GLM) làm 2 phase:
   - Phase A: market thesis.
   - Phase B: Entry/SL/TP + evidence timestamp.
4. DeepSeek V4 Flash nhận lại cùng market packet và nguyên plan của model chính.
5. Flash chỉ review và chấm 6 tiêu chí; không được sửa hướng hoặc Entry/SL/TP.
6. Python giữ nguyên plan model chính, chỉ ghép score reviewer và áp ngưỡng.

### Auto Scan
1. Flash prefilter chỉ chấm LONG/SHORT clarity, không tạo plan.
2. Phải đạt:
   - `AUTO_SCAN_MIN_PREFILTER_CONFIDENCE`
   - `AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP`
3. Bias phải cùng hướng đủ `AUTO_SCAN_DIRECTION_CONFIRMATIONS` snapshot liên tiếp.
4. Model chính lập plan.
5. Flash reviewer độc lập chấm vòng cuối.
6. Chỉ gửi khi:
   - reviewer APPROVE;
   - score đạt ngưỡng;
   - trạng thái là `READY_TO_ENTER`.
7. `SETUP_WAITING_TRIGGER` chỉ lưu snapshot nội bộ, chưa gửi user.

## Vai trò timeframe

SCALP:
- 4H: hướng/cấu trúc.
- 1H: setup, Entry, SL, target.
- 15M: timing.
- 1D: macro.
- Raw candles: 15M=16, 1H=48, 4H=36, 1D=12.
- Live candles: 1H, 4H và 15M.

SWING:
- 1D: hướng/cấu trúc.
- 4H: setup, Entry, SL, TP1.
- 1H: timing.
- 1W: macro/target mở rộng.
- Raw candles: 1H=16, 4H=60, 1D=50, 1W=24.
- Live candles: 4H, 1D và 1H.

## Dữ liệu gửi model

Có:
- OHLCV + timestamp.
- EMA7/EMA25/EMA50.
- RSI14.
- MACD + signal.
- Volume và taker-buy nếu Binance có.
- Swing/vùng phản ứng có timestamp.
- Số lần touch/reject/close-through và trạng thái fresh/tested/weakened.
- Nến live kèm tiến độ.

Không có:
- History.
- Open plan.
- Fear & Greed.
- ATR.
- Fibonacci.
- Python trend/regime label.
- Level Map.
- Preferred direction.
- RR/ATR guard ép giá.

## Entry/SL/TP

- Model chính tự chọn hoàn toàn.
- Entry, SL và TP1 bắt buộc có evidence bằng timeframe + timestamp/cụm nến.
- TP2 tùy chọn và có thể là `N/A`.
- Python không sửa số.
- Flash reviewer không sửa số.
- Reviewer chỉ APPROVE/REJECT và chấm:
  - Thesis: 20
  - Setup: 20
  - Entry: 20
  - SL: 15
  - Target: 15
  - Trigger: 10

## Trạng thái setup

- `READY_TO_ENTER`
- `SETUP_WAITING_TRIGGER`
- `NO_TRADE`

Không được vừa ghi READY vừa yêu cầu chờ xác nhận.

## Snapshot và outcome

Bảng `analysis_snapshots` lưu:
- input planner;
- output planner;
- output reviewer;
- reviewer score/verdict;
- setup status;
- model;
- data variant;
- current price.

Predictions bổ sung:
- lifecycle_status;
- reviewer_score/verdict;
- MAE;
- MFE.

Các lifecycle tương thích:
- WAITING_TRIGGER
- ENTRY_FILLED
- TP1_HIT
- SL_HIT
- EXPIRED_NOT_FILLED
- EXPIRED_AFTER_ENTRY
- AMBIGUOUS_TP_SL

## A/B data variants

```env
ANALYSIS_DATA_VARIANT="A"  # raw + indicators
ANALYSIS_DATA_VARIANT="B"  # A + swing timestamp
ANALYSIS_DATA_VARIANT="C"  # B + zone-quality statistics (khuyên dùng)
```

## Railway variables đề xuất

```env
AI_PROVIDER="deepseek"

DEEPSEEK_API_KEY=""
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-v4-flash"

DEEPSEEK_FINAL_BASE_URL="https://api.deepseek.com"
DEEPSEEK_FINAL_MODEL="deepseek-v4-pro"
DEEPSEEK_FINAL_REASONING_EFFORT="max"
DEEPSEEK_FINAL_RETRY_REASONING_EFFORT="max"

# Flash reviewer
DEEPSEEK_REVIEW_MODEL="deepseek-v4-flash"
DEEPSEEK_REVIEW_MAX_OUTPUT_TOKENS="1800"
DEEPSEEK_REVIEW_TEMPERATURE="0"
FINAL_REVIEW_MIN_SIGNAL_SCORE="72"

# Auto Scan
AUTO_SCAN_MIN_PREFILTER_CONFIDENCE="72"
AUTO_SCAN_PREFILTER_MIN_DIRECTION_GAP="20"
AUTO_SCAN_DIRECTION_CONFIRMATIONS="2"
AUTO_SCAN_MIN_FINAL_SIGNAL_SCORE="72"
AUTO_SCAN_INTERVAL_SECONDS="900"
AUTO_SCAN_SIGNAL_COOLDOWN_MINUTES="180"

# Manual gate
TEOPARD_MIN_SIGNAL_SCORE="62"

# Data experiment
ANALYSIS_DATA_VARIANT="C"
```

`DEEPSEEK_FINAL_API_KEY` có thể bỏ trống nếu dùng chung `DEEPSEEK_API_KEY`.

## Lưu ý

- Planner output luôn được giữ trong biến riêng.
- Reviewer output được lưu riêng.
- Bot không ghi đè planner result bằng reviewer result.
- Entry/SL/TP gửi user luôn lấy từ planner.
- Điểm tín hiệu gửi user lấy từ Flash reviewer.
