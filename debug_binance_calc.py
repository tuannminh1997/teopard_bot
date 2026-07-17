"""
Debug Teopard market calculations without calling the LLM.

Usage:
  python debug_binance_calc.py ETH short
  python debug_binance_calc.py BTC short --llm-output last_output.txt
  python debug_binance_calc.py ETH swing

It fetches Binance candles, computes the same indicators/structure/SL/RR logic
used by analyze.py, and prints why a parsed LLM trade would be accepted or rejected.
Pseudo-liquidation zones are intentionally not calculated or printed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

# Debug script does not call the LLM, so allow running even if anthropic is not installed.
try:
    import anthropic  # noqa: F401
except Exception:
    sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=lambda *args, **kwargs: None)

import analyze



def _print_atr_and_risk(timeframe_data, mode, price):
    print("\n=== ATR / RISK GUARD ===")
    labels = analyze._mode_labels(mode)
    for label in labels:
        df = timeframe_data.get(label)
        atr = analyze._current_atr(df)
        close = analyze._last_close(df)
        print(f"{label:>3}: close={analyze.fmt(close)} ATR14={analyze.fmt(atr)}")
    print(f"risk_reference(prompt)={analyze.fmt(analyze._risk_floor(timeframe_data, mode, price))}")
    print(f"min_stop_distance(guard)={analyze.fmt(analyze._minimum_stop_distance(timeframe_data, mode, price))}")
    print(f"structural_sl_buffer={analyze.fmt(analyze._structural_sl_buffer(timeframe_data, mode, price))}")


def _print_structural_levels(timeframe_data, mode, price):
    print("\n=== STRUCTURAL LEVELS USED FOR SL ===")
    for side in ("low", "high"):
        levels = analyze._collect_structural_levels(timeframe_data, mode, side)
        if side == "low":
            # Long invalidation levels below/near price first
            levels = sorted(levels, key=lambda x: abs(float(x["price"]) - price))[:12]
        else:
            levels = sorted(levels, key=lambda x: abs(float(x["price"]) - price))[:12]
        print(f"{side.upper()} levels:")
        for lv in levels:
            print(f"  {analyze.fmt(lv['price'])} | {lv.get('label')} {lv.get('kind')}")


def _validate_llm_output(path: Path, timeframe_data, mode, price):
    text = path.read_text(encoding="utf-8")
    pred = analyze.parse_prediction_from_output(text)
    print("\n=== PARSED LLM OUTPUT ===")
    print(pred)
    if (pred.get("direction") or "").upper() in ("LONG", "SHORT"):
        pred2, _ = analyze._normalize_trade_plan_structural_sl(pred, timeframe_data, mode, price, text)
    else:
        pred2 = pred
    print("\n=== AFTER STRUCTURAL SL NORMALIZE ===")
    print(pred2)
    rr = analyze._plan_worst_case_risk_reward(pred2)
    print("\n=== WORST CASE RR ===")
    print(rr)
    errors = analyze._validate_actionable_trade_plan(pred2, timeframe_data, mode, price, text)
    print("\n=== GUARD RESULT ===")
    if errors:
        print("REJECTED / NO_TRADE")
        for e in errors:
            print("-", e)
    else:
        print("ACCEPTED / CAN_TRACK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", help="BTC, ETH, or BTCUSDT/ETHUSDT")
    parser.add_argument("mode", nargs="?", default="short", choices=["short", "swing"])
    parser.add_argument("--llm-output", help="Path to a saved LLM output text to parse/validate")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    configs = analyze.SHORT_TERM_TIMEFRAMES if args.mode == "short" else analyze.LONG_TERM_TIMEFRAMES
    timeframe_data = {}
    print(f"Fetching Binance candles for {symbol} mode={args.mode}...")
    for label, (interval, limit) in configs.items():
        df = analyze.load_timeframe_data(symbol, interval, limit)
        timeframe_data[label] = df
        print(f"  {label}: interval={interval} limit={limit} rows={0 if df is None else len(df)}")

    if not any(df is not None and not df.empty for df in timeframe_data.values()):
        raise SystemExit("No Binance data fetched. Check network/DNS or Binance restrictions.")

    _price_text, price = analyze.get_current_price_str(symbol)
    if price is None:
        price = analyze._last_close_from_data(timeframe_data)
    print(f"\nCURRENT_PRICE={analyze.fmt(price)}")

    _print_atr_and_risk(timeframe_data, args.mode, price)
    _print_structural_levels(timeframe_data, args.mode, price)

    if args.llm_output:
        _validate_llm_output(Path(args.llm_output), timeframe_data, args.mode, price)
    else:
        print("\nTip: save a model response to last_output.txt then run:")
        print(f"  python debug_binance_calc.py {symbol} {args.mode} --llm-output last_output.txt")


if __name__ == "__main__":
    main()
