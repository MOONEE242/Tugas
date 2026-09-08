"""CLI entry point for the AI trading ecosystem.

    python main.py fetch-data     --symbol XAUUSD --timeframes M1,M5,M15,H1,H4,D1
    python main.py build-dataset  --symbol XAUUSD --timeframe M15
    python main.py train          --dataset data/processed/XAUUSD_M15_dataset.parquet
    python main.py backtest       --symbol XAUUSD --timeframe M15 [--model models/xgboost/baseline.joblib]
    python main.py paper          --symbol XAUUSD --timeframe M15 [--model ...]
    python main.py live           --symbol XAUUSD --timeframe M15 [--model ...] [--send-real-orders]

See README.md for the full phase-by-phase explanation of each command.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from utils.config import load_config


def cmd_fetch_data(args: argparse.Namespace) -> None:
    from data.mt5_connector import MT5Connector
    from data.pipeline import clean_ohlc, save_processed

    timeframes = args.timeframes.split(",")
    conn = MT5Connector()
    conn.connect()
    try:
        for tf in timeframes:
            print(f"Downloading {args.symbol} {tf} ({args.bars} bars)...")
            raw = conn.get_ohlc(args.symbol, tf, n_bars=args.bars)
            clean = clean_ohlc(raw)
            path = save_processed(clean, args.symbol, tf)
            print(f"  -> saved {len(clean)} bars to {path}")
    finally:
        conn.shutdown()


def cmd_build_dataset(args: argparse.Namespace) -> None:
    from data.pipeline import load_processed, save_processed
    from features.feature_engineering import build_dataset

    config = load_config()
    raw = load_processed(args.symbol, args.timeframe)

    secondary = None
    if args.secondary_symbol:
        secondary = load_processed(args.secondary_symbol, args.timeframe)

    print(f"Building features for {args.symbol} {args.timeframe} ({len(raw)} bars)...")
    dataset = build_dataset(raw, config, secondary_df=secondary)

    out_path = save_processed(dataset, args.symbol, f"{args.timeframe}_dataset")
    print(f"Dataset: {len(dataset)} rows, {dataset.shape[1]} columns -> {out_path}")
    print("\nTarget distribution:")
    print(dataset["target"].value_counts().rename({0: "NO_TRADE/LOSS", 1: "BUY_WIN", 2: "SELL_WIN"}))


def cmd_train(args: argparse.Namespace) -> None:
    from pathlib import Path

    from models.train_baseline import train

    dataset = pd.read_parquet(args.dataset)
    model_out = Path(args.model_out) if args.model_out else None
    metrics = train(dataset, train_frac=args.train_frac, val_frac=args.val_frac, model_out=model_out)
    print(json.dumps(metrics, indent=2, default=str))


def _load_model(path: str | None):
    if not path:
        return None
    import joblib

    bundle = joblib.load(path)
    return bundle["model"]


def cmd_backtest(args: argparse.Namespace) -> None:
    from data.pipeline import load_processed

    config = load_config()
    dataset = load_processed(args.symbol, f"{args.timeframe}_dataset")

    if args.model:
        from backtest.backtest import run_backtest
        from features.feature_engineering import feature_columns

        model = _load_model(args.model)
        feat_cols = feature_columns(dataset)
        preds = model.predict(dataset[feat_cols])
        direction = pd.Series("NO_TRADE", index=dataset.index)
        direction[preds == 1] = "BUY"
        direction[preds == 2] = "SELL"

        label_cfg = config["labeling"]
        atr, close = dataset["atr"], dataset["close"]
        sl = pd.Series(index=dataset.index, dtype=float)
        tp = pd.Series(index=dataset.index, dtype=float)
        buy, sell = direction == "BUY", direction == "SELL"
        sl[buy] = close[buy] - atr[buy] * label_cfg["sl_atr_multiple"]
        tp[buy] = close[buy] + atr[buy] * label_cfg["tp_atr_multiple"]
        sl[sell] = close[sell] + atr[sell] * label_cfg["sl_atr_multiple"]
        tp[sell] = close[sell] - atr[sell] * label_cfg["tp_atr_multiple"]

        bt_cfg = config["backtest"]
        result = run_backtest(
            dataset, direction, sl, tp,
            timeframe=args.timeframe,
            initial_cash=bt_cfg["initial_cash"], fee_pct=bt_cfg["fee_pct"], slippage_pct=bt_cfg["slippage_pct"],
        )
    else:
        from backtest.backtest import run_backtest_from_confluence

        print("(tidak ada --model, memakai rule-based confluence signal sebagai baseline sanity check)")
        result = run_backtest_from_confluence(dataset, config, min_score=args.min_score)

    print("\n=== Backtest stats ===")
    print(result.stats.to_string())


def cmd_walk_forward(args: argparse.Namespace) -> None:
    from backtest.walk_forward import run_walk_forward
    from data.pipeline import load_processed

    config = load_config()
    dataset = load_processed(args.symbol, f"{args.timeframe}_dataset")
    results = run_walk_forward(dataset, config, use_model=not args.rule_based)
    print(results.to_string())
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


def cmd_paper(args: argparse.Namespace) -> None:
    from data.pipeline import load_processed
    from execution.auto_entry import run_paper_replay
    from features.feature_engineering import feature_columns

    config = load_config()
    dataset = load_processed(args.symbol, f"{args.timeframe}_dataset")
    model = _load_model(args.model)
    feat_cols = feature_columns(dataset)

    log = run_paper_replay(dataset, args.symbol, config, feat_cols, model=model, starting_equity=args.equity)
    print(f"\n{len(log)} setups evaluated, {int(log['executed'].sum()) if len(log) else 0} executed (dry-run).")
    if len(log):
        print(log.tail(20).to_string())
    if args.out:
        log.to_csv(args.out, index=False)
        print(f"\nSaved full log to {args.out}")


def cmd_live(args: argparse.Namespace) -> None:
    from execution.auto_entry import run_live_loop
    from execution.mt5_executor import MT5Executor
    from features.feature_engineering import feature_columns

    config = load_config()
    model = _load_model(args.model)

    if args.send_real_orders:
        confirm = input(
            f"PERINGATAN: ini akan mengirim order SUNGGUHAN untuk {args.symbol} ke akun MT5 "
            "yang sedang login (pastikan akun DEMO, bukan live, kecuali kamu benar-benar yakin). "
            "Ketik 'YA' untuk lanjut: "
        )
        if confirm.strip().upper() != "YA":
            print("Dibatalkan.")
            sys.exit(1)
        executor = MT5Executor(dry_run=False)
    else:
        executor = MT5Executor(dry_run=True)
        print("Mode DRY-RUN: tidak ada order sungguhan yang dikirim. Pakai --send-real-orders untuk live/demo order_send.")

    # feature_columns needs a sample dataset shape - build a tiny one from
    # a fresh MT5 pull so live rows have exactly the same columns as training.
    from data.mt5_connector import MT5Connector
    from features.feature_engineering import build_dataset

    conn = MT5Connector()
    conn.connect()
    sample_raw = conn.get_ohlc(args.symbol, args.timeframe, n_bars=max(500, args.lookback))
    conn.shutdown()
    sample_dataset = build_dataset(sample_raw, config)
    feat_cols = feature_columns(sample_dataset)

    def on_decision(decision):
        print(
            f"[{decision.setup.timestamp}] {decision.setup.direction} {args.symbol} "
            f"entry={decision.setup.entry:.5f} sl={decision.setup.stop_loss:.5f} tp={decision.setup.take_profit:.5f} "
            f"conf={decision.prediction.confidence:.2%} rr={decision.setup.rr:.2f} "
            f"-> {'EXECUTED' if decision.allowed else 'SKIPPED'} ({decision.reason})"
        )

    run_live_loop(
        args.symbol,
        args.timeframe,
        config,
        feat_cols,
        model=model,
        executor=executor,
        starting_equity=args.equity,
        lookback_bars=args.lookback,
        poll_seconds=args.poll_seconds,
        on_decision=on_decision,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Ecosystem - MT5 / Exness")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-data", help="Download historical OHLCV from MT5")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframes", default="M1,M5,M15,H1,H4,D1")
    p.add_argument("--bars", type=int, default=50_000)
    p.set_defaults(func=cmd_fetch_data)

    p = sub.add_parser("build-dataset", help="Compute ICT/MSRN/Alchemist/SNAR features + labels")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--secondary-symbol", default=None, help="Correlated symbol for SMT divergence")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("train", help="Train baseline XGBoost model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--model-out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="VectorBT backtest on the processed dataset")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--model", default=None, help="Path to a trained model (joblib); omit to use rule-based confluence")
    p.add_argument("--min-score", type=float, default=0.6)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("walk-forward", help="Rolling train/val/test walk-forward validation")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--rule-based", action="store_true", help="Use rule-based confluence instead of retraining a model per window")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_walk_forward)

    p = sub.add_parser("paper", help="Replay dataset bar-by-bar through the auto-entry loop (dry-run)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("live", help="Live/demo loop against MT5 (Windows only)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--lookback", type=int, default=2000)
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--send-real-orders", action="store_true", help="Actually call MT5 order_send (default is dry-run)")
    p.set_defaults(func=cmd_live)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
