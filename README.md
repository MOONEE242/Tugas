# AI Trading Ecosystem — MT5 / Exness (XAUUSD-first)

Sistem AI quant trading berbasis ICT + MSRN + Alchemist + SNAR sebagai *expert
features*, digabung lewat feature engineering ke model ML/DL (XGBoost →
LSTM/Transformer), lalu diputuskan lewat risk engine sebelum eksekusi
otomatis ke MetaTrader 5.

Filosofi: **jangan mengejar win rate 100%**. Target sebenarnya adalah
*positive expectancy*, drawdown rendah, dan robust di data out-of-sample.
Model tidak pernah mengeksekusi order secara langsung — semua sinyal harus
lolos `risk/risk_manager.py` dulu.

## Struktur project

```
Tugas/
├── config/
│   └── config.yaml            # semua parameter (symbol, session, risk, threshold)
├── data/
│   ├── mt5_connector.py       # koneksi & download data dari MT5
│   ├── pipeline.py            # cleaning / resampling / cache parquet
│   ├── raw/                   # (gitignored) hasil download MT5
│   └── processed/             # (gitignored) parquet siap pakai
├── strategy/
│   ├── ict.py                 # BOS, MSS, FVG, liquidity sweep, OB, premium/discount, killzone, SMT
│   ├── msrn.py                # expert signal MSRN (rule-based, siap kamu sesuaikan)
│   ├── alchemist.py           # expert signal Alchemist
│   ├── snar.py                # expert signal SNAR
│   └── confluence.py          # menggabungkan ICT+MSRN+Alchemist+SNAR jadi confluence score
├── features/
│   └── feature_engineering.py # OHLCV → dataset numerik + label (triple-barrier)
├── models/
│   ├── dataset.py             # split time-series (train/val/test tanpa shuffle)
│   └── train_baseline.py      # baseline XGBoost (BUY/SELL/NO_TRADE)
├── risk/
│   └── risk_manager.py        # position sizing, daily loss limit, RR filter, duplicate protection
├── execution/
│   ├── mt5_executor.py        # order_send ke MT5 (SL/TP, lot dinamis)
│   ├── signal.py              # struktur Setup/Prediction/Signal
│   └── auto_entry.py          # loop utama: setup baru → validasi → entry berkali-kali
├── backtest/
│   ├── backtest.py            # VectorBT backtest dari sinyal historis
│   └── walk_forward.py        # walk-forward validation (rolling train/val/test)
├── notebooks/                 # eksplorasi ad-hoc
└── main.py                    # CLI: fetch-data | build-dataset | train | backtest | paper | live
```

## Kenapa urutannya seperti ini

1. **ICT dulu, bukan LSTM dulu.** `strategy/ict.py` mengubah konsep diskresioner
   (liquidity sweep, FVG, OB, BOS/MSS, premium/discount) menjadi kolom numerik.
2. **XGBoost sebagai baseline**, sebelum deep learning. Kalau XGBoost dari
   `models/train_baseline.py` sudah robust, baru kita tambah LSTM/Transformer.
3. **Model tidak pernah eksekusi langsung.** Alur selalu: `AI confidence → RR
   check → risk_manager.allow_trade() → execute`.
4. **Multi-entry per hari, dibatasi risk bukan jumlah trade.** Lihat
   `execution/auto_entry.py` — tidak ada `if trade_count >= N: stop()`, yang ada
   `if not risk_manager.allow_trade(setup): continue`.
5. **Time-series split, bukan random split.** `models/dataset.py` selalu
   mempertahankan urutan waktu (train masa lalu → val → test masa depan).

## Setup

### 1. Wajib
- Python 3.10+
- VS Code
- Git
- **MetaTrader 5** (terminal) + akun **Demo** dari broker Exness (atau broker
  MT5 lain) — buka dan login manual sekali agar terminal siap dipakai Python.

> Catatan penting: package `MetaTrader5` di PyPI **hanya berjalan di
> Windows** (ia membungkus terminal MT5 lewat IPC lokal). Semua kode di
> `data/mt5_connector.py` dan `execution/mt5_executor.py` mendeteksi ini secara
> otomatis: di Linux/macOS ia akan raise `MT5UnavailableError` yang jelas,
> bukan gagal diam-diam. Untuk development di container/Linux ini, gunakan
> data historis (CSV/parquet) lewat `pipeline.load_ohlc_csv()`, lalu jalankan
> versi live/executor-nya nanti di Windows (laptop kamu atau VPS Windows).

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Konfigurasi

Edit `config/config.yaml`: symbol, timeframe, session windows (WIB/UTC),
risk per trade, daily loss limit, minimum RR, minimum AI confidence.

Kredensial MT5 **jangan** dihardcode. Isi lewat environment variable:

```bash
export MT5_LOGIN=12345678
export MT5_PASSWORD="..."
export MT5_SERVER="Exness-MT5Trial"
```

`.env` dan `config/secrets.yaml` sudah ada di `.gitignore` — jangan pernah
commit kredensial broker.

## Menjalankan pipeline (di Windows, dengan MT5 terminal terbuka)

```bash
# Phase 1: download & simpan data historis XAUUSD multi-timeframe
python main.py fetch-data --symbol XAUUSD --timeframes M1,M5,M15,H1,H4,D1

# Phase 2-4: hitung fitur ICT/MSRN/Alchemist/SNAR + bangun dataset training
python main.py build-dataset --symbol XAUUSD --timeframe M15

# Phase 5: latih baseline XGBoost
python main.py train --dataset data/processed/XAUUSD_M15_dataset.parquet

# Phase 7: backtest sinyal historis dengan VectorBT
python main.py backtest --symbol XAUUSD --timeframe M15

# Phase 9: paper trading (loop auto-entry, dry-run, tidak kirim order asli)
python main.py paper --symbol XAUUSD --timeframe M15

# Phase 10: live/demo trading (order_send ke MT5 sungguhan — pastikan akun DEMO dulu)
python main.py live --symbol XAUUSD --timeframe M15
```

## Roadmap implementasi

| Fase | Status | Modul |
|---|---|---|
| 1. Data MT5 → parquet | ✅ scaffold siap | `data/` |
| 2. ICT engine (BOS/MSS/FVG/liquidity/OB/killzone) | ✅ implementasi penuh | `strategy/ict.py` |
| 3. MSRN/Alchemist/SNAR sebagai expert feature | ✅ scaffold rule-based, **perlu kamu sesuaikan aturan persisnya** | `strategy/msrn.py`, `alchemist.py`, `snar.py` |
| 4. Feature engineering + label (triple-barrier) | ✅ implementasi penuh | `features/feature_engineering.py` |
| 5. Baseline ML (XGBoost) | ✅ implementasi penuh | `models/train_baseline.py` |
| 6. Deep learning (LSTM/GRU/Transformer, PyTorch) | ⏳ belum — lakukan setelah baseline XGBoost terbukti robust out-of-sample | `models/` (belum dibuat) |
| 7. Backtest (VectorBT) | ✅ implementasi penuh | `backtest/backtest.py` |
| 8. Walk-forward validation | ✅ implementasi penuh | `backtest/walk_forward.py` |
| 9. Risk engine (position sizing, daily loss limit, multi-entry) | ✅ implementasi penuh | `risk/risk_manager.py` |
| 10. Auto-entry loop (multi-trade/hari) | ✅ implementasi penuh | `execution/auto_entry.py` |
| 11. MT5 executor (order_send, SL/TP) | ✅ implementasi penuh (butuh Windows untuk jalan) | `execution/mt5_executor.py` |
| 12. Dashboard (Next.js/Streamlit/FastAPI) | ⏳ belum | — |
| 13. PostgreSQL untuk trade log | ⏳ belum — saat ini pakai parquet/CSV | — |
| 14. VPS deployment | ⏳ belum — infrastruktur, bukan kode | — |

**Langkah berikutnya yang saya sarankan:** jalankan Fase 1 di laptop Windows
kamu (MT5 + `python main.py fetch-data`) untuk mendapatkan data historis
XAUUSD sungguhan, lalu kirim balik ke saya (atau commit ke `data/processed/`)
supaya `strategy/msrn.py`, `alchemist.py`, `snar.py` bisa disesuaikan dengan
aturan persis yang kamu pelajari — placeholder saat ini memakai heuristik
umum berbasis struktur ICT supaya pipeline bisa langsung dites end-to-end.
