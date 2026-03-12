from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
INDEX_FILE = PUBLIC_DIR / "index.html"
UNIVERSE_FILE = BASE_DIR / "universe.json"

app = FastAPI(title="Yahoo Finance Portfolio Dashboard")

CACHE = {
    "snapshot": None,
    "timestamp": 0
}
CACHE_TTL_SECONDS = 600

def load_universe():
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_universe(universe_dict):
    rows = []
    for group in universe_dict["groups"]:
        for item in group["items"]:
            rows.append({
                "group": group["name"],
                "group_color": group["color"],
                "symbol": item["symbol"],
                "label": item["label"],
                "target": item["target"]
            })
    return rows

def safe_pct_return(series: pd.Series, lookback: int):
    series = series.dropna()
    if len(series) <= lookback:
        return None
    start = float(series.iloc[-lookback - 1])
    end = float(series.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0

def ytd_return(series: pd.Series):
    series = series.dropna()
    if len(series) < 2:
        return None

    latest_date = pd.to_datetime(series.index[-1])
    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)

    series_ytd = series[series.index >= year_start]
    if len(series_ytd) < 2:
        return None

    start = float(series_ytd.iloc[0])
    end = float(series_ytd.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0

def annualized_volatility(close_series: pd.Series, window: int = 30):
    close_series = close_series.dropna()
    if len(close_series) < window + 1:
        return None
    log_returns = np.log(close_series / close_series.shift(1)).dropna().tail(window)
    if len(log_returns) < 2:
        return None
    return float(log_returns.std() * np.sqrt(252) * 100.0)

def avg_volume(volume_series: pd.Series, window: int = 20):
    volume_series = volume_series.dropna()
    if len(volume_series) < 1:
        return None
    return float(volume_series.tail(window).mean())

def extract_symbol_frame(download_df: pd.DataFrame, symbol: str, total_symbols: int):
    if total_symbols == 1:
        df = download_df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(symbol, axis=1, level=0)
            except Exception:
                pass
        return df

    if isinstance(download_df.columns, pd.MultiIndex):
        if symbol in download_df.columns.get_level_values(0):
            return download_df[symbol].copy()

    raise KeyError(f"Data not found for {symbol}")

def build_snapshot():
    universe = load_universe()
    items = flatten_universe(universe)
    symbols = [x["symbol"] for x in items]

    df = yf.download(
        tickers=symbols,
        period="8mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker"
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance returned empty dataset")

    result_rows = []

    for item in items:
        symbol = item["symbol"]
        try:
            sdf = extract_symbol_frame(df, symbol, len(symbols)).copy()
            sdf = sdf.dropna(how="all")
            if sdf.empty or "Close" not in sdf.columns:
                continue

            close = sdf["Close"].dropna()
            volume = sdf["Volume"].dropna() if "Volume" in sdf.columns else pd.Series(dtype=float)

            if len(close) < 2:
                continue

            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            daily_change_pct = ((last_price / prev_close) - 1.0) * 100.0 if prev_close != 0 else None

            spark = close.tail(20).round(6).tolist()

            row = {
                "group": item["group"],
                "group_color": item["group_color"],
                "symbol": symbol,
                "label": item["label"],
                "target": item["target"],
                "price": last_price,
                "prev_close": prev_close,
                "daily_change_pct": daily_change_pct,
                "ret_1m_pct": safe_pct_return(close, 21),
                "ret_3m_pct": safe_pct_return(close, 63),
                "ytd_pct": ytd_return(close),
                "vol_30d_pct": annualized_volatility(close, 30),
                "avg_volume_20d": avg_volume(volume, 20),
                "sparkline": spark
            }
            result_rows.append(row)

        except Exception:
            continue

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "count": len(result_rows),
        "rows": result_rows
    }

@app.get("/")
def root():
    return FileResponse(INDEX_FILE)

@app.get("/api/universe")
def api_universe():
    return JSONResponse(content=load_universe())

@app.get("/api/snapshot")
def api_snapshot(force: bool = False):
    current_time = time.time()

    if (
        not force
        and CACHE["snapshot"] is not None
        and (current_time - CACHE["timestamp"] < CACHE_TTL_SECONDS)
    ):
        return JSONResponse(content=CACHE["snapshot"])

    try:
        snapshot = build_snapshot()
        CACHE["snapshot"] = snapshot
        CACHE["timestamp"] = current_time
        return JSONResponse(content=snapshot)
    except Exception as e:
        if CACHE["snapshot"] is not None:
            stale = dict(CACHE["snapshot"])
            stale["stale"] = True
            stale["error"] = str(e)
            return JSONResponse(content=stale)
        raise HTTPException(status_code=500, detail=str(e))