from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
INDEX_FILE = PUBLIC_DIR / "index.html" if (PUBLIC_DIR / "index.html").exists() else BASE_DIR / "index.html"
UNIVERSE_FILE = BASE_DIR / "universe.json"

TRADING_DAYS = 252

app = FastAPI(title="Institutional Portfolio Dashboard")


def load_universe() -> dict:
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_universe(universe_dict: dict) -> List[dict]:
    rows = []
    for group in universe_dict.get("groups", []):
        family = group.get("family", "Other")
        subgroup = group.get("name", "Unknown")
        color = group.get("color", "#64748b")
        for item in group.get("items", []):
            rows.append({
                "family": family,
                "group": subgroup,
                "group_color": color,
                "symbol": item.get("symbol"),
                "label": item.get("label"),
                "target": float(item.get("target", 0.0))
            })
    return rows


def get_rows(family: str = "ALL", group: str = "ALL", include_reference: bool = False) -> List[dict]:
    rows = flatten_universe(load_universe())
    if family != "ALL":
        rows = [r for r in rows if r["family"] == family]
    if group != "ALL":
        rows = [r for r in rows if r["group"] == group]
    if not include_reference:
        rows = [r for r in rows if r["family"] != "World Indices"]
    return rows


def universe_metadata() -> dict:
    u = load_universe()
    families = sorted({g.get("family", "Other") for g in u.get("groups", [])})
    groups_by_family = {}
    for g in u.get("groups", []):
        fam = g.get("family", "Other")
        groups_by_family.setdefault(fam, [])
        groups_by_family[fam].append({
            "name": g.get("name", "Unknown"),
            "color": g.get("color", "#64748b"),
            "count": len(g.get("items", []))
        })
    for fam in groups_by_family:
        groups_by_family[fam] = sorted(groups_by_family[fam], key=lambda x: x["name"])
    return {"families": families, "groups_by_family": groups_by_family, "groups": u.get("groups", [])}


def _extract_close(downloaded: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        frames = []
        for sym in symbols:
            if sym in downloaded.columns.get_level_values(0):
                sub = downloaded[sym]
                col = None
                for candidate in ["Adj Close", "Close"]:
                    if candidate in sub.columns:
                        col = candidate
                        break
                if col is not None:
                    s = pd.to_numeric(sub[col], errors="coerce").rename(sym)
                    frames.append(s)
        if not frames:
            return pd.DataFrame()
        prices = pd.concat(frames, axis=1)
    else:
        col = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
        if col not in downloaded.columns:
            return pd.DataFrame()
        name = symbols[0] if symbols else "Asset"
        prices = downloaded[[col]].rename(columns={col: name})
        prices[name] = pd.to_numeric(prices[name], errors="coerce")

    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def clean_aligned_prices(prices: pd.DataFrame, min_coverage: float = 0.70, min_obs: int = 60) -> tuple[pd.DataFrame, dict]:
    diag = {"input_columns": list(prices.columns), "dropped_columns": [], "coverage": {}, "start_dates": {}}
    if prices.empty:
        return prices, diag

    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.ffill()

    # column coverage check after forward fill
    cov = (prices.notna().sum() / len(prices)).to_dict()
    diag["coverage"] = {k: float(v) for k, v in cov.items()}
    keep_cols = [c for c, v in cov.items() if v >= min_coverage]
    diag["dropped_columns"] = [c for c in prices.columns if c not in keep_cols]
    prices = prices[keep_cols]
    if prices.empty:
        return prices, diag

    # equal-length alignment without over-killing the panel:
    # start from the latest first-valid date across kept columns
    first_valid = {}
    for c in prices.columns:
        idx = prices[c].first_valid_index()
        first_valid[c] = None if idx is None else idx.strftime("%Y-%m-%d")
    diag["start_dates"] = first_valid

    valid_idxs = [prices[c].first_valid_index() for c in prices.columns if prices[c].first_valid_index() is not None]
    if not valid_idxs:
        return pd.DataFrame(), diag

    common_start = max(valid_idxs)
    prices = prices.loc[common_start:].copy()
    prices = prices.ffill()
    prices = prices.dropna(axis=0, how="any")
    prices = prices.loc[:, prices.nunique(dropna=True) > 1]

    if len(prices) < min_obs:
        return pd.DataFrame(), diag
    return prices, diag


def fetch_prices(symbols: List[str], period: str = "2y", interval: str = "1d") -> tuple[pd.DataFrame, dict]:
    symbols = [s for s in symbols if s]
    if not symbols:
        return pd.DataFrame(), {"requested_symbols": [], "fetched_symbols": [], "dropped_symbols": [], "reason": "empty symbol list"}
    try:
        data = yf.download(
            tickers=symbols,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo Finance download failed: {e}")

    raw = _extract_close(data, symbols)
    if raw.empty:
        return raw, {"requested_symbols": symbols, "fetched_symbols": [], "dropped_symbols": symbols, "reason": "no raw close prices"}

    cleaned, diag = clean_aligned_prices(raw)
    fetched = list(raw.columns)
    dropped = [s for s in fetched if s not in cleaned.columns]
    outdiag = {
        "requested_symbols": symbols,
        "fetched_symbols": fetched,
        "retained_symbols": list(cleaned.columns),
        "dropped_symbols": dropped,
        "row_count": int(len(cleaned)),
        **diag
    }
    return cleaned, outdiag


def normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.div(prices.iloc[0]).mul(100.0)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return returns


def annualized_metrics(returns: pd.Series) -> Dict[str, float]:
    ann_return = float((1.0 + returns.mean()) ** TRADING_DAYS - 1.0)
    ann_vol = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    mdd = float(dd.min()) if not dd.empty else 0.0
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": mdd}


def make_portfolio(returns: pd.DataFrame, rows: List[dict], method: str = "current") -> Tuple[pd.Series, Dict[str, float]]:
    symbols = list(returns.columns)
    row_map = {r["symbol"]: r for r in rows}
    if method == "equal":
        w = np.repeat(1.0 / len(symbols), len(symbols))
    elif method == "inverse_vol":
        inv = 1.0 / returns.std(ddof=1).replace(0, np.nan)
        inv = inv.fillna(0.0)
        w = (inv / inv.sum()).to_numpy()
    else:
        raw = np.array([max(float(row_map[s]["target"]), 0.0) for s in symbols], dtype=float)
        if raw.sum() <= 0:
            w = np.repeat(1.0 / len(symbols), len(symbols))
        else:
            w = raw / raw.sum()
    port = returns.mul(w, axis=1).sum(axis=1)
    weights = {s: float(v) for s, v in zip(symbols, w)}
    return port, weights


def ensure_min_assets(rows: List[dict], min_assets: int = 2):
    if len(rows) < min_assets:
        raise HTTPException(status_code=400, detail=f"At least {min_assets} assets are required after filtering. Current selection has {len(rows)}.")


def chart_records(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    recs = []
    for dt, row in df.iterrows():
        r = {"date": dt.strftime("%Y-%m-%d")}
        for col, val in row.items():
            r[col] = None if pd.isna(val) else float(val)
        recs.append(r)
    return recs


def series_records(series: pd.Series, name: str) -> List[dict]:
    if series.empty:
        return []
    return [{"date": idx.strftime("%Y-%m-%d"), name: float(val)} for idx, val in series.items()]


def technical_indicators(prices: pd.Series) -> dict:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    rolling_vol = prices.pct_change().rolling(20).std() * math.sqrt(TRADING_DAYS)
    latest = prices.iloc[-1]
    out = {
        "last_price": float(latest),
        "rsi14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else None,
        "macd_signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else None,
        "macd_hist": float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else None,
        "sma20": float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else None,
        "sma50": float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else None,
        "sma200": float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else None,
        "dist_sma20_pct": float((latest / sma20.iloc[-1] - 1) * 100) if not pd.isna(sma20.iloc[-1]) else None,
        "dist_sma50_pct": float((latest / sma50.iloc[-1] - 1) * 100) if not pd.isna(sma50.iloc[-1]) else None,
        "dist_sma200_pct": float((latest / sma200.iloc[-1] - 1) * 100) if not pd.isna(sma200.iloc[-1]) else None,
        "vol20_ann": float(rolling_vol.iloc[-1]) if not pd.isna(rolling_vol.iloc[-1]) else None,
        "mom_1m_pct": float((prices.iloc[-1] / prices.iloc[-22] - 1) * 100) if len(prices) >= 22 else None,
        "mom_3m_pct": float((prices.iloc[-1] / prices.iloc[-66] - 1) * 100) if len(prices) >= 66 else None,
        "mom_1y_pct": float((prices.iloc[-1] / prices.iloc[-252] - 1) * 100) if len(prices) >= 252 else None,
    }
    return out


def get_dataset(family: str = "ALL", group: str = "ALL", period: str = "2y", include_reference: bool = False):
    rows = get_rows(family=family, group=group, include_reference=include_reference)
    ensure_min_assets(rows, 1)
    symbols = [r["symbol"] for r in rows]
    prices, diag = fetch_prices(symbols, period=period)
    if prices.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No valid aligned price data could be built after cleaning, forward fill, and equal-length alignment.",
                "diagnostics": diag,
                "family": family,
                "group": group,
                "period": period,
            },
        )
    rows = [r for r in rows if r["symbol"] in prices.columns]
    if not rows:
        raise HTTPException(status_code=400, detail={"message": "No selected assets survived price cleaning.", "diagnostics": diag})
    returns = compute_returns(prices)
    if returns.empty:
        raise HTTPException(status_code=400, detail={"message": "No valid return series available after cleaning.", "diagnostics": diag})
    return rows, prices, returns


@app.get("/api/data-diagnostics")
def api_data_diagnostics(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y"),
    include_reference: bool = Query(False)
):
    rows = get_rows(family=family, group=group, include_reference=include_reference)
    symbols = [r["symbol"] for r in rows]
    prices, diag = fetch_prices(symbols, period=period)
    return JSONResponse(content={
        "family": family,
        "group": group,
        "period": period,
        "selected_count": len(rows),
        "selected_symbols": symbols,
        "diagnostics": diag,
        "aligned_shape": [int(prices.shape[0]), int(prices.shape[1])] if not prices.empty else [0, 0]
    })

@app.get("/")
def root():
    return FileResponse(INDEX_FILE)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/universe")
def api_universe():
    return JSONResponse(content=universe_metadata())


@app.get("/api/snapshot")
def api_snapshot():
    items = flatten_universe(load_universe())
    rows = []
    for x in items:
        included = x["family"] != "World Indices"
        rows.append({
            "family": x["family"],
            "group": x["group"],
            "group_color": x["group_color"],
            "symbol": x["symbol"],
            "label": x["label"],
            "current_weight": round(x["target"] * 100.0, 4),
            "target": x["target"],
            "included": included,
            "status": "Included" if included else "Reference"
        })
    return JSONResponse(content={"rows": rows})


@app.get("/api/overview")
def api_overview(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y")
):
    rows, prices, returns = get_dataset(family=family, group=group, period=period)
    ensure_min_assets(rows, 2 if len(rows) > 1 else 1)
    port, weights = make_portfolio(returns, rows, method="current")
    port_metrics = annualized_metrics(port)
    norm = normalized_prices(prices)
    norm["Portfolio"] = normalized_prices((1 + port).cumprod().to_frame("Portfolio"))["Portfolio"]
    kpis = {
        "assets": len(prices.columns),
        "observations": int(len(prices)),
        "start_date": prices.index.min().strftime("%Y-%m-%d"),
        "end_date": prices.index.max().strftime("%Y-%m-%d"),
        "ann_return_pct": round(port_metrics["ann_return"] * 100, 2),
        "ann_vol_pct": round(port_metrics["ann_vol"] * 100, 2),
        "sharpe": round(port_metrics["sharpe"], 3),
        "max_drawdown_pct": round(port_metrics["max_drawdown"] * 100, 2),
    }
    return JSONResponse(content={
        "kpis": kpis,
        "weights": weights,
        "normalized_chart": chart_records(norm),
        "aligned_symbols": list(prices.columns),
    })


@app.get("/api/technical")
def api_technical(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y")
):
    rows, prices, _ = get_dataset(family=family, group=group, period=period)
    out = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in prices.columns:
        info = technical_indicators(prices[sym])
        out.append({
            "family": row_map[sym]["family"],
            "group": row_map[sym]["group"],
            "symbol": sym,
            "label": row_map[sym]["label"],
            **info
        })
    return JSONResponse(content={"rows": out})


@app.get("/api/analytics")
def api_analytics(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y")
):
    rows, prices, returns = get_dataset(family=family, group=group, period=period)
    port, weights = make_portfolio(returns, rows, method="current")
    corr = returns.corr().round(4).fillna(0)
    metrics = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in returns.columns:
        m = annualized_metrics(returns[sym])
        metrics.append({
            "family": row_map[sym]["family"],
            "group": row_map[sym]["group"],
            "symbol": sym,
            "label": row_map[sym]["label"],
            "ann_return_pct": round(m["ann_return"] * 100, 2),
            "ann_vol_pct": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
            "max_drawdown_pct": round(m["max_drawdown"] * 100, 2)
        })
    port_norm = normalized_prices((1 + port).cumprod().to_frame("Portfolio"))
    return JSONResponse(content={
        "metrics": metrics,
        "correlation_columns": list(corr.columns),
        "correlation_matrix": corr.values.tolist(),
        "portfolio_chart": chart_records(port_norm),
        "weights": weights,
    })


@app.get("/api/risk-dashboard")
def api_risk_dashboard(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y")
):
    rows, prices, returns = get_dataset(family=family, group=group, period=period)
    ensure_min_assets(rows, 2)
    port, weights = make_portfolio(returns, rows, method="current")
    cov = returns.cov() * TRADING_DAYS
    w = np.array([weights[s] for s in returns.columns], dtype=float)
    port_vol = float(np.sqrt(w @ cov.values @ w))
    mrc = cov.values @ w / port_vol if port_vol > 0 else np.zeros_like(w)
    rc = w * mrc
    rc_pct = rc / rc.sum() if rc.sum() > 0 else np.zeros_like(rc)
    hist_var_95 = float(np.quantile(port, 0.05))
    hist_cvar_95 = float(port[port <= hist_var_95].mean()) if (port <= hist_var_95).any() else hist_var_95
    drawdown = (1 + port).cumprod() / (1 + port).cumprod().cummax() - 1
    contribution_rows = []
    row_map = {r["symbol"]: r for r in rows}
    for i, sym in enumerate(returns.columns):
        contribution_rows.append({
            "family": row_map[sym]["family"],
            "group": row_map[sym]["group"],
            "symbol": sym,
            "label": row_map[sym]["label"],
            "weight_pct": round(weights[sym] * 100, 2),
            "risk_contribution_pct": round(float(rc_pct[i]) * 100, 2),
        })
    return JSONResponse(content={
        "summary": {
            "portfolio_vol_pct": round(port_vol * 100, 2),
            "hist_var_95_pct": round(hist_var_95 * 100, 2),
            "hist_cvar_95_pct": round(hist_cvar_95 * 100, 2),
            "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        },
        "risk_contributions": contribution_rows,
        "drawdown_chart": series_records(drawdown, "Drawdown"),
    })


@app.get("/api/forecast")
def api_forecast(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    period: str = Query("2y"),
    horizon: int = Query(20)
):
    rows, prices, returns = get_dataset(family=family, group=group, period=period)
    port, weights = make_portfolio(returns, rows, method="current")
    port_index = normalized_prices((1 + port).cumprod().to_frame("Portfolio"))["Portfolio"]
    y = port_index.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    coeff = np.polyfit(x, y, deg=1)
    x_future = np.arange(len(y), len(y) + horizon, dtype=float)
    y_future = coeff[0] * x_future + coeff[1]
    future_dates = pd.bdate_range(port_index.index[-1] + pd.Timedelta(days=1), periods=horizon)
    forecast = [{"date": d.strftime("%Y-%m-%d"), "Forecast": float(v)} for d, v in zip(future_dates, y_future)]
    history = [{"date": d.strftime("%Y-%m-%d"), "Portfolio": float(v)} for d, v in port_index.items()]
    return JSONResponse(content={"history": history, "forecast": forecast})


@app.get("/api/construct")
def api_construct(
    family: str = Query("ALL"),
    group: str = Query("ALL"),
    method: str = Query("current"),
    period: str = Query("2y")
):
    rows, prices, returns = get_dataset(family=family, group=group, period=period)
    ensure_min_assets(rows, 2)
    port, weights = make_portfolio(returns, rows, method=method)
    out = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in returns.columns:
        out.append({
            "family": row_map[sym]["family"],
            "group": row_map[sym]["group"],
            "symbol": sym,
            "label": row_map[sym]["label"],
            "weight_pct": round(weights[sym] * 100, 2),
        })
    return JSONResponse(content={
        "message": "Construction universe validated successfully.",
        "method": method,
        "used_symbols": list(returns.columns),
        "weights": out,
    })


@app.get("/api/futures-dashboard")
def api_futures_dashboard(period: str = Query("1y")):
    rows, prices, returns = get_dataset(family="Futures", group="ALL", period=period)
    norm = normalized_prices(prices)
    metrics = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in returns.columns:
        m = annualized_metrics(returns[sym])
        metrics.append({
            "group": row_map[sym]["group"],
            "symbol": sym,
            "label": row_map[sym]["label"],
            "ann_return_pct": round(m["ann_return"] * 100, 2),
            "ann_vol_pct": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
        })
    return JSONResponse(content={"chart": chart_records(norm), "metrics": metrics})


@app.get("/api/major-world-indices")
def api_major_world_indices(period: str = Query("1y")):
    rows, prices, returns = get_dataset(family="World Indices", group="ALL", period=period, include_reference=True)
    norm = normalized_prices(prices)
    metrics = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in returns.columns:
        m = annualized_metrics(returns[sym])
        metrics.append({
            "symbol": sym,
            "label": row_map[sym]["label"],
            "group": row_map[sym]["group"],
            "ann_return_pct": round(m["ann_return"] * 100, 2),
            "ann_vol_pct": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
        })
    return JSONResponse(content={"chart": chart_records(norm), "metrics": metrics})


@app.get("/api/msci-em-analysis")
def api_msci_em_analysis(period: str = Query("2y")):
    core_rows = get_rows(family="ETF", group="MSCI Emerging Markets - Core", include_reference=False)
    tur_rows = get_rows(family="ETF", group="Country ETF - Turkey Equity", include_reference=False)
    bond_rows = get_rows(family="ETF", group="Credit & EM Bond ETF", include_reference=False)
    rows = core_rows + tur_rows + bond_rows
    symbols = [r["symbol"] for r in rows]
    prices = fetch_prices(symbols, period=period)
    if prices.empty:
        raise HTTPException(status_code=400, detail="No valid aligned MSCI EM / Turkey dataset could be built.")
    returns = compute_returns(prices)
    norm = normalized_prices(prices)
    metrics = []
    row_map = {r["symbol"]: r for r in rows}
    for sym in returns.columns:
        m = annualized_metrics(returns[sym])
        block = "Turkey Equity" if sym == "TUR" else ("EM Bonds" if row_map[sym]["group"] == "Credit & EM Bond ETF" else "MSCI EM Core")
        metrics.append({
            "block": block,
            "symbol": sym,
            "label": row_map[sym]["label"],
            "ann_return_pct": round(m["ann_return"] * 100, 2),
            "ann_vol_pct": round(m["ann_vol"] * 100, 2),
            "sharpe": round(m["sharpe"], 3),
            "max_drawdown_pct": round(m["max_drawdown"] * 100, 2),
        })
    return JSONResponse(content={"chart": chart_records(norm), "metrics": metrics})