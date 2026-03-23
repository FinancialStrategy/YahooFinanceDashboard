from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LinearRegression

try:
    import cvxpy as cp
except Exception:
    cp = None

try:
    import quantstats as qs
except Exception:
    qs = None

from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt import objective_functions
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt.cla import CLA
try:
    from pypfopt import black_litterman
except Exception:
    black_litterman = None

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
INDEX_FILE = PUBLIC_DIR / "index.html"
if not INDEX_FILE.exists():
    INDEX_FILE = BASE_DIR / "index.html"
if not INDEX_FILE.exists():
    alt = BASE_DIR / "index_FINAL_v2_FEASIBILITY.html"
    if alt.exists():
        INDEX_FILE = alt
UNIVERSE_FILE = BASE_DIR / "universe.json"
if not UNIVERSE_FILE.exists():
    alt_universe = BASE_DIR / "universe_repaired_assetclass.json"
    if alt_universe.exists():
        UNIVERSE_FILE = alt_universe

app = FastAPI(title="Institutional Quant Platform")

RISK_FREE_RATE = 0.035
BENCHMARK_SYMBOL = "^GSPC"
BENCHMARK_FALLBACKS = ["^GSPC", "SPY", "IVV", "VOO"]
TRADING_DAYS = 252

MAJOR_WORLD_INDICES = [
    {"symbol": "^GSPC", "label": "S&P 500", "group": "North America"},
    {"symbol": "^DJI", "label": "Dow Jones Industrial Average", "group": "North America"},
    {"symbol": "^IXIC", "label": "NASDAQ Composite", "group": "North America"},
    {"symbol": "^RUT", "label": "Russell 2000", "group": "North America"},
    {"symbol": "^FTSE", "label": "FTSE 100", "group": "Europe"},
    {"symbol": "^GDAXI", "label": "DAX", "group": "Europe"},
    {"symbol": "^FCHI", "label": "CAC 40", "group": "Europe"},
    {"symbol": "^STOXX50E", "label": "EURO STOXX 50", "group": "Europe"},
    {"symbol": "^N225", "label": "Nikkei 225", "group": "Asia Pacific"},
    {"symbol": "^HSI", "label": "Hang Seng Index", "group": "Asia Pacific"},
    {"symbol": "000001.SS", "label": "Shanghai Composite", "group": "Asia Pacific"},
    {"symbol": "^AXJO", "label": "S&P/ASX 200", "group": "Asia Pacific"},
    {"symbol": "^BSESN", "label": "BSE SENSEX", "group": "Asia Pacific"},
    {"symbol": "^BVSP", "label": "Ibovespa", "group": "Latin America"},
    {"symbol": "^MXX", "label": "IPC Mexico", "group": "Latin America"},
    {"symbol": "XU100.IS", "label": "BIST 100", "group": "Türkiye"}
]
EM_BENCHMARK_SYMBOLS = ["EEM", "IEMG", "VWO", "SPEM"]
TURKEY_EQUITY_SYMBOLS = ["TUR"]
TURKEY_BOND_PROXY_SYMBOLS = ["EMB", "EMLC", "EMHY", "EMBX", "LEMB"]

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
        family = group.get("family", "Other")
        for item in group["items"]:
            rows.append({
                "family": family,
                "group": group["name"],
                "group_color": group["color"],
                "symbol": item["symbol"],
                "label": item["label"],
                "target": item["target"]
            })
    return rows


def symbols_for_group(group_filter, family_filter="ALL"):
    universe = load_universe()
    items = flatten_universe(universe)

    rows = items
    if family_filter and family_filter != "ALL":
        rows = [x for x in rows if x.get("family") == family_filter]

    if not group_filter or group_filter == "ALL":
        return [x["symbol"] for x in rows]

    return [x["symbol"] for x in rows if x["group"] == group_filter]


def label_map():
    universe = load_universe()
    items = flatten_universe(universe)
    return {x["symbol"]: x["label"] for x in items}


def symbol_group_map():
    universe = load_universe()
    mapping = {}
    for group in universe.get("groups", []):
        gname = group.get("name", "Unknown")
        for item in group.get("items", []):
            mapping[item.get("symbol")] = gname
    return mapping

def symbol_family_map():
    universe = load_universe()
    mapping = {}
    for group in universe.get("groups", []):
        family = group.get("family", "Other")
        for item in group.get("items", []):
            mapping[item.get("symbol")] = family
    return mapping


def normalize_weight_dict(weight_dict, symbols):
    """
    Normalize a dict of weights to sum to 1.0 and align to `symbols`.
    Accepts inputs either in fractions (0-1) or percents (0-100).
    """
    if not isinstance(weight_dict, dict):
        weight_dict = {}
    vals = {s: float(weight_dict.get(s, 0.0)) for s in symbols}
    mx = max(vals.values()) if vals else 0.0
    if mx > 1.5:
        vals = {k: v / 100.0 for k, v in vals.items()}
    total = sum(max(v, 0.0) for v in vals.values())
    if total <= 0:
        # equal-weight fallback
        eq = 1.0 / max(len(symbols), 1)
        return {s: eq for s in symbols}
    vals = {k: max(v, 0.0) / total for k, v in vals.items()}
    return vals


def group_weight_breakdown(weight_series: pd.Series, group_map: dict):
    rows = []
    if weight_series is None or weight_series.empty:
        return rows
    tmp = weight_series.copy()
    tmp.index = [group_map.get(s, "Unknown") for s in tmp.index]
    by_group = tmp.groupby(tmp.index).sum().sort_values(ascending=False)
    for g, w in by_group.items():
        rows.append({"group": str(g), "weight": float(w)})
    return rows


def align_weights_for_risk(returns: pd.DataFrame, weight_series: pd.Series, min_abs=1e-10):
    """
    Align weights to returns columns, drop flat/empty series, renormalize.
    Returns aligned weight series and aligned returns dataframe.
    """
    if returns is None or returns.empty:
        raise RuntimeError("Not enough return data to compute risk contributions.")
    if weight_series is None or weight_series.empty:
        raise RuntimeError("Weights sum to zero after alignment.")

    # Keep only columns with finite variance
    r = returns.copy().dropna(how="all")
    var = r.var(skipna=True)
    keep = var[var > 0].index.tolist()
    if len(keep) < 2:
        raise RuntimeError("Too many flat assets after cleaning.")

    r = r[keep].dropna()

    w = weight_series.reindex(r.columns).fillna(0.0)
    w = w[w.abs() > min_abs]
    if w.empty or float(w.sum()) == 0.0:
        raise RuntimeError("Weights sum to zero after alignment.")
    w = w / float(w.sum())

    # align returns to the remaining weight set
    r = r[w.index].dropna()
    if r.shape[1] < 2 or len(r) < 60:
        raise RuntimeError("Not enough return data to compute risk contributions.")
    return r, w


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
    return float(log_returns.std() * np.sqrt(TRADING_DAYS) * 100.0)


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


def download_price_matrix(symbols, period="2y", interval="1d"):
    symbols = list(dict.fromkeys([s for s in symbols if s]))

    raw = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker"
    )

    if raw.empty:
        raise RuntimeError("Yahoo Finance returned empty dataset")

    close_df = pd.DataFrame()
    total_symbols = len(symbols)

    for symbol in symbols:
        try:
            sdf = extract_symbol_frame(raw, symbol, total_symbols).copy()
            sdf = sdf.dropna(how="all")
            close_col = "Close" if "Close" in sdf.columns else ("Adj Close" if "Adj Close" in sdf.columns else None)
            if close_col:
                close_df[symbol] = pd.to_numeric(sdf[close_col], errors="coerce")
        except Exception:
            continue

    if close_df.empty:
        raise RuntimeError("No valid close-price matrix could be built")

    close_df = close_df.sort_index()
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    close_df = close_df.ffill()

    # Keep columns with reasonable coverage
    coverage = close_df.notna().sum() / max(len(close_df), 1)
    keep_cols = coverage[coverage >= 0.70].index.tolist()
    close_df = close_df[keep_cols]
    if close_df.empty:
        raise RuntimeError("No symbols survived the coverage filter")

    # Align to equal length using the latest first valid date across columns
    first_valid = [close_df[c].first_valid_index() for c in close_df.columns if close_df[c].first_valid_index() is not None]
    if not first_valid:
        raise RuntimeError("No valid start dates were found in the panel")
    common_start = max(first_valid)
    close_df = close_df.loc[common_start:].ffill().dropna(axis=0, how="any")

    min_obs = max(60, int(0.35 * len(close_df))) if len(close_df) else 60
    close_df = close_df.dropna(axis=1, thresh=min_obs)

    if close_df.shape[1] < 1 or len(close_df) < 30:
        raise RuntimeError("Not enough valid symbols after cleaning and equal-length alignment")

    return close_df


def build_snapshot():
    universe = load_universe()
    items = flatten_universe(universe)
    symbols = [x["symbol"] for x in items]

    df = yf.download(
        tickers=symbols,
        period="2y",
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
            close_col = "Close" if "Close" in sdf.columns else ("Adj Close" if "Adj Close" in sdf.columns else None)
            if sdf.empty or not close_col:
                continue

            close = pd.to_numeric(sdf[close_col], errors="coerce").dropna()
            volume = pd.to_numeric(sdf["Volume"], errors="coerce").dropna() if "Volume" in sdf.columns else pd.Series(dtype=float)

            if len(close) < 2:
                continue

            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            daily_change_pct = ((last_price / prev_close) - 1.0) * 100.0 if prev_close != 0 else None
            target_pct = float(item["target"]) * 100.0 if float(item["target"]) <= 1.5 else float(item["target"])

            row = {
                "family": item.get("family", "Other"),
                "group": item["group"],
                "group_color": item["group_color"],
                "symbol": item["symbol"],
                "label": item["label"],
                "target": target_pct,
                "price": last_price,
                "prev_close": prev_close,
                "daily_change_pct": daily_change_pct,
                "ret_1d_pct": daily_change_pct,
                "ret_1w_pct": safe_pct_return(close, 5),
                "ret_1m_pct": safe_pct_return(close, 21),
                "ret_3m_pct": safe_pct_return(close, 63),
                "ytd_pct": ytd_return(close),
                "ret_6m_pct": safe_pct_return(close, 126),
                "ret_1y_pct": safe_pct_return(close, 252),
                "vol_30d_pct": annualized_volatility(close, 30),
                "avg_volume_20d": avg_volume(volume, 20),
                "sparkline": close.tail(20).round(6).tolist()
            }
            result_rows.append(row)
        except Exception:
            continue

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "count": len(result_rows),
        "rows": result_rows
    }


def build_market_snapshot(items, period="2y"):
    symbols = [x["symbol"] for x in items]
    df = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker"
    )
    if df.empty:
        raise RuntimeError("Yahoo Finance returned empty dataset for market snapshot")
    result_rows = []
    for item in items:
        symbol = item["symbol"]
        try:
            sdf = extract_symbol_frame(df, symbol, len(symbols)).copy().dropna(how="all")
            if sdf.empty or "Close" not in sdf.columns:
                continue
            close = sdf["Close"].dropna()
            if len(close) < 2:
                continue
            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            row = {
                "group": item.get("group", "Market"),
                "symbol": symbol,
                "label": item.get("label", symbol),
                "price": last_price,
                "ret_1d_pct": ((last_price / prev_close) - 1.0) * 100.0 if prev_close != 0 else None,
                "ret_1w_pct": safe_pct_return(close, 5),
                "ret_1m_pct": safe_pct_return(close, 21),
                "ret_3m_pct": safe_pct_return(close, 63),
                "ytd_pct": ytd_return(close),
                "ret_1y_pct": safe_pct_return(close, 252),
                "vol_30d_pct": annualized_volatility(close, 30)
            }
            result_rows.append(row)
        except Exception:
            continue
    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "count": len(result_rows),
        "rows": result_rows
    }





def build_asset_class_snapshot(group_filter="ALL", family_filter="ALL", period="2y"):
    universe = load_universe()
    items = flatten_universe(universe)

    selected = items
    if family_filter and family_filter != "ALL":
        selected = [x for x in selected if x.get("family", "Other") == family_filter]
    if group_filter and group_filter != "ALL":
        selected = [x for x in selected if x.get("group") == group_filter]

    if not selected:
        raise RuntimeError("No instruments matched the selected asset class.")

    symbols = [x["symbol"] for x in selected]
    labels = {x["symbol"]: x.get("label", x["symbol"]) for x in selected}

    df = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker"
    )
    if df.empty:
        raise RuntimeError("Yahoo Finance returned empty dataset for Asset Class Analysis")

    price_frames = []
    rows = []
    total_symbols = len(symbols)

    for item in selected:
        symbol = item["symbol"]
        try:
            sdf = extract_symbol_frame(df, symbol, total_symbols).copy().dropna(how="all")
            close_col = "Close" if "Close" in sdf.columns else ("Adj Close" if "Adj Close" in sdf.columns else None)
            if sdf.empty or not close_col:
                continue
            close = pd.to_numeric(sdf[close_col], errors="coerce").dropna().sort_index()
            if len(close) < 2:
                continue

            price_frames.append(close.rename(symbol))

            last_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            rows.append({
                "family": item.get("family", "Other"),
                "group": item.get("group", "Unknown"),
                "symbol": symbol,
                "label": item.get("label", symbol),
                "price": last_price,
                "ret_1d_pct": ((last_price / prev_close) - 1.0) * 100.0 if prev_close != 0 else None,
                "ret_1w_pct": safe_pct_return(close, 5),
                "ret_1m_pct": safe_pct_return(close, 21),
                "ret_3m_pct": safe_pct_return(close, 63),
                "ytd_pct": ytd_return(close),
                "ret_1y_pct": safe_pct_return(close, 252),
                "vol_30d_pct": annualized_volatility(close, 30),
            })
        except Exception:
            continue

    eq_curve = []
    relative_table = []
    best_worst = {"best": None, "worst": None}

    if price_frames:
        panel = pd.concat(price_frames, axis=1).sort_index().ffill().dropna(how="all")
        if not panel.empty:
            first_valid = [panel[c].first_valid_index() for c in panel.columns if panel[c].first_valid_index() is not None]
            if first_valid:
                common_start = max(first_valid)
                panel = panel.loc[common_start:].ffill().dropna(axis=0, how="any")

        if not panel.empty and panel.shape[1] >= 1 and len(panel) >= 2:
            norm_panel = panel / panel.iloc[0] * 100.0
            eq_series = norm_panel.mean(axis=1)
            eq_curve = [{"date": pd.to_datetime(i).strftime("%Y-%m-%d"), "value": float(v)} for i, v in eq_series.items()]

            last_vals = norm_panel.iloc[-1].sort_values(ascending=False)
            relative_table = []
            for sym, val in last_vals.items():
                row_meta = next((r for r in rows if r["symbol"] == sym), None)
                relative_table.append({
                    "symbol": sym,
                    "label": labels.get(sym, sym),
                    "relative_strength_indexed": float(val),
                    "family": row_meta.get("family", "Other") if row_meta else "Other",
                    "group": row_meta.get("group", "Unknown") if row_meta else "Unknown",
                    "ytd_pct": row_meta.get("ytd_pct") if row_meta else None,
                    "ret_1y_pct": row_meta.get("ret_1y_pct") if row_meta else None,
                    "vol_30d_pct": row_meta.get("vol_30d_pct") if row_meta else None,
                })

            if relative_table:
                best_worst["best"] = relative_table[0]
                best_worst["worst"] = relative_table[-1]

    rows = sorted(rows, key=lambda x: (x["ytd_pct"] if x["ytd_pct"] is not None else -999), reverse=True)

    return {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "selected_group": group_filter,
        "selected_family": family_filter,
        "selected_count": len(selected),
        "count": len(rows),
        "rows": rows,
        "equal_weight_cumulative": eq_curve,
        "relative_ranking": relative_table,
        "leaders_laggards": best_worst
    }



def clean_weight_dict(weights: dict):
    out = {}
    for k, v in weights.items():
        fv = float(v)
        if abs(fv) > 1e-8:
            out[k] = round(fv, 6)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def build_frontier_points(mu, cov_matrix, weight_bounds=(0, 1)):
    points = []
    try:
        ef_min = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)
        ef_min.min_volatility()
        min_ret, min_vol, _ = ef_min.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
    except Exception:
        return points

    try:
        ef_max = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)
        ef_max.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        max_ret, _, _ = ef_max.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
    except Exception:
        max_ret = min_ret

    if max_ret < min_ret:
        min_ret, max_ret = max_ret, min_ret

    grid = np.linspace(float(min_ret), float(max_ret), 20)
    for tr in grid:
        try:
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)
            ef.efficient_return(float(tr))
            ret, vol, shp = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
            points.append({
                "expected_return": float(ret),
                "volatility": float(vol),
                "sharpe": float(shp)
            })
        except Exception:
            continue

    if not points:
        try:
            points.append({
                "expected_return": float(min_ret),
                "volatility": float(min_vol),
                "sharpe": float((min_ret - RISK_FREE_RATE) / min_vol) if min_vol > 0 else 0.0
            })
        except Exception:
            pass
    return points


def _normalize_group_targets(group_targets):
    if isinstance(group_targets, dict) and len(group_targets) > 0:
        gt = {str(k): float(v) for k, v in group_targets.items()}
        mx = max(gt.values()) if gt else 0.0
        if mx > 1.5:
            gt = {k: v / 100.0 for k, v in gt.items()}
        tot = sum(max(v, 0.0) for v in gt.values())
        if tot > 0:
            return {k: max(v, 0.0) / tot for k, v in gt.items()}
    return None


def _normalize_group_bounds(group_bounds):
    out = {}
    if not isinstance(group_bounds, dict):
        return out
    for g, cfg in group_bounds.items():
        if not isinstance(cfg, dict):
            continue
        mn = float(cfg.get("min", 0.0) or 0.0)
        mx = float(cfg.get("max", 1.0) or 1.0)
        if mx > 1.5:
            mx /= 100.0
        if mn > 1.5:
            mn /= 100.0
        mn = max(0.0, mn)
        mx = min(1.0, mx)
        if mn <= mx:
            out[str(g)] = {"min": mn, "max": mx}
    return out


def _group_to_symbol_view(group_views, group_map, symbols):
    if not isinstance(group_views, dict):
        return {}
    symbol_views = {}
    for g, view in group_views.items():
        try:
            val = float(view)
        except Exception:
            continue
        members = [s for s in symbols if group_map.get(s, "Unknown") == g]
        if not members:
            continue
        per_symbol = val / len(members)
        for s in members:
            symbol_views[s] = symbol_views.get(s, 0.0) + per_symbol
    return symbol_views


def _create_all_group_bounds(symbols, group_map, min_weight, max_weight, group_bounds=None, group_targets=None):
    """
    Create bounds for ALL groups present in the universe.
    For groups with explicit user bounds, use those.
    For groups without explicit bounds, use structural capacity (n * min_weight to n * max_weight).
    This ensures sum(min) <= 1 <= sum(max) for the entire universe.
    """
    present_groups = {}
    for s in symbols:
        g = group_map.get(s, "Unknown")
        present_groups.setdefault(g, []).append(s)

    # Get user-defined bounds
    user_bounds = {}
    if isinstance(group_bounds, dict):
        for g, cfg in group_bounds.items():
            if isinstance(cfg, dict):
                user_bounds[str(g)] = {
                    "min": float(cfg.get("min", 0.0)),
                    "max": float(cfg.get("max", 1.0))
                }

    # Get target-defined bands
    target_bounds = {}
    if isinstance(group_targets, dict):
        for g, tgt in group_targets.items():
            g_str = str(g)
            tgt_val = float(tgt)
            # Apply a ±5% band around target (soft constraint)
            target_bounds[g_str] = {
                "min": max(0.0, tgt_val - 0.05),
                "max": min(1.0, tgt_val + 0.05)
            }

    out = {}
    for g, members in present_groups.items():
        # Structural capacity
        cap_min = max(0.0, len(members) * float(min_weight))
        cap_max = min(1.0, len(members) * float(max_weight))

        # Determine final bounds: user bounds have highest priority, then target bands, then structural
        if g in user_bounds:
            mn = user_bounds[g]["min"]
            mx = user_bounds[g]["max"]
            # Intersect with structural capacity
            mn = max(mn, cap_min)
            mx = min(mx, cap_max)
        elif g in target_bounds:
            mn = target_bounds[g]["min"]
            mx = target_bounds[g]["max"]
            # Intersect with structural capacity
            mn = max(mn, cap_min)
            mx = min(mx, cap_max)
        else:
            # No explicit class-level constraint: keep the group structurally unconstrained
            # and let per-asset min/max control feasibility.
            mn = 0.0
            mx = 1.0

        if mn > mx + 1e-12:
            # If bounds conflict, relax to structural range
            mn = cap_min
            mx = cap_max

        out[g] = {"min": mn, "max": mx}

    return out


def _check_global_bounds_feasibility(bounds):
    """
    Check that sum(min) <= 1 <= sum(max) for all bounds.
    """
    if not bounds:
        return True, "No bounds to check"
    
    sum_min = sum(float(v.get("min", 0.0)) for v in bounds.values())
    sum_max = sum(float(v.get("max", 1.0)) for v in bounds.values())
    
    if sum_min > 1.0 + 1e-8:
        return False, f"Total minimum ({sum_min:.2%}) exceeds 100%"
    
    if sum_max < 1.0 - 1e-8:
        return False, f"Total maximum ({sum_max:.2%}) is less than 100%"
    
    return True, f"Feasible: min={sum_min:.2%}, max={sum_max:.2%}"


def _apply_group_constraints(ef, symbols, group_map, group_bounds=None):
    if cp is None:
        return ef
    for g, bounds in (group_bounds or {}).items():
        idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
        if not idxs:
            continue
        mn = float(bounds.get("min", 0.0))
        mx = float(bounds.get("max", 1.0))
        ef.add_constraint(lambda w, idxs=idxs, mn=mn: cp.sum(w[idxs]) >= mn)
        ef.add_constraint(lambda w, idxs=idxs, mx=mx: cp.sum(w[idxs]) <= mx)
    return ef


def _build_black_litterman_mu(prices, cov_matrix, group_map, symbols, group_views=None, view_confidences=None):
    base_mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    if black_litterman is None or not group_views:
        return base_mu

    abs_views = _group_to_symbol_view(group_views, group_map, symbols)
    if not abs_views:
        return base_mu

    try:
        market_prior = black_litterman.market_implied_prior_returns(
            market_caps={s: 1.0 for s in symbols},
            risk_aversion=2.5,
            cov_matrix=cov_matrix
        )
    except Exception:
        market_prior = base_mu

    conf_list = []
    for s in abs_views.keys():
        g = group_map.get(s, "Unknown")
        c = None if not isinstance(view_confidences, dict) else view_confidences.get(g)
        try:
            c = float(c)
        except Exception:
            c = 0.5
        if c > 1:
            c = c / 100.0
        conf_list.append(min(max(c, 0.05), 0.99))

    try:
        bl = black_litterman.BlackLittermanModel(
            cov_matrix,
            pi=market_prior,
            absolute_views=abs_views,
            view_confidences=np.array(conf_list, dtype=float)
        )
        posterior = bl.bl_returns()
        return posterior.reindex(base_mu.index).fillna(base_mu)
    except Exception:
        return base_mu


def optimize_with_strategy(prices, strategy, target_volatility, target_return, prior_weights=None, group_targets=None, min_weight=0.0, max_weight=1.0, group_map=None, group_bounds=None, group_views=None, view_confidences=None):
    cov_matrix = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    symbols = list(prices.columns)
    if group_map is None:
        group_map = symbol_group_map()

    if strategy.startswith("black_litterman"):
        mu = _build_black_litterman_mu(prices, cov_matrix, group_map, symbols, group_views=group_views, view_confidences=view_confidences)
    else:
        mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    rets = expected_returns.returns_from_prices(prices)

    if prior_weights is not None:
        prior_weights = normalize_weight_dict(prior_weights, symbols)

    group_targets = _normalize_group_targets(group_targets)
    group_bounds = _normalize_group_bounds(group_bounds)

    # Create bounds for ALL groups present
    all_group_bounds = _create_all_group_bounds(
        symbols, group_map, min_weight, max_weight, 
        group_bounds=group_bounds, group_targets=group_targets
    )
    
    # Check feasibility
    feasible, msg = _check_global_bounds_feasibility(all_group_bounds)
    if not feasible:
        # Try to auto-fix: if sum_min > 1, reduce mins proportionally
        sum_min = sum(v["min"] for v in all_group_bounds.values())
        if sum_min > 1.0 + 1e-8:
            scale = 0.99 / sum_min
            for g in all_group_bounds:
                all_group_bounds[g]["min"] = all_group_bounds[g]["min"] * scale
        # If sum_max < 1, increase maxs for unconstrained groups
        sum_max = sum(v["max"] for v in all_group_bounds.values())
        if sum_max < 1.0 - 1e-8:
            deficit = 1.0 - sum_max
            # Find groups that can absorb extra weight
            adjustable = [g for g in all_group_bounds if g not in group_bounds and g not in group_targets]
            if adjustable and deficit > 0:
                per_group = deficit / len(adjustable)
                for g in adjustable:
                    all_group_bounds[g]["max"] = min(1.0, all_group_bounds[g]["max"] + per_group)

    def _run_strategy_once(bounds_for_run):
        if strategy == "max_sharpe":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "min_volatility":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.min_volatility()
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "max_quadratic_utility":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.max_quadratic_utility(risk_aversion=1.0)
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "efficient_risk":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.efficient_risk(target_volatility=float(target_volatility))
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "efficient_return":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.efficient_return(target_return=float(target_return))
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "black_litterman_max_sharpe":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "black_litterman_min_volatility":
            ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            _apply_group_constraints(ef, symbols, group_map, group_bounds=bounds_for_run)
            ef.min_volatility()
            return ef.clean_weights(), ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "hrp":
            hrp = HRPOpt(rets)
            return hrp.optimize(), hrp.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "cla_max_sharpe":
            cla = CLA(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            cla.max_sharpe()
            return cla.clean_weights(), cla.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "cla_min_volatility":
            cla = CLA(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
            cla.min_volatility()
            return cla.clean_weights(), cla.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

        elif strategy == "prior_only":
            if prior_weights is None:
                raise RuntimeError("Prior weights were not supplied.")
            wvec = np.array([prior_weights[s] for s in symbols], dtype=float)
            port_ret = float(np.asarray(mu.values @ wvec).squeeze())
            port_vol = float(np.sqrt(np.asarray(wvec.T @ cov_matrix.values @ wvec).squeeze()))
            sharpe = float((port_ret - RISK_FREE_RATE) / port_vol) if port_vol > 0 else 0.0
            weights = {s: float(round(prior_weights[s], 6)) for s in symbols if prior_weights[s] > 1e-8}
            perf = (port_ret, port_vol, sharpe)
            return weights, perf

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    try:
        weights, perf = _run_strategy_once(all_group_bounds)
    except Exception as e:
        # Fallback: try with no group constraints
        weights, perf = _run_strategy_once({})

    frontier = build_frontier_points(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
    return mu, cov_matrix, clean_weight_dict(weights), perf, frontier


def resolve_symbols(payload):
    symbols = payload.get("symbols", [])
    family_filter = payload.get("family_filter", "ALL")
    group_filter = payload.get("group_filter", "ALL")

    if not symbols:
        symbols = symbols_for_group(group_filter, family_filter)

    allowed = set(symbols_for_group(group_filter, family_filter))
    if allowed:
        symbols = [s for s in symbols if s in allowed]

    symbols = [s for s in symbols if s != BENCHMARK_SYMBOL]
    symbols = list(dict.fromkeys(symbols))
    return symbols


def resolve_benchmark_price_series(price_df):
    """
    Return the first usable benchmark series from the fallback chain.
    """
    for sym in BENCHMARK_FALLBACKS:
        if sym in price_df.columns:
            s = pd.to_numeric(price_df[sym], errors="coerce").dropna()
            if not s.empty:
                return s, sym
    return None, None


def build_portfolio_context(payload):
    strategy = payload.get("strategy", "max_sharpe")
    target_volatility = float(payload.get("target_volatility", 0.15))
    target_return = float(payload.get("target_return", 0.10))
    symbols = resolve_symbols(payload)
    if not symbols:
        raise RuntimeError("No instruments matched the selected Asset Class filter.")

    # Prior weights (from UI) and asset-class targets (optional)
    prior_weights_input = payload.get("user_weights") or payload.get("prior_weights") or {}
    group_targets_input = payload.get("group_targets") or {}
    group_bounds_input = payload.get("group_bounds") or {}
    group_views_input = payload.get("group_views") or {}
    view_confidences_input = payload.get("view_confidences") or {}
    min_weight = float(payload.get("min_weight", 0.0))
    max_weight = float(payload.get("max_weight", 1.0))
    if max_weight <= 0 or max_weight > 1:
        max_weight = 1.0
    if min_weight < 0:
        min_weight = 0.0

    group_map = symbol_group_map()
    prior_weights = normalize_weight_dict(prior_weights_input, symbols)
    # For convenience: if group targets are not provided, derive them from prior weights
    if not group_targets_input:
        tmp = pd.Series(prior_weights)
        tmp.index = [group_map.get(s, "Unknown") for s in tmp.index]
        group_targets_input = tmp.groupby(tmp.index).sum().to_dict()

    min_required_assets = 1 if strategy == "prior_only" else 2
    if len(symbols) < min_required_assets:
        raise RuntimeError(f"The selected asset-class universe contains only {len(symbols)} instrument(s). {'Prior-only mode requires at least 1 instrument.' if strategy == 'prior_only' else 'Portfolio construction requires at least 2 investable instruments after filtering.'}")

    # Auto-relax impossible per-asset weight settings for the current filtered universe.
    # If n * max_weight < 1, the portfolio can never reach 100%.
    # If n * min_weight > 1, the minimums alone already exceed 100%.
    n_assets = len(symbols)
    feasibility_notes = []
    if n_assets > 0 and max_weight * n_assets < 1.0 - 1e-8:
        old_max_weight = max_weight
        max_weight = min(1.0, 1.0 / n_assets + 1e-6)
        feasibility_notes.append(
            f"Per-asset max weight was auto-relaxed from {old_max_weight:.2%} to {max_weight:.2%} because {n_assets} assets cannot sum to 100% otherwise."
        )
    if n_assets > 0 and min_weight * n_assets > 1.0 + 1e-8:
        old_min_weight = min_weight
        min_weight = max(0.0, 1.0 / n_assets - 1e-6)
        feasibility_notes.append(
            f"Per-asset min weight was auto-relaxed from {old_min_weight:.2%} to {min_weight:.2%} because the selected universe would otherwise force total minimum allocation above 100%."
        )
    if min_weight >= max_weight:
        min_weight = 0.0

    labels = label_map()
    benchmark_candidates = []
    for sym in BENCHMARK_FALLBACKS:
        if sym not in benchmark_candidates:
            benchmark_candidates.append(sym)
    price_symbols = list(dict.fromkeys(symbols + benchmark_candidates))
    price_df = download_price_matrix(price_symbols, period="2y", interval="1d")

    benchmark_prices, benchmark_symbol_used = resolve_benchmark_price_series(price_df)
    benchmark_available = benchmark_prices is not None and not benchmark_prices.empty

    prices = price_df.drop(columns=[c for c in benchmark_candidates if c in price_df.columns], errors="ignore")

    if prices.shape[1] < min_required_assets:
        raise RuntimeError(f"Only {prices.shape[1]} instrument(s) remained after price-data cleaning. {'Prior-only mode requires at least 1 instrument.' if strategy == 'prior_only' else 'Portfolio construction requires at least 2 investable instruments after filtering and data cleaning.'}")

    mu, cov_matrix, weights, perf, frontier = optimize_with_strategy(
        prices=prices,
        strategy=strategy,
        target_volatility=target_volatility,
        target_return=target_return,
        prior_weights=prior_weights,
        group_targets=group_targets_input,
        min_weight=min_weight,
        max_weight=max_weight,
        group_map=group_map,
        group_bounds=group_bounds_input,
        group_views=group_views_input,
        view_confidences=view_confidences_input
    )

    returns = prices.pct_change().dropna()
    cov_returns = returns.cov() * TRADING_DAYS

    benchmark_returns = benchmark_prices.pct_change().dropna() if benchmark_available else pd.Series(dtype=float)

    weight_series = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    prior_weight_series = pd.Series(prior_weights).reindex(prices.columns).fillna(0.0)
    portfolio_returns = returns.mul(weight_series, axis=1).sum(axis=1).dropna()

    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1
    ).dropna() if benchmark_available else pd.DataFrame({"portfolio": portfolio_returns}).dropna()

    beta = None
    alpha_ann = None
    tracking_error = None
    information_ratio = None
    benchmark_ann_return = None
    benchmark_ann_vol = None

    if not aligned.empty and aligned["benchmark"].var() != 0:
        beta = float(aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var())
        rf_daily = RISK_FREE_RATE / TRADING_DAYS
        alpha_ann = float(((aligned["portfolio"].mean() - rf_daily) - beta * (aligned["benchmark"].mean() - rf_daily)) * TRADING_DAYS)
        diff = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(diff.std() * np.sqrt(TRADING_DAYS)) if diff.std() == diff.std() else None
        information_ratio = float((diff.mean() * TRADING_DAYS) / tracking_error) if tracking_error and tracking_error != 0 else None
        benchmark_ann_return = float(aligned["benchmark"].mean() * TRADING_DAYS)
        benchmark_ann_vol = float(aligned["benchmark"].std() * np.sqrt(TRADING_DAYS))

    asset_vol = returns.std() * np.sqrt(TRADING_DAYS)
    asset_df = pd.DataFrame({
        "symbol": prices.columns,
        "label": [labels.get(sym, sym) for sym in prices.columns],
        "expected_return": mu.reindex(prices.columns).values,
        "volatility": asset_vol.reindex(prices.columns).values
    }).replace([np.inf, -np.inf], np.nan).dropna()

    asset_df["sharpe_proxy"] = (asset_df["expected_return"] - RISK_FREE_RATE) / asset_df["volatility"]
    top_assets = asset_df.sort_values(
        ["sharpe_proxy", "expected_return"],
        ascending=[False, False]
    ).head(10)

    weight_table = []
    for sym, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        weight_table.append({
            "symbol": sym,
            "label": labels.get(sym, sym),
            "group": group_map.get(sym, "Unknown"),
            "weight": float(w),
            "prior_weight": float(prior_weight_series.get(sym, 0.0))
        })

    corr = returns.corr().fillna(0.0)

    return {
        "labels": labels,
        "strategy": strategy,
        "prices": prices,
        "returns": returns,
        "weights": weights,
        "weight_series": weight_series,
        "cov_matrix": cov_matrix,
        "correlation_matrix": corr,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "aligned": aligned,
        "frontier": frontier,
        "top_assets": top_assets,
        "weight_table": weight_table,
        "prior_weight_series": prior_weight_series,
        "group_map": group_map,
        "group_weights_prior": group_weight_breakdown(prior_weight_series, group_map),
        "group_weights_optimized": group_weight_breakdown(weight_series, group_map),
        "group_targets_used": group_targets_input,
        "group_bounds_used": group_bounds_input,
        "group_views_used": group_views_input,
        "view_confidences_used": view_confidences_input,
        "cov_returns": cov_returns,

        "benchmark_available": benchmark_available,
        "benchmark_symbol_used": benchmark_symbol_used,
        "feasibility_notes": feasibility_notes,
        "metrics": {
            "expected_return": float(perf[0]),
            "volatility": float(perf[1]),
            "sharpe": float(perf[2]),
            "beta_vs_sp500": beta,
            "alpha_vs_sp500": alpha_ann,
            "tracking_error_vs_sp500": tracking_error,
            "information_ratio_vs_sp500": information_ratio,
            "benchmark_return": benchmark_ann_return,
            "benchmark_volatility": benchmark_ann_vol
        }
    }


def series_to_records(series):
    out = []
    for idx, val in series.dropna().items():
        out.append({
            "date": pd.to_datetime(idx).strftime("%Y-%m-%d"),
            "value": float(val)
        })
    return out


def compute_rolling_sharpe(portfolio_returns, window=63):
    rf_daily = RISK_FREE_RATE / TRADING_DAYS
    excess = portfolio_returns - rf_daily
    sharpe = (excess.rolling(window).mean() / excess.rolling(window).std()) * np.sqrt(TRADING_DAYS)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def compute_drawdown(portfolio_returns):
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return cumulative, drawdown


def compute_rolling_beta(aligned_returns, window=63):
    cov = aligned_returns["portfolio"].rolling(window).cov(aligned_returns["benchmark"])
    var = aligned_returns["benchmark"].rolling(window).var()
    beta = cov / var
    return beta.replace([np.inf, -np.inf], np.nan)


def compute_relative_performance(aligned_returns):
    relative = (1 + aligned_returns["portfolio"]).cumprod() / (1 + aligned_returns["benchmark"]).cumprod() - 1.0
    return relative.replace([np.inf, -np.inf], np.nan)


def compute_rolling_information_ratio(aligned_returns, window=63):
    active = aligned_returns["portfolio"] - aligned_returns["benchmark"]
    rolling_mean = active.rolling(window).mean() * TRADING_DAYS
    rolling_te = active.rolling(window).std() * np.sqrt(TRADING_DAYS)
    ir = rolling_mean / rolling_te
    return ir.replace([np.inf, -np.inf], np.nan)


def compute_hmm_regimes(portfolio_returns, n_states=3):
    clean = portfolio_returns.dropna()
    if len(clean) < 80:
        raise RuntimeError("Not enough observations for HMM regime detection")

    X = clean.values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=42
    )
    model.fit(X)
    hidden_states = model.predict(X)

    state_df = pd.DataFrame({
        "date": clean.index,
        "return": clean.values,
        "state": hidden_states
    })

    state_means = state_df.groupby("state")["return"].mean().sort_values()
    rank_map = {}
    ordered_states = state_means.index.tolist()

    if len(ordered_states) == 3:
        rank_map[ordered_states[0]] = "Bear"
        rank_map[ordered_states[1]] = "Neutral"
        rank_map[ordered_states[2]] = "Bull"
    else:
        for i, s in enumerate(ordered_states):
            rank_map[s] = f"State {i+1}"

    state_df["label"] = state_df["state"].map(rank_map)

    return [
        {
            "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "state": int(row["state"]),
            "label": row["label"],
            "return": float(row["return"])
        }
        for _, row in state_df.iterrows()
    ]


def compute_risk_metrics(portfolio_returns, confidence=0.95, horizon=1):
    clean = portfolio_returns.dropna()
    if len(clean) < 50:
        raise RuntimeError("Not enough observations for risk calculations")

    alpha = 1 - confidence

    hist_q = clean.quantile(alpha)
    hist_var = -float(hist_q) * np.sqrt(horizon)
    hist_tail = clean[clean <= hist_q]
    hist_cvar = -float(hist_tail.mean()) * np.sqrt(horizon) if len(hist_tail) > 0 else None

    mu = float(clean.mean()) * horizon
    sigma = float(clean.std()) * np.sqrt(horizon)
    z = norm.ppf(alpha)

    param_var = -(mu + z * sigma)
    param_cvar = -(mu - sigma * (norm.pdf(z) / alpha))

    return {
        "confidence": confidence,
        "horizon_days": horizon,
        "historical_var": hist_var,
        "historical_cvar": hist_cvar,
        "historical_es": hist_cvar,
        "parametric_var": float(param_var),
        "parametric_cvar": float(param_cvar),
        "parametric_es": float(param_cvar)
    }


def compute_risk_contributions(weight_series, cov_matrix, labels):
    cov_aligned = cov_matrix.loc[weight_series.index, weight_series.index]
    w = weight_series.values.reshape(-1, 1)
    sigma = cov_aligned.values

    # Numpy may return a 1x1 array here; convert safely to a Python scalar.
    portfolio_var_arr = w.T @ sigma @ w
    portfolio_var = float(np.asarray(portfolio_var_arr).squeeze())
    if not np.isfinite(portfolio_var) or portfolio_var <= 0:
        return []

    portfolio_vol = float(np.sqrt(portfolio_var))
    marginal_contrib = (sigma @ w) / portfolio_vol
    component_contrib = w * marginal_contrib
    pct_contrib = component_contrib / portfolio_vol

    rows = []
    for i, sym in enumerate(weight_series.index):
        rows.append({
            "symbol": sym,
            "label": labels.get(sym, sym),
            "weight": float(np.asarray(w[i, 0]).squeeze()),
            "marginal_risk_contribution": float(np.asarray(marginal_contrib[i, 0]).squeeze()),
            "component_risk_contribution": float(np.asarray(component_contrib[i, 0]).squeeze()),
            "percent_risk_contribution": float(np.asarray(pct_contrib[i, 0]).squeeze())
        })

    return sorted(rows, key=lambda x: x["percent_risk_contribution"], reverse=True)


def compute_lightweight_forecast(symbol, period="3y", lookback=30, forecast_horizon=20):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        raise RuntimeError("No data returned for forecast")

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"][symbol].dropna()
    else:
        close = df["Close"].dropna()

    if len(close) < max(lookback + 20, 80):
        raise RuntimeError("Not enough observations for forecast")

    values = close.values.astype(float)
    returns = pd.Series(values).pct_change().dropna().values

    if len(returns) < lookback + 10:
        raise RuntimeError("Not enough return observations for forecast")

    X, y = [], []
    for i in range(lookback, len(returns)):
        X.append(returns[i - lookback:i])
        y.append(returns[i])

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    rolling_window = returns[-lookback:].copy()
    predicted_returns = []

    for _ in range(forecast_horizon):
        pred = float(model.predict(rolling_window.reshape(1, -1))[0])
        pred = float(np.clip(pred, -0.08, 0.08))
        predicted_returns.append(pred)
        rolling_window = np.roll(rolling_window, -1)
        rolling_window[-1] = pred

    last_price = float(close.iloc[-1])
    forecast_prices = []
    current_price = last_price
    for r in predicted_returns:
        current_price = current_price * (1 + r)
        forecast_prices.append(current_price)

    last_date = pd.to_datetime(close.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    history = close.tail(120)

    return {
        "symbol": symbol,
        "model": "Lightweight autoregressive forecast",
        "history": [
            {"date": pd.to_datetime(idx).strftime("%Y-%m-%d"), "value": float(val)}
            for idx, val in history.items()
        ],
        "forecast": [
            {"date": dt.strftime("%Y-%m-%d"), "value": float(val)}
            for dt, val in zip(future_dates, forecast_prices)
        ]
    }


# ============================
# Technical Analysis (single instrument)
# ============================

def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan)

def _safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None

def _download_close(symbol: str, period: str = "2y", interval: str = "1d"):
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if df is None or df.empty:
        raise RuntimeError("No market data returned for the selected instrument")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close = df["Close"][symbol].dropna()
        except Exception:
            close = df.xs(symbol, axis=1, level=0)["Close"].dropna()
    else:
        close = df["Close"].dropna()

    if len(close) < 60:
        raise RuntimeError("Not enough price history for technical analysis")
    close = close.sort_index().ffill().dropna()
    return close

def _quant_metrics(returns: pd.Series):
    if qs is None:
        rf_daily = RISK_FREE_RATE / TRADING_DAYS
        excess = returns - rf_daily
        cagr = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
        vol = returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS) if excess.std() and excess.std() == excess.std() else None
        downside = returns[returns < 0]
        sortino = (excess.mean() * TRADING_DAYS) / (downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 5 and downside.std() else None
        cum = (1 + returns).cumprod()
        dd = cum / cum.cummax() - 1
        mdd = dd.min()
        calmar = (cagr / abs(mdd)) if mdd and mdd < 0 else None
        return {
            "cagr": _safe_float(cagr),
            "volatility": _safe_float(vol),
            "sharpe": _safe_float(sharpe),
            "sortino": _safe_float(sortino),
            "max_drawdown": _safe_float(mdd),
            "calmar": _safe_float(calmar),
            "skew": _safe_float(returns.skew()),
            "kurtosis": _safe_float(returns.kurtosis()),
            "win_rate": _safe_float((returns > 0).mean())
        }

    return {
        "cagr": _safe_float(qs.stats.cagr(returns)),
        "volatility": _safe_float(qs.stats.volatility(returns)),
        "sharpe": _safe_float(qs.stats.sharpe(returns, rf=RISK_FREE_RATE)),
        "sortino": _safe_float(qs.stats.sortino(returns, rf=RISK_FREE_RATE)),
        "max_drawdown": _safe_float(qs.stats.max_drawdown(returns)),
        "calmar": _safe_float(qs.stats.calmar(returns)),
        "skew": _safe_float(qs.stats.skew(returns)),
        "kurtosis": _safe_float(qs.stats.kurtosis(returns)),
        "win_rate": _safe_float(qs.stats.win_rate(returns))
    }

def _monthly_heatmap(returns: pd.Series):
    m = (1 + returns).resample("M").prod() - 1
    if m.empty:
        return {"years": [], "months": [], "z": []}
    df = m.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="mean").sort_index()
    years = [int(y) for y in pivot.index.tolist()]
    months = [int(c) for c in pivot.columns.tolist()]
    z = pivot.fillna(0.0).values.tolist()
    return {"years": years, "months": months, "z": z}

def build_technical_analysis(symbol: str, period: str = "2y"):
    close = _download_close(symbol, period=period, interval="1d")
    returns = close.pct_change().dropna()

    metrics = _quant_metrics(returns)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = _rsi(close, 14)

    macd = _ema(close, 12) - _ema(close, 26)
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    rolling_vol = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
    cumulative, drawdown = compute_drawdown(returns)

    heatmap = _monthly_heatmap(returns)

    def _rec(s):
        return [{"date": pd.to_datetime(i).strftime("%Y-%m-%d"), "value": float(v)} for i, v in s.dropna().items()]

    payload = {
        "symbol": symbol,
        "latest_price": float(close.iloc[-1]),
        "returns_count": int(len(returns)),
        "metrics": metrics,
        "price": _rec(close),
        "sma20": _rec(sma20),
        "sma50": _rec(sma50),
        "rsi14": _rec(rsi14),
        "macd": _rec(macd),
        "macd_signal": _rec(macd_signal),
        "macd_hist": _rec(macd_hist),
        "drawdown": _rec(drawdown),
        "rolling_vol": _rec(rolling_vol),
        "monthly_heatmap": heatmap
    }
    return payload


@app.post("/api/technical")
def api_technical(payload: dict = Body(...)):
    try:
        symbol = payload.get("symbol")
        period = payload.get("period", "2y")

        if not symbol:
            raise RuntimeError("Symbol is required for technical analysis")

        allowed = {"6mo", "1y", "2y", "5y", "10y", "max"}
        if isinstance(period, str) and period not in allowed:
            period = "2y"

        result = build_technical_analysis(symbol=symbol, period=period)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _download_close_matrix(symbols, period="5y", interval="1d"):
    if not symbols:
        raise RuntimeError("No symbols supplied")
    frames = {}
    for sym in symbols:
        try:
            s = _download_close(sym, period=period, interval=interval)
            frames[sym] = s.rename(sym)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No price histories could be downloaded for the selected symbols")
    df = pd.concat(frames.values(), axis=1).sort_index().ffill().dropna(how="all")
    return df


def _series_records(s: pd.Series):
    return [{"date": pd.to_datetime(i).strftime("%Y-%m-%d"), "value": float(v)} for i, v in s.dropna().items()]


def _compare_symbol_block(symbol: str, period: str = "5y"):
    close = _download_close(symbol, period=period, interval="1d")
    returns = close.pct_change().dropna()
    metrics = _quant_metrics(returns)
    dd = compute_drawdown(returns)[1]
    rv = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
    return {
        "symbol": symbol,
        "latest_price": float(close.iloc[-1]),
        "metrics": metrics,
        "price": _series_records(close),
        "cumulative": _series_records((1 + returns).cumprod()),
        "drawdown": _series_records(dd),
        "rolling_vol": _series_records(rv)
    }


def build_msci_em_turkey_analysis(period: str = "5y"):
    universe = load_universe()
    items = flatten_universe(universe)
    universe_symbols = {x["symbol"] for x in items}

    em_core = [s for s in EM_BENCHMARK_SYMBOLS if s in universe_symbols]
    turkey_equity = [s for s in TURKEY_EQUITY_SYMBOLS if s in universe_symbols]
    turkey_bond = [s for s in TURKEY_BOND_PROXY_SYMBOLS if s in universe_symbols]

    em_country_symbols = [x["symbol"] for x in items if x["group"].startswith("Country ETF - EM") or x["group"].startswith("Country ETF - Turkey")]

    focus = []
    for s in em_core + turkey_equity + turkey_bond + em_country_symbols:
        if s not in focus:
            focus.append(s)

    price_df = _download_close_matrix(focus, period=period, interval="1d")
    ret_df = price_df.pct_change().dropna(how="all")

    metrics_rows = []
    labels = label_map()
    groups = symbol_group_map()
    for sym in ret_df.columns:
        r = ret_df[sym].dropna()
        if len(r) < 40:
            continue
        m = _quant_metrics(r)
        metrics_rows.append({
            "symbol": sym,
            "label": labels.get(sym, sym),
            "group": groups.get(sym, "Unknown"),
            **m
        })

    metrics_rows = sorted(metrics_rows, key=lambda x: (x.get("sharpe") if x.get("sharpe") is not None else -999), reverse=True)

    cum_df = (1 + ret_df).cumprod()
    rolling_vol_df = ret_df.rolling(63).std() * np.sqrt(TRADING_DAYS)

    def chart_records(symbols):
        out = {}
        for sym in symbols:
            if sym in cum_df.columns:
                out[sym] = {
                    "label": labels.get(sym, sym),
                    "group": groups.get(sym, "Unknown"),
                    "cumulative": _series_records(cum_df[sym]),
                    "rolling_vol": _series_records(rolling_vol_df[sym]),
                }
        return out

    turkey_equity_blocks = []
    for sym in turkey_equity:
        if sym in price_df.columns:
            turkey_equity_blocks.append(_compare_symbol_block(sym, period=period))

    turkey_bond_blocks = []
    for sym in turkey_bond:
        if sym in price_df.columns:
            turkey_bond_blocks.append(_compare_symbol_block(sym, period=period))

    return {
        "period": period,
        "em_core_symbols": em_core,
        "turkey_equity_symbols": turkey_equity,
        "turkey_bond_proxy_symbols": [b["symbol"] for b in turkey_bond_blocks],
        "metrics_table": metrics_rows,
        "em_core_charts": chart_records(em_core),
        "country_em_charts": chart_records(em_country_symbols),
        "turkey_equity": turkey_equity_blocks,
        "turkey_bond_proxies": turkey_bond_blocks,
        "notes": [
            "Turkey equity sleeve uses TUR when available.",
            "Turkey fixed-income sleeve is modeled with emerging-markets bond proxies because a dedicated U.S.-listed single-country Turkey sovereign bond ETF is not generally available in this universe."
        ]
    }


@app.post("/api/msci-em-analysis")
def api_msci_em_analysis(payload: dict = Body(...)):
    try:
        period = payload.get("period", "5y")
        if period not in {"1y", "2y", "5y", "10y", "max"}:
            period = "5y"
        result = build_msci_em_turkey_analysis(period=period)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return FileResponse(INDEX_FILE)


@app.get("/healthz")
def healthz():
    return {"ok": True}


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




@app.get("/api/asset-class-analysis")
def api_asset_class_analysis(group: str = "ALL", family: str = "ALL", period: str = "2y"):
    try:
        if period not in {"6mo", "1y", "2y", "5y", "10y", "max"}:
            period = "2y"
        return JSONResponse(content=build_asset_class_snapshot(group_filter=group, family_filter=family, period=period))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/major-world-indices")
def api_major_world_indices():
    try:
        return JSONResponse(content=build_market_snapshot(MAJOR_WORLD_INDICES, period="2y"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimization")
def api_optimization(payload: dict = Body(...)):
    try:
        ctx = build_portfolio_context(payload)
        top_assets = ctx["top_assets"]

        donut = [
            {
                "symbol": row["symbol"],
                "label": row["label"],
                "weight": row["weight"]
            }
            for row in ctx["weight_table"]
        ]

        result = {
            "strategy": ctx["strategy"],
            "benchmark_symbol": ctx.get("benchmark_symbol_used") if ctx.get("benchmark_available") else None,
            "risk_free_rate": RISK_FREE_RATE,
            "used_symbols": list(ctx["prices"].columns),
            "metrics": ctx["metrics"],
            "weights": ctx["weight_table"],
            "group_weights_prior": ctx.get("group_weights_prior", []),
            "group_weights_optimized": ctx.get("group_weights_optimized", []),
            "allocation_donut": donut,
            "frontier": ctx["frontier"],
            "feasibility_notes": ctx.get("feasibility_notes", []),
            "top_assets": [
                {
                    "symbol": row["symbol"],
                    "label": row["label"],
                    "expected_return": float(row["expected_return"]),
                    "volatility": float(row["volatility"]),
                    "sharpe_proxy": float(row["sharpe_proxy"])
                }
                for _, row in top_assets.iterrows()
            ]
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analytics")
def api_analytics(payload: dict = Body(...)):
    try:
        ctx = build_portfolio_context(payload)

        rolling_sharpe = compute_rolling_sharpe(ctx["portfolio_returns"], window=63)
        cumulative, drawdown = compute_drawdown(ctx["portfolio_returns"])
        regimes = compute_hmm_regimes(ctx["portfolio_returns"], n_states=3)

        if ctx.get("benchmark_available") and "benchmark" in ctx["aligned"].columns and not ctx["aligned"].empty:
            rolling_beta = compute_rolling_beta(ctx["aligned"], window=63)
            benchmark_cumulative = (1 + ctx["aligned"]["benchmark"]).cumprod()
            relative_performance = compute_relative_performance(ctx["aligned"])
            rolling_ir = compute_rolling_information_ratio(ctx["aligned"], window=63)
        else:
            rolling_beta = pd.Series(dtype=float)
            benchmark_cumulative = pd.Series(dtype=float)
            relative_performance = pd.Series(dtype=float)
            rolling_ir = pd.Series(dtype=float)

        portfolio_cumulative = (1 + ctx["portfolio_returns"]).cumprod()

        result = {
            "strategy": ctx["strategy"],
            "benchmark_symbol": ctx.get("benchmark_symbol_used") if ctx.get("benchmark_available") else None,
            "benchmark_summary": ctx["metrics"],
            "rolling_sharpe": series_to_records(rolling_sharpe),
            "cumulative": series_to_records(cumulative),
            "drawdown": series_to_records(drawdown),
            "rolling_beta": series_to_records(rolling_beta),
            "regimes": regimes,
            "portfolio_cumulative": series_to_records(portfolio_cumulative),
            "benchmark_cumulative": series_to_records(benchmark_cumulative) if not benchmark_cumulative.empty else [],
            "relative_performance": series_to_records(relative_performance),
            "rolling_information_ratio": series_to_records(rolling_ir),
            "benchmark_available": bool(ctx.get("benchmark_available")),
            "feasibility_notes": ctx.get("feasibility_notes", [])
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk")
def api_risk(payload: dict = Body(...)):
    try:
        confidence = float(payload.get("confidence", 0.95))
        horizon = int(payload.get("horizon_days", 1))

        ctx = build_portfolio_context(payload)
        risk_95 = compute_risk_metrics(ctx["portfolio_returns"], confidence=0.95, horizon=horizon)
        risk_99 = compute_risk_metrics(ctx["portfolio_returns"], confidence=0.99, horizon=horizon)
        risk_custom = compute_risk_metrics(ctx["portfolio_returns"], confidence=confidence, horizon=horizon)

        aligned_returns, aligned_weights = align_weights_for_risk(ctx["returns"], ctx["weight_series"])

        risk_contributions = compute_risk_contributions(
            aligned_weights,
            (aligned_returns.cov() * TRADING_DAYS),
            ctx["labels"]
        )

        heatmap = {
            "x": list(ctx["correlation_matrix"].columns),
            "y": list(ctx["correlation_matrix"].index),
            "z": ctx["correlation_matrix"].round(4).values.tolist()
        }

        result = {
            "strategy": ctx["strategy"],
            "benchmark_symbol": ctx.get("benchmark_symbol_used") if ctx.get("benchmark_available") else None,
            "custom": risk_custom,
            "risk_95": risk_95,
            "risk_99": risk_99,
            "risk_contributions": risk_contributions,
            "risk_contrib_group_breakdown": group_weight_breakdown(aligned_weights, ctx.get("group_map", {})),
            "weights_used_for_risk": [{"symbol": s, "weight": float(w)} for s, w in aligned_weights.items()],
            "correlation_heatmap": heatmap,
            "feasibility_notes": ctx.get("feasibility_notes", [])
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lstm-forecast")
@app.post("/api/forecast")
def api_lstm_forecast(payload: dict = Body(...)):
    try:
        family_filter = payload.get("family_filter", "ALL")
        group_filter = payload.get("group_filter", "ALL")
        symbol = payload.get("symbol")

        if not symbol:
            candidates = symbols_for_group(group_filter, family_filter)
            if not candidates:
                raise RuntimeError("No symbols found for selected group")
            symbol = candidates[0]

        result = compute_lightweight_forecast(
            symbol=symbol,
            period=payload.get("period", "3y"),
            lookback=int(payload.get("lookback", 30)),
            forecast_horizon=int(payload.get("forecast_horizon", 20))
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))