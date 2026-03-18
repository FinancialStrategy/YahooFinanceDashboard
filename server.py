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

app = FastAPI(title="Institutional Quant Platform")

RISK_FREE_RATE = 0.035
BENCHMARK_SYMBOL = "^GSPC"
TRADING_DAYS = 252

CACHE = {
    "snapshot": None,
    "timestamp": 0
}
CACHE_TTL_SECONDS = 600


def load_universe():
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def family_from_group_name(name: str) -> str:
    if name.startswith("Futures"):
        return "Futures"
    if name.startswith("Major Crypto"):
        return "Major Crypto"
    if name.startswith("MSCI Emerging Markets"):
        return "MSCI Emerging Markets"
    if name.startswith("Major World Indices"):
        return "Major World Indices"
    if name.startswith("Equity ETF"):
        return "Equity ETF"
    if name.startswith("Country ETF"):
        return "Country ETF"
    if name.startswith("Fixed Income ETF") or name.startswith("Emerging Market Bond ETF") or name.startswith("Inflation / Floating / Cash ETF"):
        return "Fixed Income ETF"
    if name.startswith("Real Assets ETF"):
        return "Real Assets ETF"
    if name.startswith("Commodity ETF") or name.startswith("Precious Metals ETF"):
        return "Commodity ETF"
    if name.startswith("Alternative Strategy ETF"):
        return "Alternative Strategy ETF"
    return name.split(" - ")[0]


def flatten_universe(universe_dict):
    rows = []
    for group in universe_dict["groups"]:
        family = group.get("family") or family_from_group_name(group.get("name", "Unknown"))
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


def symbols_for_group(group_filter):
    universe = load_universe()
    items = flatten_universe(universe)

    if not group_filter or group_filter == "ALL":
        return [x["symbol"] for x in items]

    return [x["symbol"] for x in items if x["group"] == group_filter]


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
    symbols = list(dict.fromkeys(symbols))

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
            if "Close" in sdf.columns:
                close_df[symbol] = sdf["Close"]
        except Exception:
            continue

    if close_df.empty:
        raise RuntimeError("No valid close-price matrix could be built")

    min_obs = max(120, int(0.65 * len(close_df)))
    close_df = close_df.dropna(axis=1, thresh=min_obs)
    close_df = close_df.sort_index().ffill().dropna()

    if close_df.shape[1] < 3:
        raise RuntimeError("Not enough valid symbols after cleaning")

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
                "family": item.get("family", family_from_group_name(item["group"])),
                "group": item["group"],
                "group_color": item["group_color"],
                "symbol": item["symbol"],
                "label": item["label"],
                "target": item["target"],
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


def clean_weight_dict(weights: dict):
    out = {}
    for k, v in weights.items():
        fv = float(v)
        if abs(fv) > 1e-8:
            out[k] = round(fv, 6)
    return dict(sorted(out.items(), key=lambda x: x[1], reverse=True))


def build_frontier_points(mu, cov_matrix):
    points = []

    try:
        ef_min = EfficientFrontier(mu, cov_matrix, weight_bounds=(0, 1))
        ef_min.min_volatility()
        min_ret, _, _ = ef_min.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
    except Exception:
        return points

def optimize_with_strategy(prices, strategy, target_volatility, target_return, prior_weights=None, group_targets=None, min_weight=0.0, max_weight=1.0, group_map=None):
    mu = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    cov_matrix = risk_models.CovarianceShrinkage(prices, frequency=TRADING_DAYS).ledoit_wolf()
    rets = expected_returns.returns_from_prices(prices)

    # Prior weights (optional)
    symbols = list(prices.columns)
    if group_map is None:
        group_map = symbol_group_map()
    if prior_weights is not None:
        prior_weights = normalize_weight_dict(prior_weights, symbols)

    # Group targets (optional) - normalize to sum to 1
    if isinstance(group_targets, dict) and len(group_targets) > 0:
        gt = {str(k): float(v) for k, v in group_targets.items()}
        mx = max(gt.values()) if gt else 0.0
        if mx > 1.5:
            gt = {k: v/100.0 for k, v in gt.items()}
        tot = sum(max(v, 0.0) for v in gt.values())
        if tot > 0:
            group_targets = {k: max(v, 0.0)/tot for k, v in gt.items()}
        else:
            group_targets = None

    # A special mode: use prior weights as the portfolio (no optimization)
    if strategy == "prior_only":
        if prior_weights is None:
            raise RuntimeError("Prior weights were not supplied.")
        wvec = np.array([prior_weights[s] for s in symbols], dtype=float)
        port_ret = float(mu.values @ wvec)
        port_vol = float(np.sqrt(wvec.T @ cov_matrix.values @ wvec))
        sharpe = float((port_ret - RISK_FREE_RATE) / port_vol) if port_vol > 0 else 0.0
        weights = {s: float(round(prior_weights[s], 6)) for s in symbols if prior_weights[s] > 1e-8}
        perf = (port_ret, port_vol, sharpe)
        frontier = build_frontier_points(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
        return mu, cov_matrix, clean_weight_dict(weights), perf, frontier


    if strategy == "max_sharpe":
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))

        if group_targets and cp is not None:
            # enforce asset-class allocations as hard constraints
            for g, tgt in group_targets.items():
                idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
                if len(idxs) == 0:
                    continue
                ef.add_constraint(lambda w, idxs=idxs, tgt=float(tgt): cp.sum(w[idxs]) == tgt)
        ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "min_volatility":
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))

        if group_targets and cp is not None:
            # enforce asset-class allocations as hard constraints
            for g, tgt in group_targets.items():
                idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
                if len(idxs) == 0:
                    continue
                ef.add_constraint(lambda w, idxs=idxs, tgt=float(tgt): cp.sum(w[idxs]) == tgt)
        ef.min_volatility()
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "max_quadratic_utility":
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))

        if group_targets and cp is not None:
            # enforce asset-class allocations as hard constraints
            for g, tgt in group_targets.items():
                idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
                if len(idxs) == 0:
                    continue
                ef.add_constraint(lambda w, idxs=idxs, tgt=float(tgt): cp.sum(w[idxs]) == tgt)
        ef.max_quadratic_utility(risk_aversion=1.0)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "efficient_risk":
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))

        if group_targets and cp is not None:
            # enforce asset-class allocations as hard constraints
            for g, tgt in group_targets.items():
                idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
                if len(idxs) == 0:
                    continue
                ef.add_constraint(lambda w, idxs=idxs, tgt=float(tgt): cp.sum(w[idxs]) == tgt)
        ef.efficient_risk(target_volatility=float(target_volatility))
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "efficient_return":
        ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))

        if group_targets and cp is not None:
            # enforce asset-class allocations as hard constraints
            for g, tgt in group_targets.items():
                idxs = [i for i, s in enumerate(symbols) if group_map.get(s, "Unknown") == g]
                if len(idxs) == 0:
                    continue
                ef.add_constraint(lambda w, idxs=idxs, tgt=float(tgt): cp.sum(w[idxs]) == tgt)
        ef.efficient_return(target_return=float(target_return))
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "hrp":
        hrp = HRPOpt(rets)
        weights = hrp.optimize()
        perf = hrp.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "cla_max_sharpe":
        cla = CLA(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
        cla.max_sharpe()
        weights = cla.clean_weights()
        perf = cla.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    elif strategy == "cla_min_volatility":
        cla = CLA(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
        cla.min_volatility()
        weights = cla.clean_weights()
        perf = cla.portfolio_performance(risk_free_rate=RISK_FREE_RATE)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    frontier = build_frontier_points(mu, cov_matrix, weight_bounds=(float(min_weight), float(max_weight)))
    return mu, cov_matrix, clean_weight_dict(weights), perf, frontier


def resolve_symbols(payload):
    symbols = payload.get("symbols", [])
    group_filter = payload.get("group_filter", "ALL")

    if not symbols:
        symbols = symbols_for_group(group_filter)

    if group_filter and group_filter != "ALL":
        allowed = set(symbols_for_group(group_filter))
        symbols = [s for s in symbols if s in allowed]

    symbols = [s for s in symbols if s != BENCHMARK_SYMBOL]
    symbols = list(dict.fromkeys(symbols))
    return symbols


def build_portfolio_context(payload):
    strategy = payload.get("strategy", "max_sharpe")
    target_volatility = float(payload.get("target_volatility", 0.15))
    target_return = float(payload.get("target_return", 0.10))
    symbols = resolve_symbols(payload)

    # Prior weights (from UI) and asset-class targets (optional)
    prior_weights_input = payload.get("user_weights") or payload.get("prior_weights") or {}
    group_targets_input = payload.get("group_targets") or {}
    min_weight = float(payload.get("min_weight", 0.0))
    max_weight = float(payload.get("max_weight", 1.0))
    if max_weight <= 0 or max_weight > 1:
        max_weight = 1.0
    if min_weight < 0 or min_weight >= max_weight:
        min_weight = 0.0

    group_map = symbol_group_map()
    prior_weights = normalize_weight_dict(prior_weights_input, symbols)
    # For convenience: if group targets are not provided, derive them from prior weights
    if not group_targets_input:
        tmp = pd.Series(prior_weights)
        tmp.index = [group_map.get(s, "Unknown") for s in tmp.index]
        group_targets_input = tmp.groupby(tmp.index).sum().to_dict()

    if len(symbols) < 3:
        raise RuntimeError("At least 3 assets are required for portfolio construction")

    labels = label_map()
    price_symbols = symbols + [BENCHMARK_SYMBOL]
    price_df = download_price_matrix(price_symbols, period="2y", interval="1d")

    if BENCHMARK_SYMBOL not in price_df.columns:
        raise RuntimeError("Benchmark could not be downloaded")

    benchmark_prices = price_df[BENCHMARK_SYMBOL].copy()
    prices = price_df.drop(columns=[BENCHMARK_SYMBOL], errors="ignore")

    if prices.shape[1] < 3:
        raise RuntimeError("Not enough assets available after cleaning")

    mu, cov_matrix, weights, perf, frontier = optimize_with_strategy(
        prices=prices,
        strategy=strategy,
        target_volatility=target_volatility,
        target_return=target_return,
        prior_weights=prior_weights,
        group_targets=group_targets_input,
        min_weight=min_weight,
        max_weight=max_weight,
        group_map=group_map
    )

    returns = prices.pct_change().dropna()
    cov_returns = returns.cov() * TRADING_DAYS

    benchmark_returns = benchmark_prices.pct_change().dropna()

    weight_series = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    prior_weight_series = pd.Series(prior_weights).reindex(prices.columns).fillna(0.0)
    portfolio_returns = returns.mul(weight_series, axis=1).sum(axis=1).dropna()

    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1
    ).dropna()

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
        "cov_returns": cov_returns,

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

    portfolio_var = float(w.T @ sigma @ w)
    if portfolio_var <= 0:
        return []

    portfolio_vol = np.sqrt(portfolio_var)
    marginal_contrib = (sigma @ w) / portfolio_vol
    component_contrib = w * marginal_contrib
    pct_contrib = component_contrib / portfolio_vol

    rows = []
    for i, sym in enumerate(weight_series.index):
        rows.append({
            "symbol": sym,
            "label": labels.get(sym, sym),
            "weight": float(w[i, 0]),
            "marginal_risk_contribution": float(marginal_contrib[i, 0]),
            "component_risk_contribution": float(component_contrib[i, 0]),
            "percent_risk_contribution": float(pct_contrib[i, 0])
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
# - Uses deterministic Yahoo Finance data only
# - Metrics are computed on daily returns; charts on price and derived series
# ============================

def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder smoothing via EMA with alpha=1/period
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
    # Returns are daily simple returns (not log returns)
    if qs is None:
        # Fallback: compute essentials without naming dependencies in messages
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

    # Primary path: metrics engine
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

        # Keep allowed periods tight
        allowed = {"6mo", "1y", "2y", "5y", "10y", "max"}
        if isinstance(period, str) and period not in allowed:
            period = "2y"

        result = build_technical_analysis(symbol=symbol, period=period)
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
    universe = load_universe()
    for group in universe.get("groups", []):
        if not group.get("family"):
            group["family"] = family_from_group_name(group.get("name", "Unknown"))
    return JSONResponse(content=universe)


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
            "benchmark_symbol": BENCHMARK_SYMBOL,
            "risk_free_rate": RISK_FREE_RATE,
            "used_symbols": list(ctx["prices"].columns),
            "metrics": ctx["metrics"],
            "weights": ctx["weight_table"],
            "group_weights_prior": ctx.get("group_weights_prior", []),
            "group_weights_optimized": ctx.get("group_weights_optimized", []),

            "allocation_donut": donut,
            "frontier": ctx["frontier"],
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
        rolling_beta = compute_rolling_beta(ctx["aligned"], window=63)
        regimes = compute_hmm_regimes(ctx["portfolio_returns"], n_states=3)

        benchmark_cumulative = (1 + ctx["aligned"]["benchmark"]).cumprod()
        portfolio_cumulative = (1 + ctx["aligned"]["portfolio"]).cumprod()
        relative_performance = compute_relative_performance(ctx["aligned"])
        rolling_ir = compute_rolling_information_ratio(ctx["aligned"], window=63)

        result = {
            "strategy": ctx["strategy"],
            "benchmark_symbol": BENCHMARK_SYMBOL,
            "rolling_sharpe": series_to_records(rolling_sharpe),
            "cumulative": series_to_records(cumulative),
            "drawdown": series_to_records(drawdown),
            "rolling_beta": series_to_records(rolling_beta),
            "regimes": regimes,
            "portfolio_cumulative": series_to_records(portfolio_cumulative),
            "benchmark_cumulative": series_to_records(benchmark_cumulative),
            "relative_performance": series_to_records(relative_performance),
            "rolling_information_ratio": series_to_records(rolling_ir)
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
            "benchmark_symbol": BENCHMARK_SYMBOL,
            "custom": risk_custom,
            "risk_95": risk_95,
            "risk_99": risk_99,
            "risk_contributions": risk_contributions,
            "risk_contrib_group_breakdown": group_weight_breakdown(aligned_weights, ctx.get("group_map", {})),
            "weights_used_for_risk": [{"symbol": s, "weight": float(w)} for s, w in aligned_weights.items()],
            "correlation_heatmap": heatmap
        }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lstm-forecast")
@app.post("/api/forecast")
def api_lstm_forecast(payload: dict = Body(...)):
    try:
        group_filter = payload.get("group_filter", "ALL")
        symbol = payload.get("symbol")

        if not symbol:
            candidates = symbols_for_group(group_filter)
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

def _build_watchlist_snapshot(items, period="1y", interval="1d"):
    if not items:
        return {"generated_at": pd.Timestamp.utcnow().isoformat(), "count": 0, "rows": []}
    symbols = [x["symbol"] for x in items]
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
        raise RuntimeError("Yahoo Finance returned empty dataset for the selected watchlist")
    rows = []
    total_symbols = len(symbols)
    for item in items:
        symbol = item["symbol"]
        try:
            sdf = extract_symbol_frame(raw, symbol, total_symbols).copy().dropna(how="all")
            if sdf.empty or "Close" not in sdf.columns:
                continue
            close = sdf["Close"].dropna()
            volume = sdf["Volume"].dropna() if "Volume" in sdf.columns else pd.Series(dtype=float)
            if len(close) < 2:
                continue
            last = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            rows.append({
                "symbol": symbol,
                "label": item.get("label", symbol),
                "group": item.get("bucket") or item.get("region") or item.get("group") or "Watchlist",
                "price": last,
                "ret_1d_pct": ((last / prev) - 1.0) * 100.0 if prev else None,
                "ret_1w_pct": safe_pct_return(close, 5),
                "ret_1m_pct": safe_pct_return(close, 21),
                "ret_3m_pct": safe_pct_return(close, 63),
                "ret_6m_pct": safe_pct_return(close, 126),
                "ret_1y_pct": safe_pct_return(close, 252),
                "ytd_pct": ytd_return(close),
                "vol_30d_pct": annualized_volatility(close, 30),
                "avg_volume_20d": avg_volume(volume, 20),
                "sparkline": close.tail(30).round(6).tolist()
            })
        except Exception:
            continue
    return {"generated_at": pd.Timestamp.utcnow().isoformat(), "count": len(rows), "rows": rows}


@app.get("/api/major-world-indices")
def api_major_world_indices():
    universe = load_universe()
    items = universe.get("major_world_indices", [])
    return JSONResponse(content=_build_watchlist_snapshot(items, period="1y"))


@app.get("/api/futures-dashboard")
def api_futures_dashboard():
    universe = load_universe()
    items = universe.get("futures_watchlist", [])
    return JSONResponse(content=_build_watchlist_snapshot(items, period="1y"))
