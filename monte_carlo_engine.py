"""
Monte Carlo Sugar Price Simulation Engine
==========================================
Simulates future crop-cycle prices for:
  - Precio de referencia azúcar base estándar
  - Precio al mayoreo en 23 mercados (-6.4%)

Uses historical annual reference prices (file 09) as primary series,
supplemented by daily SNIIM prices for intra-cycle volatility estimation.

Stochastic process: Ornstein-Uhlenbeck (mean-reverting) exclusively.
Sugar prices in Mexico are a regulated commodity and always revert to
a fundamental equilibrium, making OU the appropriate model.

Run: imported by dashboard.py — not standalone.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats.mstats import winsorize

# ---------------------------------------------------------------
# ASSUMPTIONS (all editable)
# ---------------------------------------------------------------
DEFAULT_N_SIMULATIONS = 10_000
DEFAULT_N_CYCLES = 3
DEFAULT_SEED = 42
SAMPLE_PATHS_DISPLAY = 200        # paths shown on fan chart
N_HISTOGRAM_BINS = 50
PRICE_THRESHOLDS = [15_000, 18_000, 20_000]

# Drift safety rails (annualized)
MAX_DRIFT = 0.20   # +20% per year cap
MIN_DRIFT = -0.10  # -10% per year floor

# Winsorization limits for log returns
WINSORIZE_LIMITS = (0.10, 0.10)   # trim 10th/90th percentile

EXCEL_DIR = Path("excel_reports")
SNIIM_CSV = Path("sniim_sugar_prices.csv")


# ---------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------
def load_reference_prices():
    """Load annual reference prices from file 09 (11 cycles)."""
    fp = EXCEL_DIR / "09_historico_precio_referencia.xlsx"
    df = pd.read_excel(fp, sheet_name="ML Ready")
    rename = {
        "Precio de referencia azúcar base estándar": "precio_referencia",
        "Precio al mayoreo en 23 mercados (- 6.4%)": "precio_mayoreo",
        "Azúcar producida": "produccion",
        "Consumo Nacional Aparente": "cna",
        "Demanda total de azúcar": "demanda_total",
        "Exportaciones totales": "exportaciones",
        "KARBE Nacional": "karbe",
        "Oferta total de azúcar": "oferta_total",
        "Superficie Industrializada": "superficie",
        "Caña industrializada": "cana",
    }
    df = df.rename(columns=rename)
    df = df.sort_values("cycle_start").reset_index(drop=True)
    return df


def load_daily_prices():
    """Load SNIIM daily prices for volatility estimation."""
    if not SNIIM_CSV.exists():
        return None
    df = pd.read_csv(SNIIM_CSV, parse_dates=["date"])
    df = df[df["product_type"] == "estandar"].sort_values("date").reset_index(drop=True)
    return df


def load_national_balance():
    """Load monthly national balance from file 01."""
    fp = EXCEL_DIR / "01_balance_nacional_azucar.xlsx"
    df = pd.read_excel(fp, sheet_name="ML Ready")
    df = df[df["type"] == "mensual"].copy()
    return df


# ---------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------
def compute_annual_log_returns(prices):
    """Compute log returns from annual price series."""
    prices = np.array(prices, dtype=float)
    prices = prices[~np.isnan(prices)]
    prices = prices[prices > 0]
    log_returns = np.diff(np.log(prices))
    return log_returns


def winsorize_returns(log_returns):
    """Winsorize log returns at 10th/90th percentiles to remove extreme spikes."""
    if len(log_returns) < 4:
        return log_returns
    return np.array(winsorize(log_returns, limits=WINSORIZE_LIMITS))


def compute_weighted_theta(prices, cycle_starts):
    """Compute long-run mean (theta) with 3x weight on pre-spike cycles (2014-2020).

    Cycles 2014-2020 get weight=3, cycles 2021+ get weight=1.
    This avoids over-weighting the recent 2021-2024 price spike.
    """
    prices = np.array(prices, dtype=float)
    cycle_starts = np.array(cycle_starts, dtype=int)
    weights = np.where(cycle_starts <= 2020, 3.0, 1.0)
    theta = float(np.average(prices, weights=weights))
    return theta


def estimate_ou_parameters(prices, cycle_starts, theta=None):
    """Estimate Ornstein-Uhlenbeck params from price levels.

    dP = kappa*(theta - P)*dt + sigma*dW

    Uses OLS regression of (P_t - P_{t-1}) on (P_{t-1} - theta) to get kappa.
    sigma = std of residuals from that regression.
    """
    prices = np.array(prices, dtype=float)
    prices = prices[~np.isnan(prices)]

    if theta is None:
        theta = compute_weighted_theta(prices, cycle_starts)

    x = prices[:-1]
    dx = np.diff(prices)

    # Regress dx on (theta - x) to get kappa
    # dx_t = kappa * (theta - x_t) + eps
    deviation = theta - x
    if np.std(deviation) < 1e-6:
        # Degenerate case
        return 0.1, theta, np.std(dx)

    slope, intercept, _, _, _ = stats.linregress(deviation, dx)
    kappa = max(slope, 0.01)  # kappa must be positive

    residuals = dx - (kappa * deviation + intercept)
    sigma = float(np.std(residuals))

    return kappa, theta, max(sigma, 100.0)  # floor sigma at 100 MXN


def fit_best_distribution(log_returns):
    """Fit normal and t-distribution; pick best via KS test."""
    results = {}

    # Normal
    mu, sigma = stats.norm.fit(log_returns)
    ks_stat, _ = stats.kstest(log_returns, "norm", args=(mu, sigma))
    results["normal"] = {"params": (mu, sigma), "ks": ks_stat}

    # T-distribution
    df_t, loc_t, scale_t = stats.t.fit(log_returns)
    ks_stat_t, _ = stats.kstest(log_returns, "t", args=(df_t, loc_t, scale_t))
    results["t"] = {"params": (df_t, loc_t, scale_t), "ks": ks_stat_t}

    best = min(results, key=lambda k: results[k]["ks"])
    return best, results[best], results


def compute_autocorrelation(log_returns):
    """Lag-1 autocorrelation of log returns."""
    if len(log_returns) < 4:
        return 0.0
    series = pd.Series(log_returns)
    return float(series.autocorr(lag=1))


def analyze_correlations(ref_df):
    """Correlation between price and supply/demand drivers."""
    cols = ["precio_referencia", "produccion", "cna", "demanda_total",
            "exportaciones", "oferta_total", "karbe"]
    sub = ref_df[cols].dropna()
    if len(sub) < 3:
        return {}
    corr = sub.corr()["precio_referencia"].drop("precio_referencia")
    return corr.to_dict()


# ---------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------
def run_simulation(
    n_simulations=DEFAULT_N_SIMULATIONS,
    n_cycles=DEFAULT_N_CYCLES,
    production_shock=0.0,
    world_balance_shock=0.0,
    fx_rate_assumption=None,
    price_thresholds=None,
    seed=DEFAULT_SEED,
    price_series="referencia",
    estimation_window=None,
):
    """
    Run full Monte Carlo simulation using Ornstein-Uhlenbeck process.

    Parameters
    ----------
    n_simulations : int
    n_cycles : int – forecast horizon in crop cycles
    production_shock : float – pct shock applied to theta (e.g., -0.10 = -10%)
    world_balance_shock : float – additional theta shift
    fx_rate_assumption : float or None – not yet used
    price_thresholds : list[float] – thresholds for probability calc
    seed : int
    price_series : str – "referencia" or "mayoreo"
    estimation_window : int or None – if set, use only last N cycles for estimation

    Returns
    -------
    dict with keys: summary, fan_chart, distribution, methodology,
                    paths (ndarray), historical
    """
    if price_thresholds is None:
        price_thresholds = PRICE_THRESHOLDS

    rng = np.random.default_rng(seed)

    # ------ Load data ------
    ref_df = load_reference_prices()
    daily_df = load_daily_prices()

    price_col = "precio_referencia" if price_series == "referencia" else "precio_mayoreo"
    all_prices = ref_df[price_col].dropna().values.astype(float)
    all_cycles = ref_df["cycle"].dropna().values
    all_cycle_starts = ref_df["cycle_start"].dropna().values.astype(int)

    if len(all_prices) < 3:
        raise ValueError("Not enough historical price data for simulation.")

    # ------ Apply estimation window ------
    if estimation_window is not None and 4 <= estimation_window < len(all_prices):
        est_prices = all_prices[-estimation_window:]
        est_cycle_starts = all_cycle_starts[-estimation_window:]
        actual_window = estimation_window
    else:
        est_prices = all_prices
        est_cycle_starts = all_cycle_starts
        actual_window = len(all_prices)

    S0 = all_prices[-1]  # last known price = starting point

    # ------ Log returns (winsorized) ------
    log_returns_raw = compute_annual_log_returns(est_prices)
    log_returns = winsorize_returns(log_returns_raw)

    # ------ Fit distribution (informational) ------
    best_dist, best_fit, all_fits = fit_best_distribution(log_returns)

    # ------ Autocorrelation (informational only — always use OU) ------
    rho = compute_autocorrelation(log_returns)

    # ------ Estimate OU parameters ------
    # theta: weighted average (3x weight for 2014-2020 cycles)
    theta = compute_weighted_theta(est_prices, est_cycle_starts)

    # Apply scenario shocks to theta
    theta_adjusted = theta * (1 + production_shock + world_balance_shock)

    # kappa and sigma from OLS regression
    kappa, _, sigma_ou = estimate_ou_parameters(est_prices, est_cycle_starts, theta)

    # ------ Drift cap safety rail ------
    # Compute implied drift equivalent: kappa * (theta - S0) / S0
    implied_drift = kappa * (theta_adjusted - S0) / max(S0, 1)
    drift_capped = False
    if implied_drift > MAX_DRIFT:
        print(f"WARNING: Implied drift {implied_drift:.4f} exceeds cap {MAX_DRIFT}. Capping.")
        # Scale theta down to hit cap
        theta_adjusted = S0 * (1 + MAX_DRIFT / kappa)
        drift_capped = True
    elif implied_drift < MIN_DRIFT:
        print(f"WARNING: Implied drift {implied_drift:.4f} below floor {MIN_DRIFT}. Capping.")
        theta_adjusted = S0 * (1 + MIN_DRIFT / kappa)
        drift_capped = True

    # ------ Supplement sigma from daily prices ------
    sigma_daily_annualized = None
    if daily_df is not None and len(daily_df) > 100:
        daily_df["ym"] = daily_df["date"].dt.to_period("M")
        monthly_avg = daily_df.groupby("ym")["price"].mean()
        monthly_returns = np.diff(np.log(monthly_avg.values))
        monthly_returns = monthly_returns[~np.isnan(monthly_returns)]
        if len(monthly_returns) > 12:
            sigma_daily_annualized = monthly_returns.std() * np.sqrt(12)
            # Blend daily-derived vol with OU residual vol (informational)

    # ------ Correlations ------
    correlations = analyze_correlations(ref_df)

    # ------ Simulate paths (OU exclusively) ------
    paths = np.zeros((n_simulations, n_cycles + 1))
    paths[:, 0] = S0

    dt = 1.0  # 1 crop cycle step

    for t in range(1, n_cycles + 1):
        dW = rng.normal(0, np.sqrt(dt), n_simulations)
        paths[:, t] = (paths[:, t-1]
                       + kappa * (theta_adjusted - paths[:, t-1]) * dt
                       + sigma_ou * dW)
        paths[:, t] = np.maximum(paths[:, t], 1000)  # floor at 1000 MXN

    # ------ Compute statistics per horizon ------
    summary = []
    distribution = {}
    for h in range(1, n_cycles + 1):
        sim_prices = paths[:, h]
        pcts = np.percentile(sim_prices, [5, 25, 50, 75, 95])
        row = {
            "horizon": h,
            "cycle_label": f"Ciclo +{h}",
            "mean": float(np.mean(sim_prices)),
            "median": float(pcts[2]),
            "std": float(np.std(sim_prices)),
            "p5": float(pcts[0]),
            "p25": float(pcts[1]),
            "p75": float(pcts[3]),
            "p95": float(pcts[4]),
            "min": float(np.min(sim_prices)),
            "max": float(np.max(sim_prices)),
        }
        for th in price_thresholds:
            pct_above = float(np.mean(sim_prices > th) * 100)
            row[f"prob_above_{int(th)}"] = pct_above
        summary.append(row)

        # Histogram data
        counts, bin_edges = np.histogram(sim_prices, bins=N_HISTOGRAM_BINS)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        distribution[f"horizon_{h}"] = {
            "bins": bin_centers.tolist(),
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

    # ------ Validation checks ------
    median_1 = summary[0]["median"]
    if abs(median_1 - S0) / S0 > 0.30:
        print(f"VALIDATION WARNING: Ciclo +1 median ({median_1:,.0f}) is >{30}% away from initial price ({S0:,.0f})")
    if len(summary) >= 3 and summary[2]["median"] > 2.5 * S0:
        print(f"VALIDATION WARNING: Ciclo +3 median ({summary[2]['median']:,.0f}) exceeds 2.5x initial price ({S0:,.0f})")
    if summary[0]["p95"] > 25000:
        print(f"VALIDATION WARNING: Ciclo +1 P95 ({summary[0]['p95']:,.0f}) exceeds $25,000")

    # ------ Fan chart data ------
    hist_years = [str(c) for c in all_cycles]
    hist_prices_list = all_prices.tolist()

    last_start = int(ref_df["cycle_start"].iloc[-1])
    forecast_years = [f"{last_start+i}/{last_start+i+1}" for i in range(1, n_cycles + 1)]

    fan_p5 = [float(np.percentile(paths[:, h], 5)) for h in range(1, n_cycles + 1)]
    fan_p25 = [float(np.percentile(paths[:, h], 25)) for h in range(1, n_cycles + 1)]
    fan_p50 = [float(np.percentile(paths[:, h], 50)) for h in range(1, n_cycles + 1)]
    fan_p75 = [float(np.percentile(paths[:, h], 75)) for h in range(1, n_cycles + 1)]
    fan_p95 = [float(np.percentile(paths[:, h], 95)) for h in range(1, n_cycles + 1)]

    sample_idx = rng.choice(n_simulations, min(SAMPLE_PATHS_DISPLAY, n_simulations), replace=False)
    sample_paths = paths[sample_idx, 1:].tolist()

    fan_chart = {
        "historical_years": hist_years,
        "historical_prices": hist_prices_list,
        "forecast_years": forecast_years,
        "p5": fan_p5,
        "p25": fan_p25,
        "p50": fan_p50,
        "p75": fan_p75,
        "p95": fan_p95,
        "sample_paths": sample_paths,
    }

    # ------ Sensitivity analysis ------
    sensitivity = compute_sensitivity(
        S0, kappa, theta_adjusted, sigma_ou, n_cycles, rng
    )

    # ------ Methodology ------
    methodology = {
        "process": "OU (Ornstein-Uhlenbeck)",
        "long_run_mean": float(theta_adjusted),
        "long_run_mean_raw": float(theta),
        "mean_reversion_speed": float(kappa),
        "sigma": float(sigma_ou),
        "sigma_daily_annualized": float(sigma_daily_annualized) if sigma_daily_annualized else None,
        "distribution_fit": best_dist,
        "autocorrelation_lag1": float(rho),
        "initial_price": float(S0),
        "estimation_window": actual_window,
        "n_historical_cycles": len(all_prices),
        "drift_capped": drift_capped,
        "implied_drift": float(kappa * (theta_adjusted - S0) / max(S0, 1)),
        "correlations": correlations,
    }

    return {
        "summary": summary,
        "fan_chart": fan_chart,
        "distribution": distribution,
        "methodology": methodology,
        "sensitivity": sensitivity,
        "paths": paths,
        "historical": {
            "years": hist_years,
            "prices": hist_prices_list,
            "price_col": price_col,
        },
    }


def compute_sensitivity(S0, kappa, theta, sigma_ou, n_cycles, rng):
    """Tornado-style sensitivity: vary each OU param ±1 std, measure median output change."""
    n_sens = 2000
    base_seed = 123

    def _sim_median(s0, kap, th, sig):
        _rng = np.random.default_rng(base_seed)
        p = np.zeros((n_sens, n_cycles + 1))
        p[:, 0] = s0
        for t in range(1, n_cycles + 1):
            dW = _rng.normal(0, 1, n_sens)
            p[:, t] = p[:, t-1] + kap * (th - p[:, t-1]) + sig * dW
            p[:, t] = np.maximum(p[:, t], 1000)
        return float(np.median(p[:, -1]))

    base_median = _sim_median(S0, kappa, theta, sigma_ou)

    params = {
        "Precio Inicial": (S0, S0 * 0.1),
        "Media Largo Plazo (Theta)": (theta, theta * 0.1),
        "Vel. Reversion (Kappa)": (kappa, kappa * 0.3),
        "Volatilidad (Sigma)": (sigma_ou, sigma_ou * 0.3),
    }

    results = []
    for name, (val, delta) in params.items():
        if name == "Precio Inicial":
            lo = _sim_median(val - delta, kappa, theta, sigma_ou)
            hi = _sim_median(val + delta, kappa, theta, sigma_ou)
        elif name == "Media Largo Plazo (Theta)":
            lo = _sim_median(S0, kappa, val - delta, sigma_ou)
            hi = _sim_median(S0, kappa, val + delta, sigma_ou)
        elif name == "Vel. Reversion (Kappa)":
            lo = _sim_median(S0, max(val - delta, 0.01), theta, sigma_ou)
            hi = _sim_median(S0, val + delta, theta, sigma_ou)
        elif name == "Volatilidad (Sigma)":
            lo = _sim_median(S0, kappa, theta, max(val - delta, 100))
            hi = _sim_median(S0, kappa, theta, val + delta)
        else:
            continue

        results.append({
            "parameter": name,
            "low": lo,
            "high": hi,
            "base": base_median,
            "swing": hi - lo,
        })

    results.sort(key=lambda x: abs(x["swing"]), reverse=True)
    return results


# ---------------------------------------------------------------
# Excel Export
# ---------------------------------------------------------------
def results_to_excel_bytes(result):
    """Convert simulation results to Excel file bytes."""
    import io
    buf = io.BytesIO()

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Summary
        summary_df = pd.DataFrame(result["summary"])
        summary_df.to_excel(writer, sheet_name="Resumen", index=False)

        # Sheet 2: Percentiles at each horizon
        pct_rows = []
        for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            row = {"Percentil": f"P{pct}"}
            for h in range(1, len(result["summary"]) + 1):
                sim_prices = result["paths"][:, h]
                row[f"Ciclo +{h}"] = float(np.percentile(sim_prices, pct))
            pct_rows.append(row)
        pd.DataFrame(pct_rows).to_excel(writer, sheet_name="Percentiles", index=False)

        # Sheet 3: First 1000 paths
        n_show = min(1000, result["paths"].shape[0])
        path_cols = {f"Ciclo_{i}": result["paths"][:n_show, i]
                     for i in range(result["paths"].shape[1])}
        pd.DataFrame(path_cols).to_excel(writer, sheet_name="Trayectorias", index=False)

        # Sheet 4: Methodology
        meth = result["methodology"]
        meth_rows = [{"Parametro": k, "Valor": str(v)} for k, v in meth.items()]
        pd.DataFrame(meth_rows).to_excel(writer, sheet_name="Metodologia", index=False)

        # Sheet 5: Sensitivity
        if result.get("sensitivity"):
            pd.DataFrame(result["sensitivity"]).to_excel(
                writer, sheet_name="Sensibilidad", index=False)

    buf.seek(0)
    return buf.getvalue()
