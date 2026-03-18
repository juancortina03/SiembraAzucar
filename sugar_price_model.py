"""
Sugar Price Prediction -- ML Pipeline v4
=========================================
Fundamentals + external market approach: sugar price is driven by
supply/demand balance AND international market conditions.

Core pricing logic:
  Supply  = Produccion + Importaciones + Inventario Inicial
  Demand  = Consumo Nacional Aparente + Exportaciones
  Market  = USD/MXN, ICE No.11/16 (world/US sugar), WTI, BRL/USD
  Price  ~  f(Demand, Supply, Production, Market)

The monthly model learns the relationship between balance fundamentals,
external market signals, and monthly average prices. No price momentum
or moving averages -- forecasts depend on fundamentals + market conditions.

External data sources (optional, graceful degradation):
  - Banxico SIE: USD/MXN FIX rate
  - FRED: ICE Sugar No.11, No.16, WTI crude oil, BRL/USD

A daily model is also trained for comparison/research but is NOT used
for forecasting.

Outputs in model_results/.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

PRICES_CSV = "sniim_sugar_prices.csv"
BALANCE_XLSX = "excel_reports/01_balance_nacional_azucar.xlsx"
OUTPUT_DIR = Path("model_results")
MODELS_DIR = OUTPUT_DIR / "models"
FORECAST_DAYS = 90

BALANCE_COLS = [
    "produccion", "importaciones", "exportaciones_totales",
    "consumo_nacional_aparente", "inventario_inicial", "inventario_final",
    "oferta_total", "demanda_total",
]

BALANCE_FEATURE_COLS = [
    "produccion", "importaciones", "exportaciones_totales",
    "consumo_nacional_aparente", "inventario_inicial", "inventario_final",
    "oferta_total", "demanda_total",
    "net_exports", "supply_demand_ratio", "inventory_to_consumption",
    "production_share", "excess_supply", "demand_pressure",
    "inventory_months",
]


def compute_derived_ratios(prod, imp, exp, cna, inv_i, inv_f, ot, dt_):
    """Compute all derived balance ratios from raw balance variables.
    Single source of truth -- used by load_balance, project_balance,
    _build_monthly_row, and forecast_monthly scenario block."""
    supply = prod + imp + inv_i
    demand = cna + exp
    return {
        "net_exports": exp - imp,
        "supply_demand_ratio": ot / dt_ if dt_ else 1,
        "demand_pressure": demand / supply if supply else 1,
        "excess_supply": supply - demand,
        "inventory_to_consumption": inv_f / cna if cna else 1,
        "inventory_months": inv_f / (cna / 12) if cna else 0,
        "production_share": prod / ot if ot else 0,
    }


class EnsembleModel:
    """Weighted ensemble that implements the scikit-learn .predict() interface."""
    def __init__(self, models, weights, scaler=None):
        self.models = models       # dict: name -> fitted model
        self.weights = weights     # dict: name -> float weight
        self.scaler = scaler       # StandardScaler (fitted) for Ridge
        self._is_ensemble = True

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=float)
        for name, model in self.models.items():
            use_scaled = (name == "Ridge") and self.scaler is not None
            X_m = self.scaler.transform(X) if use_scaled else X
            pred += self.weights[name] * model.predict(X_m)
        return pred


# ---------------------------------------------------------------
# 1. Load & Prepare
# ---------------------------------------------------------------

def load_prices(csv_path: str = PRICES_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"Price data is empty: {csv_path}")
    required = {"date", "price", "product_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Price CSV missing columns: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def load_balance(xlsx_path: str = BALANCE_XLSX) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Data")
    if df.empty:
        raise ValueError(f"Balance data is empty: {xlsx_path}")
    if "type" not in df.columns:
        raise ValueError(f"Balance sheet missing 'type' column")
    monthly = df[df["type"] == "mensual"].copy()
    monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce")
    monthly["month_number"] = pd.to_numeric(monthly["month_number"], errors="coerce")
    monthly = monthly.dropna(subset=["year", "month_number"])
    monthly["year"] = monthly["year"].astype(int)
    monthly["month_number"] = monthly["month_number"].astype(int)

    keep = ["year", "month_number"] + [c for c in BALANCE_COLS if c in monthly.columns]
    monthly = monthly[keep].copy()
    for c in BALANCE_COLS:
        if c in monthly.columns:
            monthly[c] = pd.to_numeric(monthly[c], errors="coerce")

    monthly = monthly.groupby(["year", "month_number"]).last().reset_index()
    monthly = monthly.sort_values(["year", "month_number"]).reset_index(drop=True)

    # --- Fundamental derived features (via shared helper) ---
    def _row_ratios(r):
        return compute_derived_ratios(
            prod=r.get("produccion", 0) or 0,
            imp=r.get("importaciones", 0) or 0,
            exp=r.get("exportaciones_totales", 0) or 0,
            cna=r.get("consumo_nacional_aparente", 0) or 0,
            inv_i=r.get("inventario_inicial", 0) or 0,
            inv_f=r.get("inventario_final", 0) or 0,
            ot=r.get("oferta_total", 0) or 0,
            dt_=r.get("demanda_total", 0) or 0,
        )
    ratios = monthly.apply(_row_ratios, axis=1, result_type="expand")
    for c in ratios.columns:
        monthly[c] = ratios[c]

    return monthly


def project_balance(balance_df, target_months, recent_years=5):
    """
    Project future balance values for each (year, month) in target_months.
    Combines:
      1) Seasonal profile: historical average for that calendar month
      2) YoY trend: recent growth rate applied on top of the seasonal baseline

    Returns a list of dicts, one per target month, with all BALANCE_COLS +
    derived ratio features.
    """
    raw_cols = [c for c in BALANCE_COLS if c in balance_df.columns]
    latest_year = int(balance_df["year"].max())
    recent = balance_df[balance_df["year"] >= latest_year - recent_years]

    # 1) Seasonal profile: mean value per calendar month (from recent years)
    seasonal = recent.groupby("month_number")[raw_cols].mean()

    # 2) YoY trend per month: average annual growth rate over recent years
    #    growth_rate = mean of (value_year / value_year-1 - 1) per month
    yoy_rates = {}
    for m in range(1, 13):
        m_data = recent[recent["month_number"] == m].sort_values("year")
        if len(m_data) < 2:
            yoy_rates[m] = {c: 0.0 for c in raw_cols}
            continue
        rates = {}
        for c in raw_cols:
            vals = m_data[c].values
            annual_changes = []
            for i in range(1, len(vals)):
                if vals[i - 1] > 0:
                    annual_changes.append(vals[i] / vals[i - 1] - 1)
            rates[c] = float(np.median(annual_changes)) if annual_changes else 0.0
        yoy_rates[m] = rates

    projected = []
    for (y, m) in target_months:
        row = {}
        years_ahead = y - latest_year
        for c in raw_cols:
            base = seasonal.loc[m, c] if m in seasonal.index else 0
            trend = yoy_rates.get(m, {}).get(c, 0)
            # Apply trend compounded for years ahead
            row[c] = base * ((1 + trend) ** years_ahead)
            row[c] = max(row[c], 0)  # no negative volumes

        # Recompute derived ratios via shared helper
        row.update(compute_derived_ratios(
            prod=row.get("produccion", 0),
            imp=row.get("importaciones", 0),
            exp=row.get("exportaciones_totales", 0),
            cna=row.get("consumo_nacional_aparente", 1) or 1,
            inv_i=row.get("inventario_inicial", 0),
            inv_f=row.get("inventario_final", 0),
            ot=row.get("oferta_total", 1) or 1,
            dt_=row.get("demanda_total", 1) or 1,
        ))

        row["year"] = y
        row["month_number"] = m
        projected.append(row)

    return projected


def prepare_product_series(df: pd.DataFrame, product: str) -> pd.DataFrame:
    s = df[df["product_type"] == product][["date", "price"]].copy()
    s = s.drop_duplicates(subset="date", keep="last")
    s = s.set_index("date").sort_index()
    s = s.asfreq("B")
    s["price"] = s["price"].interpolate(method="linear")
    s = s.dropna()
    return s.reset_index()


def remove_outliers(df, col="price"):
    d = df.copy()
    d["_ret"] = d[col].pct_change()
    ret_mean = d["_ret"].mean()
    ret_std = d["_ret"].std()
    lower = ret_mean - 3 * ret_std
    upper = ret_mean + 3 * ret_std
    outlier_mask = (d["_ret"] < lower) | (d["_ret"] > upper)
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        d.loc[outlier_mask, "_ret"] = d.loc[outlier_mask, "_ret"].clip(lower, upper)
        d["_cp"] = d[col].iloc[0]
        for i in range(1, len(d)):
            d.iloc[i, d.columns.get_loc("_cp")] = (
                d.iloc[i - 1, d.columns.get_loc("_cp")] * (1 + d.iloc[i, d.columns.get_loc("_ret")])
            )
        scale = df[col].iloc[-1] / d["_cp"].iloc[-1]
        d[col] = d["_cp"] * scale
        d = d.drop(columns=["_ret", "_cp"])
    else:
        d = d.drop(columns=["_ret"])
    return d, n_outliers


def merge_balance_to_daily(daily, balance):
    daily = daily.copy()
    daily["_year"] = daily["date"].dt.year
    daily["_month"] = daily["date"].dt.month
    merged = daily.merge(
        balance, left_on=["_year", "_month"], right_on=["year", "month_number"],
        how="left", suffixes=("", "_bal"),
    )
    bal_cols = [c for c in balance.columns if c not in ("year", "month_number")]
    for c in bal_cols:
        if c in merged.columns:
            merged[c] = merged[c].ffill()
    merged = merged.drop(columns=["_year", "_month", "year", "month_number"], errors="ignore")
    return merged

# ---------------------------------------------------------------
# 2. Build monthly price dataset
# ---------------------------------------------------------------

def build_monthly_dataset(prices_df, balance_df, product, external_df=None):
    """
    Aggregate daily prices to monthly averages and merge with balance data.
    Optionally merges external market data (USD/MXN, futures, oil).
    This is the core dataset where price = f(supply, demand, production, external).
    """
    prod_df = prices_df[prices_df["product_type"] == product].copy()
    prod_df["year"] = prod_df["date"].dt.year
    prod_df["month"] = prod_df["date"].dt.month

    monthly_price = prod_df.groupby(["year", "month"]).agg(
        avg_price=("price", "mean"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        price_std=("price", "std"),
        n_days=("price", "count"),
    ).reset_index()

    merged = monthly_price.merge(
        balance_df, left_on=["year", "month"], right_on=["year", "month_number"],
        how="inner",
    )
    merged = merged.drop(columns=["month_number"], errors="ignore")
    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    # Target variable and price reference (NOT model features)
    merged["prev_price"] = merged["avg_price"].shift(1)
    merged["price_change_1m"] = merged["avg_price"].pct_change(1)

    merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
    merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

    # Merge external market data if available
    if external_df is not None and not external_df.empty:
        merged = merged.merge(external_df, on=["year", "month"], how="left")
        # Forward-fill gaps in external data
        for col in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"]:
            if col in merged.columns:
                merged[col] = merged[col].ffill()
        # Derived external features
        if "usd_mxn" in merged.columns and "avg_price" in merged.columns:
            merged["mx_sugar_usd"] = merged["avg_price"] / merged["usd_mxn"].replace(0, np.nan)
        if "ice_no11" in merged.columns and "avg_price" in merged.columns:
            # Premium of MX price over world price (both in local terms)
            merged["premium_vs_world"] = merged["avg_price"] / merged["ice_no11"].replace(0, np.nan)

    merged = merged.dropna(subset=["prev_price"]).reset_index(drop=True)
    return merged

# ---------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------

EXTERNAL_FEATURES = [
    "usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd",
    "mx_sugar_usd", "premium_vs_world",
]

MONTHLY_FEATURES = [
    # Direct balance variables (fundamentals -- primary drivers)
    "produccion", "importaciones", "exportaciones_totales",
    "consumo_nacional_aparente", "inventario_final",
    "oferta_total", "demanda_total",
    # Derived ratios (the pricing formula)
    "supply_demand_ratio", "demand_pressure", "excess_supply",
    "inventory_to_consumption", "inventory_months",
    "net_exports", "production_share",
    # Seasonality (cyclical encoding)
    "month_sin", "month_cos",
    # External market data (added when available)
    "usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd",
    "mx_sugar_usd", "premium_vs_world",
]

DAILY_FEATURE_COLS = [
    "month_sin", "month_cos",
    "lag_ret_1", "lag_ret_2", "lag_ret_5", "lag_ret_10",
    "ma_ratio_5", "ma_ratio_20",
    "std_norm_5",
    "ema_ratio_10",
    "return_5d",
    "volatility_20",
    # balance fundamentals at daily level
    "supply_demand_ratio", "demand_pressure", "excess_supply",
    "inventory_to_consumption", "production_share",
    "produccion", "importaciones", "exportaciones_totales",
    "consumo_nacional_aparente", "inventario_final",
    "oferta_total", "demanda_total",
    "net_exports", "inventory_months",
]


def engineer_daily_features(df):
    d = df.copy()
    d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    d["return_1d"] = d["price"].pct_change(1)

    for lag in [1, 2, 5, 10]:
        d[f"lag_ret_{lag}"] = d["return_1d"].shift(lag)

    for w in [5, 20]:
        ma = d["price"].rolling(window=w).mean()
        d[f"ma_ratio_{w}"] = d["price"] / ma
    d["std_norm_5"] = d["price"].rolling(5).std() / d["price"]
    d["ema_ratio_10"] = d["price"] / d["price"].ewm(span=10).mean()
    d["return_5d"] = d["price"].pct_change(5)
    d["volatility_20"] = d["return_1d"].rolling(20).std()

    d["next_return"] = d["return_1d"].shift(-1)
    d["target_price"] = d["price"].shift(-1)
    d = d.dropna(subset=["next_return"]).reset_index(drop=True)
    return d


def get_feature_cols(df, feature_list):
    return [c for c in feature_list if c in df.columns and df[c].notna().sum() > 10]

# ---------------------------------------------------------------
# 4. Train Monthly Model (fundamentals -> price)
# ---------------------------------------------------------------

def train_monthly_model(monthly_df):
    """
    Train models on monthly data: balance fundamentals -> price % change.
    Includes TimeSeriesSplit CV, weighted ensemble, data-driven clip bounds,
    and residual-based confidence interval percentiles.
    """
    feature_cols = get_feature_cols(monthly_df, MONTHLY_FEATURES)
    df_clean = monthly_df.dropna(subset=feature_cols + ["price_change_1m"]).reset_index(drop=True)

    split_idx = int(len(df_clean) * 0.75)
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    X_train = train[feature_cols].values
    y_train = train["price_change_1m"].values
    X_test = test[feature_cols].values
    y_test = test["price_change_1m"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # TimeSeriesSplit CV
    n_splits = min(5, max(2, len(train) // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=8,
            max_features=0.7, n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=8, max_features=0.7,
            random_state=42,
        ),
    }

    results = {}
    metrics_rows = []

    for name, model in models.items():
        use_scaled = name == "Ridge"
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        # Cross-validation
        cv_maes = []
        for cv_train_idx, cv_val_idx in tscv.split(Xtr):
            m_cv = type(model)(**model.get_params())
            m_cv.fit(Xtr[cv_train_idx], y_train[cv_train_idx])
            val_pred_chg = m_cv.predict(Xtr[cv_val_idx])
            val_prev = train.iloc[cv_val_idx]["prev_price"].values
            val_pred_price = val_prev * (1 + val_pred_chg)
            val_actual_price = train.iloc[cv_val_idx]["avg_price"].values
            cv_maes.append(mean_absolute_error(val_actual_price, val_pred_price))
        cv_mae = float(np.mean(cv_maes))
        cv_mae_std = float(np.std(cv_maes))

        # Fit on full training set
        model.fit(Xtr, y_train)
        train_pred_chg = model.predict(Xtr)
        test_pred_chg = model.predict(Xte)

        test_prev = test["prev_price"].values
        test_pred_price = test_prev * (1 + test_pred_chg)
        test_actual_price = test["avg_price"].values
        train_prev = train["prev_price"].values
        train_pred_price = train_prev * (1 + train_pred_chg)
        train_actual_price = train["avg_price"].values

        train_mae = mean_absolute_error(train_actual_price, train_pred_price)
        train_r2 = r2_score(train_actual_price, train_pred_price)
        test_mae = mean_absolute_error(test_actual_price, test_pred_price)
        test_rmse = np.sqrt(mean_squared_error(test_actual_price, test_pred_price))
        test_r2 = r2_score(test_actual_price, test_pred_price)
        test_mape = np.mean(np.abs((test_actual_price - test_pred_price) / test_actual_price)) * 100
        change_r2 = r2_score(y_test, test_pred_chg)

        metrics_rows.append({
            "model": f"Monthly {name}",
            "MAE": test_mae, "RMSE": test_rmse,
            "R2_price": test_r2, "MAPE_pct": test_mape,
            "Change_R2": change_r2,
            "Train_MAE": train_mae, "Train_R2": train_r2,
            "Overfit_Ratio": train_mae / test_mae if test_mae > 0 else 0,
            "CV_MAE": cv_mae, "CV_MAE_std": cv_mae_std,
        })
        results[name] = {
            "model": model, "scaler": scaler if use_scaled else None,
            "feat_weights": None,
            "test_pred": test_pred_price, "test_actual": test_actual_price,
            "metrics": {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2,
                        "MAPE": test_mape, "Change_R2": change_r2},
            "train_metrics": {"MAE": train_mae, "R2": train_r2},
            "cv_mae": cv_mae, "cv_mae_std": cv_mae_std,
        }
        print(f"  {name}: MAE={test_mae:.2f}, CV_MAE={cv_mae:.2f}+/-{cv_mae_std:.1f}, R2={test_r2:.4f}, Change R2={change_r2:.4f}  |  Train MAE={train_mae:.2f}, Train R2={train_r2:.4f}")

    # Weighted ensemble (inverse-MAE weighting)
    inv_maes = {n: 1.0 / r["metrics"]["MAE"] for n, r in results.items()}
    total_inv = sum(inv_maes.values())
    weights = {n: v / total_inv for n, v in inv_maes.items()}

    ens_pred_chg = np.zeros_like(y_test, dtype=float)
    for name, w in weights.items():
        use_scaled = name == "Ridge"
        Xte = X_test_s if use_scaled else X_test
        ens_pred_chg += w * results[name]["model"].predict(Xte)

    ens_pred_price = test["prev_price"].values * (1 + ens_pred_chg)
    ens_mae = mean_absolute_error(test_actual_price, ens_pred_price)
    ens_rmse = np.sqrt(mean_squared_error(test_actual_price, ens_pred_price))
    ens_r2 = r2_score(test_actual_price, ens_pred_price)
    ens_mape = np.mean(np.abs((test_actual_price - ens_pred_price) / test_actual_price)) * 100
    ens_change_r2 = r2_score(y_test, ens_pred_chg)

    metrics_rows.append({
        "model": "Monthly Ensemble",
        "MAE": ens_mae, "RMSE": ens_rmse,
        "R2_price": ens_r2, "MAPE_pct": ens_mape,
        "Change_R2": ens_change_r2,
        "Train_MAE": 0, "Overfit_Ratio": 0,
        "CV_MAE": 0, "CV_MAE_std": 0,
    })
    ensemble = EnsembleModel(
        models={n: r["model"] for n, r in results.items()},
        weights=weights, scaler=scaler,
    )
    results["Ensemble"] = {
        "model": ensemble, "scaler": None,  # ensemble handles scaling internally
        "feat_weights": None,
        "test_pred": ens_pred_price, "test_actual": test_actual_price,
        "metrics": {"MAE": ens_mae, "RMSE": ens_rmse, "R2": ens_r2,
                    "MAPE": ens_mape, "Change_R2": ens_change_r2},
        "train_metrics": {"MAE": 0},
        "weights": weights,
    }
    print(f"  Ensemble ({', '.join(f'{n}:{w:.2f}' for n,w in weights.items())}): MAE={ens_mae:.2f}, R2={ens_r2:.4f}, Change R2={ens_change_r2:.4f}")

    # Data-driven clip bounds from training target distribution
    clip_bounds = (float(np.percentile(y_train, 1)), float(np.percentile(y_train, 99)))

    # Residual percentiles for confidence intervals (using best individual model)
    best_individual = min(
        [n for n in results if n != "Ensemble"],
        key=lambda k: results[k]["metrics"]["MAE"]
    )
    residuals = results[best_individual]["test_actual"] - results[best_individual]["test_pred"]
    residual_percentiles = {
        "p10": float(np.percentile(residuals, 10)),
        "p25": float(np.percentile(residuals, 25)),
        "p75": float(np.percentile(residuals, 75)),
        "p90": float(np.percentile(residuals, 90)),
    }

    return results, metrics_rows, feature_cols, train, test, scaler, clip_bounds, residual_percentiles

# ---------------------------------------------------------------
# 5. Train Daily Model (balance + momentum -> next-day return)
# ---------------------------------------------------------------

def train_daily_model(df_feat, feature_cols):
    df_clean = df_feat.dropna(subset=feature_cols + ["next_return", "target_price"]).reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    X_train = train[feature_cols].values
    y_train_ret = train["next_return"].values
    today_train = train["price"].values

    X_test = test[feature_cols].values
    y_test_ret = test["next_return"].values
    y_test_price = test["target_price"].values
    today_test = test["price"].values

    # StandardScaler for Ridge (fixes the scale-sensitivity issue)
    daily_scaler = StandardScaler()
    X_train_s = daily_scaler.fit_transform(X_train)
    X_test_s = daily_scaler.transform(X_test)

    # TimeSeriesSplit CV
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = {
        "Ridge": Ridge(alpha=100.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=50,
            max_features=0.5, n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.01,
            subsample=0.7, min_samples_leaf=50, max_features=0.5,
            random_state=42,
        ),
    }

    results = {}
    metrics_rows = []

    for name, model in models.items():
        use_scaled = name == "Ridge"
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test

        # Cross-validation
        cv_maes = []
        for cv_train_idx, cv_val_idx in tscv.split(Xtr):
            m_cv = type(model)(**model.get_params())
            m_cv.fit(Xtr[cv_train_idx], y_train_ret[cv_train_idx])
            val_pred_ret = m_cv.predict(Xtr[cv_val_idx])
            val_pred_price = train.iloc[cv_val_idx]["price"].values * (1 + val_pred_ret)
            val_actual_price = train.iloc[cv_val_idx]["target_price"].values
            cv_maes.append(mean_absolute_error(val_actual_price, val_pred_price))
        cv_mae = float(np.mean(cv_maes))
        cv_mae_std = float(np.std(cv_maes))

        model.fit(Xtr, y_train_ret)

        train_pred_ret = model.predict(Xtr)
        train_pred_price = today_train * (1 + train_pred_ret)
        train_mae = mean_absolute_error(train["target_price"].values, train_pred_price)

        test_pred_ret = model.predict(Xte)
        test_pred_price = today_test * (1 + test_pred_ret)
        test_mae = mean_absolute_error(y_test_price, test_pred_price)
        test_rmse = np.sqrt(mean_squared_error(y_test_price, test_pred_price))
        test_r2 = r2_score(y_test_price, test_pred_price)
        test_mape = np.mean(np.abs((y_test_price - test_pred_price) / y_test_price)) * 100
        ret_r2 = r2_score(y_test_ret, test_pred_ret)

        metrics_rows.append({
            "model": f"Daily {name}",
            "MAE": test_mae, "RMSE": test_rmse,
            "R2_price": test_r2, "MAPE_pct": test_mape,
            "Return_R2": ret_r2,
            "Train_MAE": train_mae,
            "Overfit_Ratio": train_mae / test_mae if test_mae > 0 else 0,
            "CV_MAE": cv_mae, "CV_MAE_std": cv_mae_std,
        })
        results[name] = {
            "model": model,
            "pred_price": test_pred_price,
            "pred_return": test_pred_ret,
            "metrics": {"MAE": test_mae, "RMSE": test_rmse, "R2": test_r2,
                         "MAPE": test_mape, "Return_R2": ret_r2},
            "train_metrics": {"MAE": train_mae},
            "cv_mae": cv_mae, "cv_mae_std": cv_mae_std,
        }
        print(f"  {name}: Test MAE={test_mae:.2f}, CV_MAE={cv_mae:.2f}+/-{cv_mae_std:.1f}, Ret R2={ret_r2:.4f}  |  Train MAE={train_mae:.2f}")

    return results, metrics_rows, train, test

# ---------------------------------------------------------------
# 6. Forecast using Monthly Fundamentals Model
# ---------------------------------------------------------------

def _build_monthly_row(balance_row, prev_prices, month, external_vals=None):
    """Build a feature row for the monthly model from balance + external data.
    No price momentum or moving averages -- prediction depends on
    supply/demand fundamentals, seasonality, and external market conditions."""
    row = {}
    for c in BALANCE_COLS:
        row[c] = balance_row.get(c, 0)

    row.update(compute_derived_ratios(
        prod=row.get("produccion", 0) or 0,
        imp=row.get("importaciones", 0) or 0,
        exp=row.get("exportaciones_totales", 0) or 0,
        cna=row.get("consumo_nacional_aparente", 1) or 1,
        inv_i=row.get("inventario_inicial", 0) or 0,
        inv_f=row.get("inventario_final", 0) or 0,
        ot=row.get("oferta_total", 1) or 1,
        dt_=row.get("demanda_total", 1) or 1,
    ))

    # Cyclical month encoding
    row["month_sin"] = np.sin(2 * np.pi * month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month / 12)

    # External market data
    if external_vals:
        for k in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"]:
            if k in external_vals:
                row[k] = external_vals[k]
        # Derived external features
        usd_mxn = external_vals.get("usd_mxn", 0)
        ice_no11 = external_vals.get("ice_no11", 0)
        if usd_mxn and len(prev_prices) >= 1:
            row["mx_sugar_usd"] = prev_prices[-1] / usd_mxn
        if ice_no11 and len(prev_prices) >= 1:
            row["premium_vs_world"] = prev_prices[-1] / ice_no11

    # prev_price still needed as reference for % change target
    if len(prev_prices) >= 1:
        row["prev_price"] = prev_prices[-1]
    return row


def project_external(external_df, target_months):
    """Project external variables for future months.
    Uses carry-forward of last known values (conservative).
    Scenario overrides can replace these values."""
    if external_df is None or external_df.empty:
        return [None] * len(target_months)

    ext_cols = [c for c in external_df.columns if c not in ("year", "month")]
    last_vals = {}
    for c in ext_cols:
        valid = external_df[c].dropna()
        if not valid.empty:
            last_vals[c] = float(valid.iloc[-1])

    if not last_vals:
        return [None] * len(target_months)

    result = []
    for y, m in target_months:
        # Check if we have actual data for this month
        match = external_df[(external_df["year"] == y) & (external_df["month"] == m)]
        if not match.empty:
            vals = {c: float(match[c].iloc[0]) if pd.notna(match[c].iloc[0]) else last_vals.get(c, 0)
                    for c in ext_cols}
        else:
            vals = dict(last_vals)
        result.append(vals)
    return result


def forecast_monthly(
    model, scaler, feature_cols, monthly_df,
    months_ahead=6, scenario=None, latest_price_date=None,
    latest_actual_price=None, feat_weights=None,
    balance_df=None,
    external_df=None,
    clip_bounds=None,
    residual_percentiles=None,
):
    """
    Forecast using the monthly fundamentals model.
    Model predicts month-over-month % change from balance features.

    Balance conditions for future months come from:
      1) Seasonal averages (historical mean for that calendar month)
      2) YoY trends (recent growth rates projected forward)
      3) Scenario overrides (user inputs from dashboard)
    """
    last_row = monthly_df.iloc[-1]
    last_year = int(last_row["year"])
    last_month = int(last_row["month"])

    if latest_actual_price is not None and latest_actual_price > 0:
        current_price = float(latest_actual_price)
    else:
        current_price = float(monthly_df["avg_price"].iloc[-1])

    prev_prices = monthly_df["avg_price"].tolist()

    if latest_price_date is not None:
        lpd = pd.Timestamp(latest_price_date)
        price_year, price_month = lpd.year, lpd.month
        skip_months = (price_year - last_year) * 12 + (price_month - last_month)
        total_months = skip_months + months_ahead
    else:
        skip_months = 0
        total_months = months_ahead

    # Build target month list and project balance conditions
    target_months = []
    for i in range(total_months):
        m = last_month + i + 1
        y = last_year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        target_months.append((y, m))

    if balance_df is not None and not balance_df.empty:
        projected = project_balance(balance_df, target_months)
    else:
        projected = [None] * total_months

    # Project external market data for future months
    projected_ext = project_external(external_df, target_months)

    forecasts = []

    for i, (y, m) in enumerate(target_months):
        # Start from projected seasonal + trend balance
        if projected[i] is not None:
            balance_vals = {k: v for k, v in projected[i].items()
                           if k not in ("year", "month_number")}
        else:
            balance_vals = {}
            for c in BALANCE_COLS + ["net_exports", "supply_demand_ratio",
                                      "demand_pressure", "excess_supply",
                                      "inventory_to_consumption", "inventory_months",
                                      "production_share"]:
                if c in last_row.index:
                    balance_vals[c] = last_row[c]

        # Apply scenario overrides on top of projected values
        if scenario:
            for k, v in scenario.items():
                balance_vals[k] = v
            balance_vals.update(compute_derived_ratios(
                prod=balance_vals.get("produccion", 0) or 0,
                imp=balance_vals.get("importaciones", 0) or 0,
                exp=balance_vals.get("exportaciones_totales", 0) or 0,
                cna=balance_vals.get("consumo_nacional_aparente", 1) or 1,
                inv_i=balance_vals.get("inventario_inicial", 0) or 0,
                inv_f=balance_vals.get("inventario_final", 0) or 0,
                ot=balance_vals.get("oferta_total", 1) or 1,
                dt_=balance_vals.get("demanda_total", 1) or 1,
            ))

        # Build external values for this month (projected + scenario overrides)
        ext_vals = projected_ext[i] if projected_ext[i] is not None else None
        if scenario and ext_vals:
            for k in EXTERNAL_FEATURES[:5]:  # usd_mxn, ice_no11, ice_no16, wti, brl_usd
                if k in scenario:
                    ext_vals[k] = scenario[k]
        elif scenario:
            ext_vals = {k: scenario[k] for k in EXTERNAL_FEATURES[:5] if k in scenario}
            ext_vals = ext_vals if ext_vals else None

        row = _build_monthly_row(balance_vals, prev_prices + [current_price], m, external_vals=ext_vals)
        row_df = pd.DataFrame([row])

        feats = row_df.reindex(columns=feature_cols, fill_value=0).values

        # EnsembleModel handles its own scaling internally
        if not getattr(model, "_is_ensemble", False):
            if scaler is not None:
                n_expected = getattr(scaler, "n_features_in_", None)
                if n_expected is not None:
                    if feats.shape[1] < n_expected:
                        pad = getattr(scaler, "mean_", None)
                        if pad is not None and len(pad) >= n_expected:
                            feats = np.hstack([feats, pad[feats.shape[1]:n_expected].reshape(1, -1)])
                        else:
                            feats = np.hstack([feats, np.zeros((feats.shape[0], n_expected - feats.shape[1]))])
                    elif feats.shape[1] > n_expected:
                        feats = feats[:, :n_expected]
                feats = scaler.transform(feats)

        pred_change = float(model.predict(feats)[0])
        # Data-driven clip bounds (fallback to +-15% if not provided)
        lo, hi = (clip_bounds if clip_bounds else (-0.15, 0.15))
        pred_change = np.clip(pred_change, lo, hi)
        next_price = current_price * (1 + pred_change)
        next_price = max(next_price, 50)

        prev_prices.append(next_price)
        current_price = next_price

        if i >= skip_months:
            step = i - skip_months + 1
            fc = {
                "year": y, "month": m,
                "predicted_price": next_price,
                "predicted_change": pred_change,
            }
            # Residual-based confidence intervals (growing with sqrt of step)
            if residual_percentiles:
                scale = np.sqrt(step)
                fc["ci_lower_80"] = next_price + residual_percentiles["p10"] * scale
                fc["ci_upper_80"] = next_price + residual_percentiles["p90"] * scale
                fc["ci_lower_50"] = next_price + residual_percentiles["p25"] * scale
                fc["ci_upper_50"] = next_price + residual_percentiles["p75"] * scale
            forecasts.append(fc)

    result = pd.DataFrame(forecasts)

    if result.empty:
        return pd.DataFrame(columns=["date", "predicted_price"]), result

    # Interpolate daily: linear between monthly midpoints for smooth curve
    lpd = pd.Timestamp(latest_price_date) if latest_price_date is not None else None
    anchor = float(latest_actual_price) if latest_actual_price else float(monthly_df["avg_price"].iloc[-1])

    anchor_points = []
    if lpd is not None:
        anchor_points.append((lpd + pd.tseries.offsets.BDay(1), anchor))

    for _, r in result.iterrows():
        mid = pd.Timestamp(year=int(r["year"]), month=int(r["month"]), day=15)
        anchor_points.append((mid, r["predicted_price"]))

    last_r = result.iloc[-1]
    end = pd.Timestamp(year=int(last_r["year"]), month=int(last_r["month"]), day=1) + pd.offsets.MonthEnd(0)
    anchor_points.append((end, last_r["predicted_price"]))

    all_bdays = pd.bdate_range(anchor_points[0][0], anchor_points[-1][0])
    a_dates = [a[0] for a in anchor_points]
    a_vals = [a[1] for a in anchor_points]
    a_nums = np.array([(d - a_dates[0]).days for d in a_dates], dtype=float)

    # Also interpolate CI columns if present
    has_ci = "ci_upper_80" in result.columns
    ci_cols = ["ci_lower_80", "ci_upper_80", "ci_lower_50", "ci_upper_50"]
    ci_anchors = {}
    if has_ci:
        for cc in ci_cols:
            pts = []
            if lpd is not None:
                pts.append(anchor)  # at forecast start, CI collapses to point
            for _, r in result.iterrows():
                pts.append(r[cc])
            pts.append(result.iloc[-1][cc])
            ci_anchors[cc] = pts

    daily_rows = []
    for d in all_bdays:
        d_num = (d - a_dates[0]).days
        row = {"date": d, "predicted_price": float(np.interp(d_num, a_nums, a_vals))}
        if has_ci:
            for cc in ci_cols:
                row[cc] = float(np.interp(d_num, a_nums, ci_anchors[cc]))
        daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)
    if lpd is not None and not daily_df.empty:
        daily_df = daily_df[daily_df["date"] > lpd].reset_index(drop=True)

    return daily_df, result


def forecast_future(
    model, df_feat, feature_cols,
    days=FORECAST_DAYS, scenario=None,
):
    """Kept for backward compat with dashboard scenario import."""
    return forecast_monthly(model, None, feature_cols,
                            pd.DataFrame(), months_ahead=3, scenario=scenario)[0]

# ---------------------------------------------------------------
# 7. Main Pipeline
# ---------------------------------------------------------------

def run_pipeline(product="estandar"):
    print(f"\n{'='*60}")
    print(f"  Sugar Price ML Pipeline v4 -- {product.upper()}")
    print(f"  Fundamentals + External: Price = f(Demand, Supply, Market)")
    print(f"{'='*60}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print("\n[1/10] Loading price data ...")
    raw = load_prices()
    print(f"  Total price rows: {len(raw)}")

    print("\n[2/10] Loading sugar balance data ...")
    balance = load_balance()
    print(f"  Monthly balance rows: {len(balance)}")
    balance.to_csv(OUTPUT_DIR / "balance_monthly.csv", index=False)

    # Load external market data
    print("\n[3/10] Loading external market data ...")
    try:
        from sources import load_external_monthly
        external_df = load_external_monthly()
    except ImportError:
        print("  [External] sources.py not found -- skipping external data")
        external_df = None
    except Exception as e:
        print(f"  [External] Error: {e} -- skipping external data")
        external_df = None

    if external_df is not None:
        print(f"  External data: {len(external_df)} monthly rows")
        ext_cols = [c for c in external_df.columns if c not in ("year", "month")]
        print(f"  Columns: {', '.join(ext_cols)}")
        external_df.to_csv(OUTPUT_DIR / "external_monthly.csv", index=False)
    else:
        print("  No external data available -- using balance-only features")

    # Stage 1: Monthly fundamentals model
    print(f"\n[4/10] Building monthly dataset (price = f(supply, demand, market)) ...")
    monthly_df = build_monthly_dataset(raw, balance, product, external_df=external_df)
    print(f"  Monthly observations: {len(monthly_df)}")

    print(f"\n[5/10] Training MONTHLY models (fundamentals -> price) ...")
    m_results, m_metrics, m_features, m_train, m_test, m_scaler, clip_bounds, residual_percentiles = train_monthly_model(monthly_df)

    best_monthly = min(m_results, key=lambda k: m_results[k]["metrics"]["MAE"])
    print(f"\n  >> Best monthly model: {best_monthly} (MAE={m_results[best_monthly]['metrics']['MAE']:.2f})")

    # Stage 2: Daily model
    print(f"\n[6/10] Preparing daily series ...")
    series = prepare_product_series(raw, product)
    print(f"  Rows before outlier removal: {len(series)}")

    print(f"\n[7/10] Removing outliers ...")
    series, n_outliers = remove_outliers(series)
    print(f"  Outliers winsorized: {n_outliers}")

    series = merge_balance_to_daily(series, balance)
    df_feat = engineer_daily_features(series)
    daily_features = get_feature_cols(df_feat, DAILY_FEATURE_COLS)
    print(f"  Daily features: {len(daily_features)}")

    print(f"\n[8/10] Training DAILY models (balance + momentum -> next-day) ...")
    d_results, d_metrics, d_train, d_test = train_daily_model(df_feat, daily_features)

    best_daily = min(d_results, key=lambda k: d_results[k]["metrics"]["MAE"])
    bm = d_results[best_daily]["metrics"]
    print(f"\n  >> Best daily model: {best_daily} (MAE={bm['MAE']:.2f}, MAPE={bm['MAPE']:.2f}%)")

    # ── Use the MONTHLY fundamentals model for forecasting ──
    latest_price_date = series["date"].max()
    print(f"\n[9/10] Forecasting with monthly fundamentals model ({best_monthly}) ...")
    print(f"       Price = f(Demand, Supply, Market)")
    print(f"       Latest price: {latest_price_date.strftime('%Y-%m-%d')}, forecast starts after this")
    best_m_model = m_results[best_monthly]["model"]
    best_m_scaler = m_results[best_monthly]["scaler"]
    latest_actual_price = float(series["price"].iloc[-1])
    print(f"       Last actual price: ${latest_actual_price:,.2f}")
    forecast_df, monthly_forecast = forecast_monthly(
        best_m_model, best_m_scaler, m_features, monthly_df,
        months_ahead=6, latest_price_date=latest_price_date,
        latest_actual_price=latest_actual_price,
        balance_df=balance,
        external_df=external_df,
        clip_bounds=clip_bounds,
        residual_percentiles=residual_percentiles,
    )
    print(f"  Monthly forecast (seasonal + YoY trend balance):")
    for _, row in monthly_forecast.iterrows():
        chg = row.get("predicted_change", 0) * 100
        print(f"    {int(row['year'])}-{int(row['month']):02d}: ${row['predicted_price']:,.2f} MXN ({chg:+.1f}%)")
    print(f"  Daily points: {len(forecast_df)}")
    if not forecast_df.empty:
        print(f"  Price range: {forecast_df['predicted_price'].min():.2f} - {forecast_df['predicted_price'].max():.2f}")

    # -- Save all outputs --
    prefix = product
    all_metrics = m_metrics + d_metrics
    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / f"{prefix}_metrics.csv", index=False)

    # Monthly predictions
    m_preds = m_test[["year", "month", "avg_price"]].copy()
    for name, res in m_results.items():
        m_preds[f"pred_{name.lower().replace(' ', '_')}"] = res["test_pred"]
    m_preds.to_csv(OUTPUT_DIR / f"{prefix}_monthly_predictions.csv", index=False)

    # Daily predictions
    preds_df = d_test[["date", "price"]].copy().rename(columns={"price": "actual_today"})
    preds_df["actual_target"] = d_test["target_price"].values
    for name, res in d_results.items():
        preds_df[name.lower().replace(" ", "_")] = res["pred_price"]
    preds_df.to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)

    # Feature importance
    fi_rows = []
    for stage, res_dict, feat_list in [
        ("monthly", m_results, m_features),
        ("daily", d_results, daily_features),
    ]:
        for name, res in res_dict.items():
            m = res["model"]
            if hasattr(m, "feature_importances_"):
                for feat, imp in zip(feat_list, m.feature_importances_):
                    fi_rows.append({"stage": stage, "model": name, "feature": feat, "importance": imp})
    if fi_rows:
        pd.DataFrame(fi_rows).to_csv(OUTPUT_DIR / f"{prefix}_feature_importance.csv", index=False)

    print(f"\n[10/10] Saving outputs ...")
    forecast_df.to_csv(OUTPUT_DIR / f"{prefix}_forecast.csv", index=False)
    monthly_forecast.to_csv(OUTPUT_DIR / f"{prefix}_monthly_forecast.csv", index=False)
    df_feat.to_csv(OUTPUT_DIR / f"{prefix}_featured.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / f"{prefix}_monthly_data.csv", index=False)

    with open(MODELS_DIR / f"{prefix}_best_model.pkl", "wb") as f:
        pickle.dump(best_m_model, f)
    if best_m_scaler is not None:
        with open(MODELS_DIR / f"{prefix}_scaler.pkl", "wb") as f:
            pickle.dump(best_m_scaler, f)
    with open(MODELS_DIR / f"{prefix}_monthly_model.pkl", "wb") as f:
        pickle.dump(best_m_model, f)
    with open(MODELS_DIR / f"{prefix}_clip_bounds.json", "w") as f:
        json.dump({"low": clip_bounds[0], "high": clip_bounds[1]}, f)
    with open(MODELS_DIR / f"{prefix}_residual_percentiles.json", "w") as f:
        json.dump(residual_percentiles, f, indent=2)
    with open(MODELS_DIR / f"{prefix}_feature_cols.json", "w") as f:
        json.dump(m_features, f)
    with open(MODELS_DIR / f"{prefix}_daily_feature_cols.json", "w") as f:
        json.dump(daily_features, f)

    latest_balance = {}
    for c in BALANCE_FEATURE_COLS:
        if c in df_feat.columns and df_feat[c].notna().any():
            latest_balance[c] = float(df_feat[c].dropna().iloc[-1])
    # Add external market values
    if external_df is not None and not external_df.empty:
        for c in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"]:
            if c in external_df.columns:
                valid = external_df[c].dropna()
                if not valid.empty:
                    latest_balance[c] = float(valid.iloc[-1])
    with open(OUTPUT_DIR / f"{prefix}_latest_balance.json", "w") as f:
        json.dump(latest_balance, f, indent=2)

    ensemble_weights = m_results.get("Ensemble", {}).get("weights", {})
    summary = {
        "product": product,
        "best_model": f"Monthly {best_monthly}" if best_monthly != "Ensemble" else "Monthly Ensemble",
        "best_monthly_model": f"Monthly {best_monthly}" if best_monthly != "Ensemble" else "Monthly Ensemble",
        "best_daily_model": f"Daily {best_daily}",
        "best_metrics": m_results[best_monthly]["metrics"],
        "train_metrics": m_results[best_monthly].get("train_metrics", {}),
        "monthly_metrics": m_results[best_monthly]["metrics"],
        "daily_metrics": d_results[best_daily]["metrics"],
        "train_size": len(m_train),
        "test_size": len(m_test),
        "daily_train_size": len(d_train),
        "daily_test_size": len(d_test),
        "outliers_removed": n_outliers,
        "forecast_days": FORECAST_DAYS,
        "forecast_months": 6,
        "date_range": [str(series["date"].min()), str(series["date"].max())],
        "all_models_metrics": all_metrics,
        "feature_cols": m_features,
        "monthly_features": m_features,
        "balance_features_used": [c for c in m_features if c in BALANCE_FEATURE_COLS],
        "latest_actual_price": latest_actual_price,
        "pricing_formula": "Price = f(Demand/Supply) where Supply=Prod+Imports+Inv, Demand=Consumption+Exports",
        "clip_bounds": list(clip_bounds),
        "residual_percentiles": residual_percentiles,
        "ensemble_weights": {k: round(v, 4) for k, v in ensemble_weights.items()},
    }
    with open(OUTPUT_DIR / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")
    return summary


def run_all():
    for product in ["estandar", "refinada"]:
        run_pipeline(product)


if __name__ == "__main__":
    run_all()
