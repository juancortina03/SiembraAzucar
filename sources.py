"""
External Market Data -- FRED & Banxico APIs
============================================
Fetches US sugar futures, WTI crude oil, BRL/USD, and USD/MXN exchange rate.
Caches results to model_results/cache/ for offline operation.
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd

CACHE_DIR = Path("model_results/cache")


# ---------------------------------------------------------------
# Banxico SIE API -- USD/MXN FIX rate
# ---------------------------------------------------------------

def fetch_banxico_usd_mxn(start: str, end: str, token: str = None) -> pd.DataFrame:
    """Fetch daily USD/MXN FIX from Banxico SIE API (series SF43718)."""
    token = token or os.getenv("BANXICO_TOKEN", "")
    if not token:
        return pd.DataFrame()

    import requests

    serie = "SF43718"
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{serie}/datos/{start}/{end}"
    headers = {"Bmx-Token": token}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [Banxico] Error fetching USD/MXN: {e}")
        return pd.DataFrame()

    series_data = data.get("bmx", {}).get("series", [{}])[0].get("datos", [])
    if not series_data:
        print("  [Banxico] No data returned")
        return pd.DataFrame()

    rows = []
    for d in series_data:
        try:
            date = pd.to_datetime(d["fecha"], format="%d/%m/%Y")
            val = float(d["dato"].replace(",", ""))
            rows.append({"date": date, "usd_mxn": val})
        except (ValueError, KeyError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------
# FRED API -- Sugar futures, WTI, BRL/USD
# ---------------------------------------------------------------

FRED_SERIES = {
    "ice_no11": "PSUGAISAUSDM",   # ICE Sugar No.11 (world, cents/lb)
    "ice_no16": "PSUGAUSAUSDM",   # ICE Sugar No.16 (US domestic, cents/lb)
    "wti": "DCOILWTICO",          # WTI Crude Oil (USD/bbl)
    "brl_usd": "DEXBZUS",         # BRL/USD exchange rate
}


def fetch_fred_series(start: str, end: str, api_key: str = None,
                      monthly: bool = True) -> pd.DataFrame:
    """Fetch sugar, oil, and FX series from FRED."""
    api_key = api_key or os.getenv("FRED_API_KEY", "")
    if not api_key:
        return pd.DataFrame()

    try:
        from fredapi import Fred
    except ImportError:
        print("  [FRED] fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()

    fred = Fred(api_key=api_key)
    frames = {}

    for name, series_id in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s = s.dropna()
            if monthly and not s.empty:
                s = s.resample("MS").mean().dropna()
            frames[name] = s
        except Exception as e:
            print(f"  [FRED] Error fetching {name} ({series_id}): {e}")
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    return df


# ---------------------------------------------------------------
# Orchestrator -- merge all external data into monthly DataFrame
# ---------------------------------------------------------------

def load_external_monthly(start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """
    Fetch all external data, resample to monthly, merge, and cache.
    Returns DataFrame with columns: [year, month, usd_mxn, ice_no11, ice_no16, wti, brl_usd]
    Returns None if no API keys and no cache available.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "external_monthly.csv"

    # Try fetching fresh data
    banxico_df = fetch_banxico_usd_mxn(start, end)
    fred_df = fetch_fred_series(start, end)

    has_banxico = not banxico_df.empty
    has_fred = not fred_df.empty

    if not has_banxico and not has_fred:
        # No API data -- try cache
        if cache_path.exists():
            print("  [External] Using cached data (no API keys or API error)")
            df = pd.read_csv(cache_path)
            return df if not df.empty else None
        print("  [External] No API keys and no cache -- external data skipped")
        return None

    # Resample Banxico daily -> monthly
    if has_banxico:
        banxico_df["date"] = pd.to_datetime(banxico_df["date"])
        banxico_monthly = (
            banxico_df.set_index("date")
            .resample("MS")
            .mean()
            .dropna()
            .reset_index()
        )
        banxico_monthly["year"] = banxico_monthly["date"].dt.year
        banxico_monthly["month"] = banxico_monthly["date"].dt.month
        banxico_monthly = banxico_monthly[["year", "month", "usd_mxn"]]
    else:
        banxico_monthly = pd.DataFrame(columns=["year", "month", "usd_mxn"])

    # FRED is already monthly from fetch_fred_series
    if has_fred:
        fred_df["date"] = pd.to_datetime(fred_df["date"])
        fred_df["year"] = fred_df["date"].dt.year
        fred_df["month"] = fred_df["date"].dt.month
        fred_monthly = fred_df.drop(columns=["date"])
    else:
        fred_monthly = pd.DataFrame(columns=["year", "month"])

    # Merge
    if has_banxico and has_fred:
        merged = banxico_monthly.merge(fred_monthly, on=["year", "month"], how="outer")
    elif has_banxico:
        merged = banxico_monthly
    else:
        merged = fred_monthly

    merged = merged.sort_values(["year", "month"]).reset_index(drop=True)

    # Forward-fill gaps
    for col in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Cache
    merged.to_csv(cache_path, index=False)
    print(f"  [External] Cached {len(merged)} monthly rows to {cache_path}")

    return merged if not merged.empty else None
