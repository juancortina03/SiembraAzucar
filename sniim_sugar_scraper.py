"""
SNIIM Sugar Price Scraper - Mexico
Scrapes daily sugar prices (Estándar and Refinada) from economia-sniim.gob.mx
as far back as available (year 2000).
"""

import re
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.economia-sniim.gob.mx/AzucarMesPorDia.asp"

# prod: 155 = Azúcar Refinada, 156 = Azúcar Estándar
PRODUCTS = {
    155: "refinada",
    156: "estandar",
}

# Spanish month abbreviation to number
MES_ABBR = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}

# Year range available on SNIIM (from their dropdown)
START_YEAR = 2000
END_YEAR = 2026  # current year in the form
DELAY_SECONDS = 0.5  # polite delay between requests


def parse_day_header(text: str) -> Optional[tuple[int, int]]:
    """
    Parse a column header like 'Jue 2-Ene' or 'Mar 3-Feb' -> (day, month_number).
    Returns (day, month) or None if not a date column.
    """
    text = (text or "").strip().lower()
    # Match patterns like "2-ene", "3-feb", "31-dic"
    match = re.search(r"(\d{1,2})\s*[-]\s*([a-z]{3})\b", text)
    if not match:
        return None
    day = int(match.group(1))
    abbr = match.group(2).lower()
    month = MES_ABBR.get(abbr)
    if month is None:
        return None
    return (day, month)


def fetch_month_page(prod: int, year: int, month: int) -> Optional[bytes]:
    """Fetch one month page; returns raw bytes for correct encoding handling."""
    params = {
        "Cons": "D",
        "prod": prod,
        "dqMesMes": month,
        "dqAnioMes": year,
        "Formato": "Nor",
        "submit": "Ver Resultados",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    }
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"  Error fetching prod={prod} {year}-{month:02d}: {e}")
        return None


def extract_national_row_and_headers(soup: BeautifulSoup):
    """
    Find the main data table: the row 'Promedio Nacional del Precio Frecuente'
    and the header row with date columns (e.g. 'Mar 2-Ene').
    Returns (header_cells, national_row_cells) or (None, None).
    """
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        national_cells = None
        header_cells = None
        for i, tr in enumerate(rows):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            texts = [c.get_text().strip() for c in cells]
            first_text = " ".join(cells[0].get_text().split()).lower()
            if "promedio nacional" in first_text and "precio frecuente" in first_text:
                national_cells = texts
                # Find header row: the one with date-like column headers (before this row)
                for j in range(i - 1, -1, -1):
                    prev_cells = rows[j].find_all(["td", "th"])
                    if not prev_cells:
                        continue
                    prev_texts = [c.get_text().strip() for c in prev_cells]
                    date_count = sum(1 for t in prev_texts if parse_day_header(t) is not None)
                    if date_count >= 5:  # header has many date columns
                        header_cells = prev_texts
                        break
                break
        if national_cells is not None and header_cells is not None:
            return (header_cells, national_cells)
    return (None, None)


def scrape_month(prod: int, year: int, month: int) -> list[dict]:
    """
    Scrape one (product, year, month). Returns list of dicts with keys:
    date, price, product_type, year, month.
    """
    raw = fetch_month_page(prod, year, month)
    if not raw:
        return []
    # SNIIM pages are typically Latin-1
    try:
        html = raw.decode("iso-8859-1")
    except Exception:
        html = raw.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    header_cells, national_cells = extract_national_row_and_headers(soup)
    if not national_cells or not header_cells:
        return []
    product_name = PRODUCTS.get(prod, str(prod))
    out = []
    # Align by index: header_cells[0] is "Centros...", national_cells[0] is "Promedio Nacional..."
    # Last column is PromMes, so we skip first and last
    for j in range(1, min(len(header_cells), len(national_cells))):
        if j >= len(header_cells):
            break
        parsed = parse_day_header(header_cells[j])
        if parsed is None:
            continue
        day, month_num = parsed
        if month_num != month:
            continue
        price_str = national_cells[j].replace(",", "").strip()
        if not price_str or not re.match(r"^[\d.]+$", price_str.replace(",", "")):
            continue
        try:
            price = float(price_str)
        except ValueError:
            continue
        try:
            date = datetime(year, month_num, day)
        except ValueError:
            continue
        out.append({
            "date": date,
            "price": price,
            "product_type": product_name,
            "year": year,
            "month": month,
        })
    return out


def scrape_all(limit_years: Optional[tuple[int, int]] = None):
    """
    Scrape Estándar and Refinada for all years and months; return a single DataFrame.
    limit_years: optional (start_year, end_year) to limit range for testing.
    """
    rows = []
    start, end = (limit_years or (START_YEAR, END_YEAR))[0], (limit_years or (START_YEAR, END_YEAR))[1]
    for prod, name in PRODUCTS.items():
        print(f"Scraping {name} (prod={prod})...")
        for year in range(start, end + 1):
            for month in range(1, 13):
                now = datetime.now()
                if year > now.year or (year == now.year and month > now.month):
                    continue
                data = scrape_month(prod, year, month)
                if data:
                    rows.extend(data)
                    print(f"  {year}-{month:02d}: {len(data)} days")
                time.sleep(DELAY_SECONDS)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["product_type", "date"]).reset_index(drop=True)
    return df


def main():
    import sys
    print("SNIIM Sugar Price Scraper - Mexico")
    print("Sources: Azúcar Estándar (prod=156), Azúcar Refinada (prod=155)")
    # Optional: python sniim_sugar_scraper.py quick  -> only 2024-2025
    quick = "--quick" in sys.argv or "quick" in sys.argv
    if quick:
        print("Quick run: 2024-2025 only\n")
        df = scrape_all(limit_years=(2024, 2025))
    else:
        print(f"Year range: {START_YEAR}–{END_YEAR}\n")
        df = scrape_all()
    if df.empty:
        print("No data collected.")
        return
    df["date"] = pd.to_datetime(df["date"])
    # Deduplicate by (date, product_type) keeping first
    df = df.drop_duplicates(subset=["date", "product_type"], keep="first")
    print("\n--- DataFrame info ---")
    print(df.info())
    print("\n--- First rows ---")
    print(df.head(20))
    print("\n--- Last rows ---")
    print(df.tail(20))
    print("\n--- Summary by product_type ---")
    print(df.groupby("product_type")["price"].agg(["count", "min", "max", "mean"]))
    out_csv = "sniim_sugar_prices.csv"
    out_xlsx = "sniim_sugar_prices.xlsx"
    df.to_csv(out_csv, index=False, date_format="%Y-%m-%d")
    print(f"\nSaved to {out_csv}")
    try:
        df.to_excel(out_xlsx, index=False, sheet_name="Precios", engine="openpyxl")
        print(f"Saved to {out_xlsx} (downloadable Excel)")
    except ImportError:
        print("Install openpyxl for Excel output: pip install openpyxl")


if __name__ == "__main__":
    main()
