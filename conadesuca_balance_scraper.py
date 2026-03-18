"""
CONADESUCA Balance Nacional de Azúcar Scraper - Mexico

Scrapes the index of monthly sugar balance PDFs from gob.mx/conadesuca for
multiple crop cycles (e.g. 2020-2021 through 2024-2025). Optionally downloads
PDFs and extracts balance tables (supply, demand, inventories) for ML-ready
monthly logs. Data is intended for use with price prediction models (e.g. with
SNIIM price series).

Sources (cycle-dependent URL paths):
  - Standard: .../documentos/politica-comercial-balance-nacional-de-azucar-{start}-{end}
  - 2017/18: .../es/documentos/politica-comercial-balance-nacional-de-azucar-2017-2018
  - 2016/17, 2015/16: .../documentos/politica-comercial-balances-azucareros-zafra-{start}-{end}
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Cycle: year_start-year_end (e.g. 2024-2025 = Oct 2024 to Sep 2025)
# Some cycles use different URL paths on gob.mx
def _cycle_url(cycle_start: int, cycle_end: int) -> str:
    base = "https://www.gob.mx/conadesuca"
    if (cycle_start, cycle_end) == (2015, 2016):
        return f"{base}/documentos/politica-comercial-balances-azucareros-zafra-2015-2016"
    if (cycle_start, cycle_end) == (2016, 2017):
        return f"{base}/documentos/politica-comercial-balances-azucareros-zafra-2016-2017?state=published"
    if (cycle_start, cycle_end) == (2017, 2018):
        return f"{base}/es/documentos/politica-comercial-balance-nacional-de-azucar-2017-2018"
    # Standard: 2018-2019 through 2024-2025
    return f"{base}/documentos/politica-comercial-balance-nacional-de-azucar-{cycle_start}-{cycle_end}?state=published"

# Spanish month name (as on the site) -> (month_number, cycle_year_offset: 0 = start year, 1 = end year)
# Oct–Sep: Oct–Dec in start year, Jan–Sep in end year
MES_ESP = {
    "octubre": (10, 0),
    "noviembre": (11, 0),
    "diciembre": (12, 0),
    "enero": (1, 1),
    "febrero": (2, 1),
    "marzo": (3, 1),
    "abril": (4, 1),
    "mayo": (5, 1),
    "junio": (6, 1),
    "julio": (7, 1),
    "agosto": (8, 1),
    "septiembre": (9, 1),
}

# Scrape cycles from 2015-2016 up to 2024-2025 (URLs: 15/16, 16/17, 17/18, 19/20, 23/24, etc.)
CYCLE_START_YEAR = 2015
CYCLE_END_YEAR = 2024
DELAY_SECONDS = 0.8
PDF_DIR = Path("conadesuca_balance_pdfs")
INDEX_CSV = "conadesuca_balance_index.csv"
INDEX_XLSX = "conadesuca_balance_index.xlsx"
BALANCE_BY_MONTH_CSV = "conadesuca_balance_by_month.csv"
BALANCE_BY_MONTH_XLSX = "conadesuca_balance_by_month.xlsx"

# Fallback: known PDF URLs when gob.mx returns a challenge (from CONADESUCA index pages)
# Format: (cycle_start, cycle_end): [ (month_name_es, year, file_id, filename_suffix), ... ]
_FALLBACK_PDFS = {
    (2024, 2025): [
        ("octubre", 2024, "1029859", "1__Balance_Azucar_octubre_2024_20251017.pdf"),
        ("noviembre", 2024, "1029860", "2__Balance_Azucar_noviembre_2024_20251017.pdf"),
        ("diciembre", 2024, "1029861", "3__Balance_Azucar_diciembre_2024_20251017.pdf"),
        ("enero", 2025, "1029862", "4__Balance_Azucar_enero_2025_20251017.pdf"),
        ("febrero", 2025, "1029864", "5__Balance_Azucar_febrero_2025_20251017.pdf"),
        ("marzo", 2025, "1029863", "6__Balance_Azucar_marzo_2025_20251017.pdf"),
        ("abril", 2025, "1029865", "7__Balance_Azucar_abril_2025_20251017.pdf"),
        ("mayo", 2025, "1029866", "8__Balance_Azucar_mayo_2025_20251017.pdf"),
        ("junio", 2025, "1029867", "9__Balance_Azucar_junio_2025_20251017.pdf"),
        ("julio", 2025, "1029868", "10__Balance_Azucar_julio_2025_20251017.pdf"),
        ("agosto", 2025, "1029869", "11__Balance_Azucar_agosto_2025_20251017.pdf"),
        ("septiembre", 2025, "1029870", "12__Balance_Azucar_septiembre_2025_20251017.pdf"),
    ],
    (2023, 2024): [],  # https://www.gob.mx/conadesuca/documentos/...-2023-2024 — add PDF IDs when available
    (2019, 2020): [],  # ...-2019-2020
    (2018, 2019): [],  # standard URL
    (2017, 2018): [],  # /es/documentos/...-2017-2018
    (2016, 2017): [],  # ...balances-azucareros-zafra-2016-2017
    (2015, 2016): [],  # ...balances-azucareros-zafra-2015-2016
    (2022, 2023): [
        ("octubre", 2022, "863216", "1._Balance_Az_car_octubre_2022_20231013.pdf"),
        ("noviembre", 2022, "863217", "2._Balance_Az_car_noviembre_2022_20231013.pdf"),
        ("diciembre", 2022, "863218", "3._Balance_Az_car_diciembre_2022_20231013.pdf"),
        ("enero", 2023, "863219", "4._Balance_Az_car_enero_2023_20231013.pdf"),
        ("febrero", 2023, "863220", "5._Balance_Az_car_febrero_2023_20231013.pdf"),
        ("marzo", 2023, "863221", "6._Balance_Az_car_marzo_2023_20231013.pdf"),
        ("abril", 2023, "863222", "7._Balance_Az_car_abril_2023_20231013.pdf"),
        ("mayo", 2023, "863223", "8._Balance_Az_car_mayo_2023_20231013.pdf"),
        ("junio", 2023, "863224", "9._Balance_Az_car_junio_2023_20231013.pdf"),
        ("julio", 2023, "863225", "10._Balance_Az_car_julio_2023_20231013.pdf"),
        ("agosto", 2023, "863226", "11._Balance_Az_car_agosto_2023_20231013.pdf"),
        ("septiembre", 2023, "863227", "12._Balance_Az_car_septiembre_2023_20231013.pdf"),
    ],
    (2021, 2022): [
        ("octubre", 2021, "830334", "1._Balance_Az_car_octubre_2021_20230605.pdf"),
        ("noviembre", 2021, "830335", "2._Balance_Az_car_noviembre_2021_20230605.pdf"),
        ("diciembre", 2021, "830336", "3._Balance_Az_car_diciembre_2021_20230605.pdf"),
        ("enero", 2022, "830337", "4._Balance_Az_car_enero_2022_20230605.pdf"),
        ("febrero", 2022, "830338", "5._Balance_Az_car_febrero_2022_20230605.pdf"),
        ("marzo", 2022, "830339", "6._Balance_Az_car_marzo_2022_20230605.pdf"),
        ("abril", 2022, "830340", "7._Balance_Az_car_abril_2022_20230605.pdf"),
        ("mayo", 2022, "830341", "8._Balance_Az_car_mayo_2022_20230605.pdf"),
        ("junio", 2022, "830342", "9._Balance_Az_car_junio_2022_20230605.pdf"),
        ("julio", 2022, "830343", "10._Balance_Az_car_julio_2022_20230605.pdf"),
        ("agosto", 2022, "830344", "11._Balance_Az_car_agosto_2022_20230605.pdf"),
        ("septiembre", 2022, "830345", "12._Balance_Az_car_septiembre_2022_20230605.pdf"),
    ],
    (2020, 2021): [
        ("octubre", 2020, "675713", "1._Balance_Az_car_octubre_2020c.pdf"),
        ("noviembre", 2020, "675712", "2._Balance_Az_car_noviembre_2020c.pdf"),
        ("diciembre", 2020, "675714", "3._Balance_Az_car_diciembre_2020c.pdf"),
        ("enero", 2021, "675715", "4._Balance_Az_car_enero_2021c.pdf"),
        ("febrero", 2021, "675716", "5._Balance_Az_car_febrero_2021c.pdf"),
        ("marzo", 2021, "675717", "6._Balance_Az_car_marzo_2021c.pdf"),
        ("abril", 2021, "675718", "7._Balance_Az_car_abril_2021c.pdf"),
        ("mayo", 2021, "675719", "8._Balance_Az_car_mayo_2021c.pdf"),
        ("junio", 2021, "675720", "9._Balance_Az_car_junio_2021c.pdf"),
        ("julio", 2021, "675721", "10._Balance_Az_car_julio_2021c.pdf"),
        ("agosto", 2021, "675722", "11._Balance_Az_car_agosto_2021c.pdf"),
        ("septiembre", 2021, "675723", "12._Balance_Az_car_septiembre_2021c.pdf"),
    ],
}


def _fallback_links_for_cycle(cycle_start: int, cycle_end: int) -> list[dict]:
    """Return list of balance link dicts from fallback data when live scrape fails."""
    key = (cycle_start, cycle_end)
    if key not in _FALLBACK_PDFS or not _FALLBACK_PDFS[key]:
        return []
    base = "https://www.gob.mx/cms/uploads/attachment/file"
    out = []
    for month_name_es, year, file_id, filename in _FALLBACK_PDFS[key]:
        month_num = MES_ESP[month_name_es][0]
        out.append({
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "month_name_es": month_name_es,
            "month_number": month_num,
            "year": year,
            "pdf_url": f"{base}/{file_id}/{filename}",
            "label": f"{month_name_es}_{year}",
        })
    return out


def _headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    }


def fetch_cycle_page(cycle_start: int, cycle_end: int) -> Optional[str]:
    """Fetch the balance index page for one cycle (plain requests). Returns HTML or None."""
    url = _cycle_url(cycle_start, cycle_end)
    try:
        r = requests.get(url, headers=_headers(), timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except Exception as e:
        print(f"  Error fetching {cycle_start}-{cycle_end}: {e}")
        return None


def fetch_cycle_page_browser(cycle_start: int, cycle_end: int) -> Optional[str]:
    """Fetch the balance index page using a real browser (Playwright). Use when server returns a challenge."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Install playwright: pip install playwright && playwright install chromium")
        return None
    url = _cycle_url(cycle_start, cycle_end)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)  # allow any late content
            html = page.content()
            browser.close()
        return html
    except Exception as e:
        print(f"  Error (browser) {cycle_start}-{cycle_end}: {e}")
        return None


def parse_balance_links(html: str, cycle_start: int, cycle_end: int) -> list[dict]:
    """
    Parse HTML for links like 'Balance nacional de azúcar octubre 2024'.
    Also matches PDF hrefs with Balance_Azucar_<month>_<year> in the path.
    Returns list of dicts: cycle_start, cycle_end, month_name_es, month_number, year, pdf_url, label.
    """
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    out = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href.endswith(".pdf"):
            continue
        # Normalize URL first
        if not href.startswith("http"):
            href_norm = "https://www.gob.mx" + (href if href.startswith("/") else "/" + href)
        else:
            href_norm = href
        text = " ".join(a.get_text().split()).strip().lower()
        month_name_es = None
        year = None
        # 1) From link text: "balance nacional de azúcar octubre 2024"
        match = re.search(
            r"balance\s+nacional\s+de\s+az[uú]car\s+(\w+)\s+(\d{4})",
            text,
            re.IGNORECASE,
        )
        if match:
            month_name_es = match.group(1).lower()
            year = int(match.group(2))
        # 2) From URL path: .../Balance_Azucar_octubre_2024_... or Balance_Az_car_noviembre_2022_...
        if month_name_es is None or year is None:
            url_match = re.search(
                r"Balance_Az[u_]?car[_\-](\w+)[_\-](\d{4})",
                href_norm,
                re.IGNORECASE,
            )
            if url_match:
                month_name_es = month_name_es or url_match.group(1).lower()
                year = year or int(url_match.group(2))
        if not month_name_es or year is None or month_name_es not in MES_ESP:
            continue
        if not (cycle_start <= year <= cycle_end):
            continue
        key = (year, month_name_es)
        if key in seen:
            continue
        seen.add(key)
        month_num, _ = MES_ESP[month_name_es]
        out.append({
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "month_name_es": month_name_es,
            "month_number": month_num,
            "year": year,
            "pdf_url": href_norm,
            "label": f"{month_name_es}_{year}",
        })
    return out


def scrape_all_cycles(
    cycle_start_year: int = CYCLE_START_YEAR,
    cycle_end_year: int = CYCLE_END_YEAR,
    use_browser: bool = False,
) -> pd.DataFrame:
    """
    Scrape index pages for each cycle; return DataFrame with one row per month
    (columns: cycle_start, cycle_end, month_name_es, month_number, year, pdf_url, label).
    use_browser: if True, fetch with Playwright (needed when gob.mx returns a challenge).
    """
    rows = []
    for c_start in range(cycle_start_year, cycle_end_year + 1):
        c_end = c_start + 1
        print(f"Cycle {c_start}-{c_end} ...")
        html = fetch_cycle_page_browser(c_start, c_end) if use_browser else fetch_cycle_page(c_start, c_end)
        if not html:
            continue
        # If we got a challenge page (no PDFs), retry once with browser
        if not use_browser and ".pdf" not in html and "Challenge" in html:
            print("  Challenge page detected, retrying with browser...")
            html = fetch_cycle_page_browser(c_start, c_end)
        if not html:
            continue
        links = parse_balance_links(html, c_start, c_end)
        if not links:
            links = _fallback_links_for_cycle(c_start, c_end)
            if links:
                print(f"  Using fallback index for {c_start}-{c_end} ({len(links)} months)")
            else:
                print(f"  No balance links found for {c_start}-{c_end}")
                continue
        for rec in links:
            rows.append(rec)
        print(f"  Found {len(links)} months")
        time.sleep(DELAY_SECONDS)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "month_number"]).reset_index(drop=True)
    return df


def download_pdfs(index_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Download each PDF in index_df into out_dir. Returns list of saved paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for _, row in index_df.iterrows():
        fname = f"balance_{row['year']}_{row['month_number']:02d}_{row['month_name_es']}.pdf"
        path = out_dir / fname
        if path.exists():
            saved.append(path)
            continue
        try:
            r = requests.get(row["pdf_url"], headers=_headers(), timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
            saved.append(path)
            print(f"  Downloaded {fname}")
        except Exception as e:
            print(f"  Failed {fname}: {e}")
        time.sleep(DELAY_SECONDS)
    return saved


def extract_balance_from_pdf(pdf_path: Path) -> Optional[dict]:
    """
    Extract balance table from one CONADESUCA PDF. Returns a dict with
    keys like inventario_inicial, produccion_nacional, importaciones, oferta_total,
    exportaciones, ventas_immex, consumo_nacional_aparente, demanda_total,
    inventario_final, inventario_optimo (values in tonnes where applicable).
    Returns None if pdfplumber not installed or extraction fails.
    """
    try:
        import pdfplumber
    except ImportError:
        return None
    out = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue
            full_text = (page.extract_text() or "").lower()
            for table in tables:
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    label = (row[0] or "").strip().lower()
                    # Try to get numeric value from last column or second column
                    val = None
                    for cell in reversed(row[1:] if len(row) > 1 else []):
                        s = (cell or "").strip().replace(",", "").replace(" ", "")
                        if re.match(r"^-?\d+\.?\d*$", s):
                            try:
                                val = float(s)
                                break
                            except ValueError:
                                pass
                    if val is None:
                        continue
                    if "inventario inicial" in label or "inicial" in label and "inventario" in label:
                        out["inventario_inicial"] = val
                    elif "producci" in label and "nacional" in label:
                        out["produccion_nacional"] = val
                    elif "importacion" in label:
                        out["importaciones"] = val
                    elif "oferta total" in label or "oferta total" in label.replace("ó", "o"):
                        out["oferta_total"] = val
                    elif "exportacion" in label:
                        out["exportaciones"] = val
                    elif "immex" in label or "maquiladora" in label:
                        out["ventas_immex"] = val
                    elif "consumo" in label and "nacional" in label:
                        out["consumo_nacional_aparente"] = val
                    elif "demanda total" in label or "demanda total" in label.replace("á", "a"):
                        out["demanda_total"] = val
                    elif "inventario final" in label or ("final" in label and "inventario" in label):
                        out["inventario_final"] = val
                    elif "óptimo" in label or "optimo" in label:
                        out["inventario_optimo"] = val
    return out if out else None


def build_balance_by_month(
    index_df: pd.DataFrame,
    pdf_dir: Path = PDF_DIR,
) -> pd.DataFrame:
    """
    For each row in index_df, load PDF from pdf_dir (or download if missing),
    extract balance table, and return one row per month with year, month_number,
    cycle_start, cycle_end, and extracted numeric columns.
    """
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        print("Install pdfplumber to extract balance data: pip install pdfplumber")
        return pd.DataFrame()
    pdf_dir = Path(pdf_dir)
    if not index_df.empty and not pdf_dir.exists():
        print("Downloading PDFs first...")
        download_pdfs(index_df, pdf_dir)
    rows = []
    for _, row in index_df.iterrows():
        fname = f"balance_{row['year']}_{row['month_number']:02d}_{row['month_name_es']}.pdf"
        path = pdf_dir / fname
        if not path.exists():
            continue
        data = extract_balance_from_pdf(path)
        if not data:
            continue
        rec = {
            "year": int(row["year"]),
            "month_number": int(row["month_number"]),
            "month_name_es": row["month_name_es"],
            "cycle_start": int(row["cycle_start"]),
            "cycle_end": int(row["cycle_end"]),
            "month_start_date": datetime(int(row["year"]), int(row["month_number"]), 1).strftime("%Y-%m-%d"),
        }
        rec.update(data)
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort for consistent monthly logs
    df = df.sort_values(["year", "month_number"]).reset_index(drop=True)
    return df


def main():
    import sys
    print("CONADESUCA Balance Nacional de Azúcar Scraper")
    print("Index URL pattern: .../politica-comercial-balance-nacional-de-azucar-{cycle_start}-{cycle_end}\n")
    download = "--download" in sys.argv or "download" in sys.argv
    extract = "--extract" in sys.argv or "extract" in sys.argv
    quick = "--quick" in sys.argv or "quick" in sys.argv
    debug_html = "--debug" in sys.argv or "debug" in sys.argv
    use_browser = "--browser" in sys.argv or "browser" in sys.argv
    if debug_html:
        html = fetch_cycle_page_browser(2024, 2025) if use_browser else fetch_cycle_page(2024, 2025)
        if html:
            Path("_debug_conadesuca_2024_2025.html").write_text(html, encoding="utf-8")
            print("Saved _debug_conadesuca_2024_2025.html")
            print(f"  .pdf occurrences: {html.count('.pdf')}")
        return
    if quick:
        index_df = scrape_all_cycles(cycle_start_year=2023, cycle_end_year=2024, use_browser=use_browser)
    else:
        index_df = scrape_all_cycles(use_browser=use_browser)
    if index_df.empty:
        print("No index data collected.")
        return
    index_df.to_csv(INDEX_CSV, index=False)
    print(f"\nIndex saved to {INDEX_CSV} ({len(index_df)} months)")
    try:
        index_df.to_excel(INDEX_XLSX, index=False, sheet_name="Balance index", engine="openpyxl")
        print(f"Index saved to {INDEX_XLSX} (downloadable Excel)")
    except ImportError:
        print("Install openpyxl for Excel output: pip install openpyxl")
    if download:
        print("\nDownloading PDFs...")
        download_pdfs(index_df, PDF_DIR)
    if extract:
        print("\nExtracting balance data from PDFs...")
        balance_df = build_balance_by_month(index_df, PDF_DIR)
        if not balance_df.empty:
            balance_df.to_csv(BALANCE_BY_MONTH_CSV, index=False)
            print(f"Balance-by-month saved to {BALANCE_BY_MONTH_CSV}")
            try:
                balance_df.to_excel(BALANCE_BY_MONTH_XLSX, index=False, sheet_name="Balance por mes", engine="openpyxl")
                print(f"Balance-by-month saved to {BALANCE_BY_MONTH_XLSX} (downloadable Excel)")
            except ImportError:
                pass
            print(balance_df.head(10))
        else:
            print("No balance data extracted (install pdfplumber and ensure PDFs are present).")


if __name__ == "__main__":
    main()
