"""
CONADESUCA Política Comercial — Full Document Scraper
=====================================================
Scrapes every document link from the CONADESUCA "Política Comercial ver más"
landing page:
  https://www.gob.mx/conadesuca/documentos/documentos-politica-comercial-ver-mas

Two link types live on this page:
  1. Direct PDF links  (e.g. Exportaciones, IMMEX, Lineamientos, Acuerdos …)
  2. Intermediate document-collection pages that themselves contain many PDF
     links (e.g. Balance Nacional de Azúcar per-cycle, Reporte Semanal …).

This scraper:
  • Parses the landing page and classifies every link into a section/category.
  • Follows intermediate gob.mx document pages to discover nested PDFs.
  • Builds a master index CSV with one row per document.
  • Optionally downloads all PDFs to a local directory.
  • Optionally extracts numeric tables from downloaded PDFs (via pdfplumber)
    and writes an ML-ready CSV with one row per extracted table row.

Outputs:
  politica_comercial_index.csv       — master document index
  politica_comercial_pdfs/           — downloaded PDFs  (with --download)
  politica_comercial_extracted.csv   — extracted table data (with --extract)

Usage:
  python conadesuca_politica_comercial_scraper.py             # index only
  python conadesuca_politica_comercial_scraper.py quick       # index, fewer sections
  python conadesuca_politica_comercial_scraper.py download    # + download PDFs
  python conadesuca_politica_comercial_scraper.py extract     # + extract table data
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANDING_URL = (
    "https://www.gob.mx/conadesuca/documentos/"
    "documentos-politica-comercial-ver-mas"
)
GOB_MX = "https://www.gob.mx"
DELAY = 1.0  # seconds between page loads
PDF_DIR = Path("politica_comercial_pdfs")
INDEX_CSV = "politica_comercial_index.csv"
INDEX_XLSX = "politica_comercial_index.xlsx"
EXTRACTED_CSV = "politica_comercial_extracted.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
}

MONTH_ES = {
    "octubre": 10, "noviembre": 11, "diciembre": 12,
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
}

# ---------------------------------------------------------------------------
# Networking: Playwright browser with cookie reuse
# ---------------------------------------------------------------------------
# gob.mx uses a JS challenge (Imperva/Incapsula) that blocks plain requests.
# We use a single Playwright browser instance that solves the challenge on the
# first navigation, then reuses the session cookies for all subsequent pages.

_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = {runtime: {}};
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5]
});
Object.defineProperty(navigator, 'languages', {
    get: () => ['es-MX', 'es', 'en']
});
"""

_browser_ctx: dict = {
    "pw": None,
    "browser": None,
    "context": None,
    "page": None,
}


def _ensure_browser():
    """Launch Playwright Chromium once (new headless + stealth) and keep alive."""
    if _browser_ctx["page"] is not None:
        return
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--headless=new",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    )
    context = browser.new_context(locale="es-MX")
    page = context.new_page()
    page.add_init_script(_STEALTH_JS)
    _browser_ctx.update(pw=pw, browser=browser, context=context, page=page)


def _close_browser():
    if _browser_ctx["browser"]:
        try:
            _browser_ctx["browser"].close()
        except Exception:
            pass
    if _browser_ctx["pw"]:
        try:
            _browser_ctx["pw"].stop()
        except Exception:
            pass
    _browser_ctx.update(pw=None, browser=None, context=None, page=None)


def _wait_past_challenge(page, max_wait_s: int = 45):
    """Poll until the page title is no longer 'Challenge Validation'."""
    waited = 0
    interval = 2000
    while waited < max_wait_s * 1000:
        title = page.title() or ""
        if "Challenge" not in title:
            return True
        page.wait_for_timeout(interval)
        waited += interval
    return False


def _fetch_html(url: str) -> Optional[str]:
    """Fetch a page via Playwright, solving any JS challenge automatically."""
    _ensure_browser()
    page = _browser_ctx["page"]
    try:
        page.goto(url, timeout=60000)
        page.wait_for_timeout(2000)
        if not _wait_past_challenge(page):
            print(f"  [timeout] Challenge did not resolve for {url[:80]}")
            return None
        page.wait_for_timeout(1000)
        return page.content()
    except Exception as e:
        print(f"  [browser error] {url[:90]} -> {e}")
        return None


def _soup(url: str) -> Optional[BeautifulSoup]:
    html = _fetch_html(url)
    if html is None:
        return None
    return BeautifulSoup(html, "lxml")


def _get(url: str, timeout: int = 30) -> Optional[requests.Response]:
    """Plain HTTP GET — used only for PDF downloads where no JS challenge."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r
    except Exception as e:
        print(f"  [GET error] {url[:90]} -> {e}")
        return None


# ---------------------------------------------------------------------------
# Parse the landing page into sections
# ---------------------------------------------------------------------------

def _normalise_url(href: str) -> str:
    """Turn a relative or protocol-relative href into a full URL."""
    href = href.strip()
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return GOB_MX + href
    if not href.startswith("http"):
        return GOB_MX + "/" + href
    return href


_CYCLE_RE = re.compile(r"(?:ciclo|zafra)\s*(\d{4})\s*/\s*(\d{4})", re.I)
_YEAR_RE = re.compile(r"(\d{4})\s*[-/]\s*(\d{4})")
_MONTH_RE = re.compile(
    r"(octubre|noviembre|diciembre|enero|febrero|marzo|"
    r"abril|mayo|junio|julio|agosto|septiembre)\s+(\d{4})",
    re.I,
)


def _parse_cycle(text: str, url: str) -> tuple[Optional[int], Optional[int]]:
    """Try to extract cycle_start / cycle_end from link text or URL."""
    for src in (text, url):
        m = _CYCLE_RE.search(src) or _YEAR_RE.search(src)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_month_year(text: str, url: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Try to extract (month_number, year, month_name_es)."""
    for src in (text, url):
        m = _MONTH_RE.search(src)
        if m:
            name = m.group(1).lower()
            return MONTH_ES.get(name), int(m.group(2)), name
    return None, None, None


def parse_landing_page(soup: BeautifulSoup) -> list[dict]:
    """
    Walk every <h4>/<strong> section on the landing page.  For each <a> beneath
    it, record the section name, link text, URL, and whether the URL points
    directly to a PDF or to an intermediate document page.
    """
    records: list[dict] = []
    current_section = "unknown"

    article = (
        soup.find("div", class_="col-article-body")
        or soup.find("div", class_="article-body")
        or soup
    )

    for el in article.descendants:
        if not isinstance(el, Tag):
            continue

        if el.name in ("h4", "strong"):
            txt = " ".join(el.get_text().split()).strip()
            if len(txt) > 3:
                current_section = txt

        if el.name != "a" or not el.get("href"):
            continue

        href = _normalise_url(el["href"])
        text = " ".join(el.get_text().split()).strip()
        if not text or len(text) < 4:
            continue

        is_pdf = href.lower().endswith(".pdf")
        cycle_start, cycle_end = _parse_cycle(text, href)
        month_num, year, month_name = _parse_month_year(text, href)

        records.append({
            "section": current_section,
            "link_text": text,
            "url": href,
            "is_pdf": is_pdf,
            "cycle_start": cycle_start,
            "cycle_end": cycle_end,
            "month_number": month_num,
            "year": year,
            "month_name_es": month_name,
        })

    return records


# ---------------------------------------------------------------------------
# Follow intermediate document pages to discover nested PDFs
# ---------------------------------------------------------------------------

def scrape_document_page(url: str, section: str,
                         cycle_start: Optional[int],
                         cycle_end: Optional[int]) -> list[dict]:
    """
    Fetch a gob.mx document-collection page and return one record per PDF link
    found on it.
    """
    soup = _soup(url)
    if soup is None:
        return []

    results: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = _normalise_url(a["href"])
        if not href.lower().endswith(".pdf"):
            continue
        if href in seen:
            continue
        seen.add(href)

        text = " ".join(a.get_text().split()).strip()
        month_num, year, month_name = _parse_month_year(text, href)
        cs, ce = _parse_cycle(text, href)

        results.append({
            "section": section,
            "link_text": text,
            "url": href,
            "is_pdf": True,
            "cycle_start": cs or cycle_start,
            "cycle_end": ce or cycle_end,
            "month_number": month_num,
            "year": year,
            "month_name_es": month_name,
            "parent_page": url,
        })

    return results


# ---------------------------------------------------------------------------
# Master index builder
# ---------------------------------------------------------------------------

_SKIP_HREFS = {
    "https://www.gob.mx/cms/uploads/attachment/file/592730/"
    "Secci_n_ver_m_s_pol_tica_comercial.pdf",
}

_SKIP_LINK_TEXT_LOWER = {
    "sección principal política comercial.",
    "seccion principal politica comercial.",
}

_SKIP_SECTIONS_LOWER = {
    "publicaciones recientes", "unknown", "enlaces",
    "¿qué es gob.mx?", "contacto", "síguenos en",
    "sección principal política comercial.",
}

QUICK_KEYWORDS = [
    "balance nacional de azúcar y edulcorantes estimado",
    "balance nacional de azúcar",
    "exportaciones de azúcar",
    "histórico del precio",
    "estudios y análisis",
]


def build_index(quick: bool = False) -> pd.DataFrame:
    """
    1. Fetch the landing page and classify every link.
    2. For intermediate (non-PDF) links, follow them to collect nested PDFs.
    3. Return a DataFrame with one row per document.
    """
    print("Fetching landing page ...")
    soup = _soup(LANDING_URL)
    if soup is None:
        print("Could not fetch landing page.")
        return pd.DataFrame()

    landing_records = parse_landing_page(soup)
    print(f"  Found {len(landing_records)} links on landing page")

    index_rows: list[dict] = []
    seen_urls: set[str] = set()

    for rec in landing_records:
        if rec["url"] in _SKIP_HREFS:
            continue
        if rec["section"].lower().strip() in _SKIP_SECTIONS_LOWER:
            continue
        link_lower = rec["link_text"].lower().strip()
        if any(s in link_lower for s in ("principal pol", "principal politica")):
            continue
        if quick:
            sec_lower = rec["section"].lower()
            if not any(kw in sec_lower for kw in QUICK_KEYWORDS):
                continue

        if rec["is_pdf"]:
            if rec["url"] not in seen_urls:
                seen_urls.add(rec["url"])
                rec["parent_page"] = LANDING_URL
                index_rows.append(rec)
        else:
            url = rec["url"]
            if "gob.mx" not in url:
                continue
            if "dof.gob.mx" in url:
                rec["is_pdf"] = False
                rec["parent_page"] = LANDING_URL
                if url not in seen_urls:
                    seen_urls.add(url)
                    index_rows.append(rec)
                continue

            print(f"  Following >> {rec['section']}: {rec['link_text'][:60]}...")
            time.sleep(DELAY)
            nested = scrape_document_page(
                url, rec["section"], rec["cycle_start"], rec["cycle_end"]
            )
            if nested:
                for nr in nested:
                    if nr["url"] not in seen_urls:
                        seen_urls.add(nr["url"])
                        index_rows.append(nr)
                print(f"    {len(nested)} PDFs found")
            else:
                rec["parent_page"] = LANDING_URL
                if rec["url"] not in seen_urls:
                    seen_urls.add(rec["url"])
                    index_rows.append(rec)
                print(f"    No nested PDFs -- kept original link")

    if not index_rows:
        return pd.DataFrame()

    df = pd.DataFrame(index_rows)

    # ----- derive missing fields -----
    df["cycle_start"] = pd.to_numeric(df["cycle_start"], errors="coerce").astype("Int64")
    df["cycle_end"] = pd.to_numeric(df["cycle_end"], errors="coerce").astype("Int64")
    df["month_number"] = pd.to_numeric(df["month_number"], errors="coerce").astype("Int64")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["section_slug"] = (
        df["section"]
        .str.lower()
        .str.replace(r"[^a-záéíóúñü0-9]+", "_", regex=True)
        .str.strip("_")
    )

    df["document_date"] = pd.NaT
    mask = df["year"].notna() & df["month_number"].notna()
    df.loc[mask, "document_date"] = pd.to_datetime(
        df.loc[mask, "year"].astype(str) + "-"
        + df.loc[mask, "month_number"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )

    col_order = [
        "section", "section_slug", "link_text",
        "cycle_start", "cycle_end",
        "year", "month_number", "month_name_es", "document_date",
        "url", "is_pdf", "parent_page",
    ]
    for c in col_order:
        if c not in df.columns:
            df[c] = None
    df = df[col_order]

    df = df.sort_values(
        ["section", "cycle_start", "year", "month_number"],
        na_position="last",
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Download PDFs
# ---------------------------------------------------------------------------

def _safe_filename(row: pd.Series) -> str:
    slug = row["section_slug"][:40]
    cycle = (
        f"{int(row['cycle_start'])}-{int(row['cycle_end'])}"
        if pd.notna(row["cycle_start"]) else "nocycle"
    )
    date_part = ""
    if pd.notna(row["year"]) and pd.notna(row["month_number"]):
        date_part = f"_{int(row['year'])}_{int(row['month_number']):02d}"
    elif pd.notna(row["year"]):
        date_part = f"_{int(row['year'])}"

    text_slug = re.sub(r"[^a-z0-9]+", "_", row["link_text"][:50].lower()).strip("_")
    return f"{slug}__{cycle}{date_part}__{text_slug}.pdf"


def download_all(index_df: pd.DataFrame, out_dir: Path = PDF_DIR) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_rows = index_df[index_df["is_pdf"] == True]  # noqa: E712
    downloaded = 0
    for idx, row in pdf_rows.iterrows():
        fname = _safe_filename(row)
        path = out_dir / fname
        if path.exists():
            downloaded += 1
            continue
        r = _get(row["url"], timeout=60)
        if r is None:
            continue
        path.write_bytes(r.content)
        downloaded += 1
        print(f"  [{downloaded}/{len(pdf_rows)}] {fname}")
        time.sleep(DELAY)
    return downloaded


# ---------------------------------------------------------------------------
# Extract tabular data from PDFs
# ---------------------------------------------------------------------------

_BALANCE_KEYS = {
    "inventario inicial":          "inventario_inicial",
    "producción nacional":         "produccion_nacional",
    "produccion nacional":         "produccion_nacional",
    "importaciones":               "importaciones",
    "oferta total":                "oferta_total",
    "exportaciones":               "exportaciones",
    "ventas immex":                "ventas_immex",
    "ventas a immex":              "ventas_immex",
    "maquiladoras":                "ventas_immex",
    "consumo nacional aparente":   "consumo_nacional_aparente",
    "consumo interno":             "consumo_nacional_aparente",
    "demanda total":               "demanda_total",
    "inventario final":            "inventario_final",
    "inventario óptimo":           "inventario_optimo",
    "inventario optimo":           "inventario_optimo",
    "producción de fructosa":      "produccion_fructosa",
    "produccion de fructosa":      "produccion_fructosa",
    "importaciones de fructosa":   "importaciones_fructosa",
    "consumo de fructosa":         "consumo_fructosa",
}

_NUM_RE = re.compile(r"^-?[\d,]+\.?\d*$")


def _first_number(cells: list[str]) -> Optional[float]:
    for c in reversed(cells):
        s = (c or "").strip().replace(",", "").replace(" ", "")
        if _NUM_RE.match(s):
            try:
                return float(s)
            except ValueError:
                pass
    return None


def extract_table_from_pdf(pdf_path: Path) -> dict:
    try:
        import pdfplumber
    except ImportError:
        print("pdfplumber not installed -- run:  pip install pdfplumber")
        return {}

    out: dict = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in (page.extract_tables() or []):
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 2:
                        continue
                    label = (row[0] or "").strip().lower()
                    for keyword, col_name in _BALANCE_KEYS.items():
                        if keyword in label:
                            val = _first_number(row[1:])
                            if val is not None and col_name not in out:
                                out[col_name] = val
                            break
    return out


def extract_all(index_df: pd.DataFrame, pdf_dir: Path = PDF_DIR) -> pd.DataFrame:
    pdf_rows = index_df[index_df["is_pdf"] == True].copy()  # noqa: E712
    rows: list[dict] = []

    for idx, row in pdf_rows.iterrows():
        fname = _safe_filename(row)
        path = pdf_dir / fname
        if not path.exists():
            continue
        if path.stat().st_size < 500:
            continue

        data = extract_table_from_pdf(path)
        if not data:
            continue

        rec = {
            "section": row["section"],
            "section_slug": row["section_slug"],
            "cycle_start": row["cycle_start"],
            "cycle_end": row["cycle_end"],
            "year": row["year"],
            "month_number": row["month_number"],
            "month_name_es": row["month_name_es"],
            "document_date": row["document_date"],
            "link_text": row["link_text"],
            "pdf_file": fname,
        }
        rec.update(data)
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["section", "cycle_start", "year", "month_number"],
        na_position="last",
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _truncate_sheet_name(name: str, max_len: int = 31) -> str:
    """Excel sheet names cannot exceed 31 characters."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "~"


def export_to_excel(index_df: pd.DataFrame, path: str = INDEX_XLSX):
    """
    Write an Excel workbook with:
      - 'All Documents' sheet (the full index)
      - One sheet per section (filtered view)
    Column widths are auto-adjusted for readability.
    """
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        index_df.to_excel(writer, sheet_name="All Documents", index=False)

        seen_names: set[str] = set()
        for section in sorted(index_df["section"].unique()):
            slug = re.sub(r"[^A-Za-z0-9 ]+", "", section).strip()[:28]
            sheet_name = _truncate_sheet_name(slug)
            if sheet_name in seen_names:
                sheet_name = _truncate_sheet_name(slug[:25] + " 2")
            seen_names.add(sheet_name)

            section_df = index_df[index_df["section"] == section].reset_index(drop=True)
            section_df.to_excel(writer, sheet_name=sheet_name, index=False)

        for ws in writer.book.worksheets:
            for col_idx, col_cells in enumerate(ws.columns, 1):
                max_width = max(
                    len(str(cell.value or "")) for cell in col_cells
                )
                ws.column_dimensions[get_column_letter(col_idx)].width = min(
                    max(max_width + 2, 12), 60
                )

    print(f"  Excel saved -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("CONADESUCA Politica Comercial -- Full Document Scraper")
    print("=" * 64)
    print(f"Source: {LANDING_URL}\n")

    quick = "--quick" in sys.argv or "quick" in sys.argv
    do_download = "--download" in sys.argv or "download" in sys.argv
    do_extract = "--extract" in sys.argv or "extract" in sys.argv

    if quick:
        print("Mode: QUICK (subset of sections)\n")
    else:
        print("Mode: FULL (all sections)\n")

    # 1. Build index
    t0 = time.time()
    index_df = build_index(quick=quick)
    elapsed = time.time() - t0

    if index_df.empty:
        print("No documents found.")
        return

    index_df.to_csv(INDEX_CSV, index=False)
    export_to_excel(index_df, INDEX_XLSX)

    n_pdf = index_df["is_pdf"].sum()
    n_sections = index_df["section"].nunique()
    print(f"\n{'-' * 48}")
    print(f"Index: {len(index_df)} documents across {n_sections} sections")
    print(f"  PDFs: {n_pdf}  |  Non-PDF links: {len(index_df) - n_pdf}")
    print(f"  CSV  -> {INDEX_CSV}")
    print(f"  Excel -> {INDEX_XLSX}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\nSection breakdown:")
    print(index_df.groupby("section").size().to_string())

    # 2. Download
    if do_download:
        print(f"\n{'-' * 48}")
        print(f"Downloading PDFs to {PDF_DIR}/ ...")
        n = download_all(index_df, PDF_DIR)
        print(f"  Downloaded/cached: {n} files")

    # 3. Extract
    if do_extract:
        print(f"\n{'-' * 48}")
        if not PDF_DIR.exists():
            print("No PDFs on disk -- run with 'download' first.")
            return
        print("Extracting table data from PDFs ...")
        ext_df = extract_all(index_df, PDF_DIR)
        if ext_df.empty:
            print("  No tables extracted (check PDFs or install pdfplumber).")
        else:
            ext_df.to_csv(EXTRACTED_CSV, index=False)
            print(f"  Extracted {len(ext_df)} rows -> {EXTRACTED_CSV}")
            print(f"\n  Columns: {list(ext_df.columns)}")
            print(f"\n  Head:\n{ext_df.head(10).to_string()}")


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_browser()
