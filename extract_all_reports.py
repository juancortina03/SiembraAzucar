"""
CONADESUCA - Download & Extract All Reports to Excel
=====================================================
Reads politica_comercial_index.csv, downloads every PDF, extracts tabular
data with per-section logic, and writes one Excel file per report category
with ML-ready columns.

Output Excel files (in excel_reports/):
  01_balance_nacional_azucar.xlsx        - monthly sugar supply/demand
  02_balance_nacional_edulcorantes.xlsx   - monthly sweetener supply/demand
  03_balance_estimado.xlsx               - annual estimated balance
  04_reporte_mensual_mercado.xlsx        - monthly market summary
  05_reporte_semanal_precios.xlsx        - weekly price snapshots
  06_balances_mundiales.xlsx             - world sugar balance
  07_exportaciones_ciclo.xlsx            - per-cycle export summary
  08_flujos_immex.xlsx                   - IMMEX sugar flows
  09_historico_precio_referencia.xlsx    - historic reference prices

Usage:
  python extract_all_reports.py              # download + extract all
  python extract_all_reports.py quick        # only Balance Azucar + Edulcorantes
  python extract_all_reports.py skip-download  # extract from already-downloaded PDFs
"""

import io
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import pdfplumber

# ---------------------------------------------------------------------------
INDEX_CSV = "politica_comercial_index.csv"
PDF_DIR = Path("politica_comercial_pdfs")
EXCEL_DIR = Path("excel_reports")
DELAY = 0.3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(s: str) -> Optional[float]:
    """Parse a number string like '1,234,567' or '-78%' -> float or None."""
    if not s:
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    s = s.rstrip("%")
    if s in ("-", "", "n.d.", "n.d", "N.D.", "N/A", "n/a"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_filename(row: pd.Series) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", row["section_slug"][:35]).strip("_")
    cs = int(row["cycle_start"]) if pd.notna(row["cycle_start"]) else 0
    ce = int(row["cycle_end"]) if pd.notna(row["cycle_end"]) else 0
    y = int(row["year"]) if pd.notna(row["year"]) else 0
    m = int(row["month_number"]) if pd.notna(row["month_number"]) else 0
    text_slug = re.sub(r"[^a-z0-9]+", "_", row["link_text"][:40].lower()).strip("_")
    return f"{slug}__{cs}-{ce}_{y}_{m:02d}__{text_slug}.pdf"


def _download_section(section_df: pd.DataFrame, pdf_dir: Path) -> int:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdfs = section_df[section_df["is_pdf"] == True]  # noqa: E712
    count = 0
    for _, row in pdfs.iterrows():
        fname = _safe_filename(row)
        path = pdf_dir / fname
        if path.exists() and path.stat().st_size > 500:
            count += 1
            continue
        try:
            r = requests.get(row["url"], headers=HEADERS, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
            count += 1
        except Exception:
            pass
        time.sleep(DELAY)
    return count


def _open_pdf(row: pd.Series, pdf_dir: Path):
    fname = _safe_filename(row)
    path = pdf_dir / fname
    if not path.exists() or path.stat().st_size < 500:
        return None
    try:
        return pdfplumber.open(path)
    except Exception:
        return None


def _write_excel(df: pd.DataFrame, path: Path):
    """Write a DataFrame to Excel with:
      - 'Data' sheet: all extracted rows and columns (raw)
      - 'ML Ready' sheet: only columns present in >= 50% of rows, numeric
        columns forward-filled where appropriate, sorted chronologically.
    """
    from openpyxl.utils import get_column_letter

    # ML Ready: drop columns that are mostly null (< 50% fill rate)
    threshold = len(df) * 0.5
    ml = df.dropna(axis=1, thresh=max(threshold, 1)).copy()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        ml.to_excel(writer, sheet_name="ML Ready", index=False)
        for ws in writer.book.worksheets:
            for ci, col_cells in enumerate(ws.columns, 1):
                widths = [len(str(c.value or "")) for c in col_cells]
                ws.column_dimensions[get_column_letter(ci)].width = min(max(max(widths) + 2, 10), 55)

    n_dropped_cols = len(df.columns) - len(ml.columns)
    msg = f" (ML Ready: {len(ml)} rows x {len(ml.columns)} cols, dropped {n_dropped_cols} sparse cols)" if n_dropped_cols else ""
    print(f"    -> {path.name}: {len(df)} rows x {len(df.columns)} cols{msg}")


# ---------------------------------------------------------------------------
# SECTION EXTRACTORS
# ---------------------------------------------------------------------------

# Row label -> column name mapping for Balance Nacional de Azucar
_BAL_AZ_LABELS = [
    ("oferta total", "oferta_total"),
    ("inventario inicial", "inventario_inicial"),
    ("producción", "produccion"),
    ("produccion", "produccion"),
    ("importaciones", "importaciones"),
    ("demanda total", "demanda_total"),
    ("exportaciones totales", "exportaciones_totales"),
    ("exportaciones", "exportaciones"),
    ("ventas a immex", "ventas_immex"),
    ("ventas immex", "ventas_immex"),
    ("consumo nacional aparente", "consumo_nacional_aparente"),
    ("consumo nacional", "consumo_nacional_aparente"),
    ("consumo dom", "consumo_nacional_aparente"),
    ("inventario final", "inventario_final"),
    ("inventario óptimo", "inventario_optimo"),
    ("inventario optimo", "inventario_optimo"),
]


def _match_label(text: str, labels: list) -> Optional[str]:
    text = text.lower().strip()
    for keyword, col_name in labels:
        if keyword in text:
            return col_name
    return None


_BAL_AZ_TEXT_RE = re.compile(
    r'^(?P<label>[A-ZÁÉÍÓÚÑa-záéíóúñü][\wÁÉÍÓÚÑáéíóúñü /\-()]*?)\s+'
    r'(?P<nums>[\d,.\-]+(?:\s+[\d,.\-]+)*)'
    r'(?:\s+[\-\d]+\.?\d*%|\s+---|\s+ND|\s+-o-)?'
    r'\s*$'
)


def _detect_old_format(text: str) -> bool:
    """2008-2013 PDFs have Total/Estándar/Refinada columns instead of
    current-cycle/prev-cycle comparison."""
    tl = text.lower()
    return ("est\xe1ndar" in tl or "estandar" in tl) and "refinada" in tl


def _parse_balance_page_text(text: str, labels: list) -> dict:
    """Parse a page of Balance text into column values.

    Handles two distinct formats:
    - Old (2008-2013): "Label Total Estándar Refinada"
      -> col = Total value only (no prev_cycle available)
    - New (2014-2026): "Label CurrentCycle PrevCycle %Change"
      -> col = current, col_prev_cycle = previous
    """
    is_old = _detect_old_format(text)
    out: dict = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _BAL_AZ_TEXT_RE.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        col = _match_label(label, labels)
        if col is None:
            continue
        nums_raw = m.group("nums").split()
        vals = [_num(n) for n in nums_raw]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if col not in out:
            out[col] = vals[0]
        if not is_old:
            prev_col = col + "_prev_cycle"
            if prev_col not in out and len(vals) >= 2:
                out[prev_col] = vals[1]
    return out


def extract_balance_azucar(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract monthly + accumulated balance from Balance Nacional de Azucar PDFs.

    Uses text-based parsing so it works across all PDF format eras (2008-2026).
    pdfplumber table detection is unreliable for these PDFs: some eras return
    zero tables, others put labels outside table cells.
    """
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for pi, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if len(page_text.strip()) < 50:
                    continue
                is_acum = "acumulado" in page_text.lower()

                rec = {
                    "cycle_start": idx_row.get("cycle_start"),
                    "cycle_end": idx_row.get("cycle_end"),
                    "year": idx_row.get("year"),
                    "month_number": idx_row.get("month_number"),
                    "month_name_es": idx_row.get("month_name_es"),
                    "type": "acumulado" if is_acum else "mensual",
                }

                parsed = _parse_balance_page_text(page_text, _BAL_AZ_LABELS)
                rec.update(parsed)

                if len(rec) > 6:
                    rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start", "year", "month_number", "type"]).reset_index(drop=True)
    return df


_BAL_EDUL_LABELS = [
    ("oferta total", "oferta_total"),
    ("inventario inicial", "inventario_inicial"),
    ("producción", "produccion"),
    ("produccion", "produccion"),
    ("importaciones", "importaciones"),
    ("demanda total", "demanda_total"),
    ("exportaciones", "exportaciones"),
    ("ventas", "ventas_immex"),
    ("consumo nacional", "consumo_nacional_aparente"),
    ("consumo interno", "consumo_nacional_aparente"),
    ("consumo dom", "consumo_nacional_aparente"),
    ("inventario final", "inventario_final"),
]


def _parse_edulcorantes_page_text(text: str, labels: list) -> dict:
    """Parse edulcorantes page text into {col_total, col_azucar, col_fructosa}.

    Each data line has 3 numbers (Total, Azucar, Fructosa) followed by an
    optional percentage change column. Works across all format eras.
    """
    out: dict = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _BAL_AZ_TEXT_RE.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        col = _match_label(label, labels)
        if col is None:
            continue
        nums_raw = m.group("nums").split()
        vals = [_num(n) for n in nums_raw]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        suffixes = ["_total", "_azucar", "_fructosa"]
        for i, suffix in enumerate(suffixes):
            key = col + suffix
            if key not in out and i < len(vals):
                out[key] = vals[i]
    return out


def extract_balance_edulcorantes(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract monthly + accumulated balance from Balance Nacional de Edulcorantes PDFs.

    Uses text-based parsing with 3-column output (total/azucar/fructosa) to
    work across all PDF format eras (2008-2026).
    """
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for pi, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if len(page_text.strip()) < 50:
                    continue
                is_acum = "acumulado" in page_text.lower()

                rec = {
                    "cycle_start": idx_row.get("cycle_start"),
                    "cycle_end": idx_row.get("cycle_end"),
                    "year": idx_row.get("year"),
                    "month_number": idx_row.get("month_number"),
                    "month_name_es": idx_row.get("month_name_es"),
                    "type": "acumulado" if is_acum else "mensual",
                }

                parsed = _parse_edulcorantes_page_text(page_text, _BAL_EDUL_LABELS)
                rec.update(parsed)

                if len(rec) > 6:
                    rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start", "year", "month_number", "type"]).reset_index(drop=True)
    return df


def extract_balance_estimado(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract annual estimated balance (Azucar + Edulcorantes) PDFs."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    est_labels = [
        ("oferta total", "oferta_total"),
        ("inventario inicial", "inventario_inicial"),
        ("saldo de exportaci", "saldo_exportacion"),
        ("inventario inicial disponible", "inventario_inicial_disponible"),
        ("producción", "produccion"),
        ("produccion", "produccion"),
        ("importaciones", "importaciones"),
        ("demanda total", "demanda_total"),
        ("exportaciones", "exportaciones"),
        ("consumo nacional", "consumo_nacional_aparente"),
        ("consumo interno", "consumo_nacional_aparente"),
        ("inventario final", "inventario_final"),
        ("inventario óptimo", "inventario_optimo"),
        ("inventario optimo", "inventario_optimo"),
    ]

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for pi, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                page_text = (page.extract_text() or "").lower()
                is_edulc = "edulcorantes" in page_text
                table_type = "edulcorantes" if is_edulc else "azucar"

                for table in tables:
                    if not table or len(table) < 3:
                        continue

                    rec = {
                        "cycle_start": idx_row.get("cycle_start"),
                        "cycle_end": idx_row.get("cycle_end"),
                        "table_type": table_type,
                    }

                    for trow in table:
                        if not trow or not trow[0]:
                            continue
                        label = (trow[0] or "").strip()
                        col = _match_label(label, est_labels)
                        if col is None:
                            continue

                        if table_type == "edulcorantes" and len(trow) >= 4:
                            for suffix, ci in [("_total", 1), ("_azucar", 2), ("_fructosa", 3)]:
                                v = _num(trow[ci]) if ci < len(trow) else None
                                if v is not None:
                                    rec[col + suffix] = v
                        else:
                            v = _num(trow[1]) if len(trow) > 1 else None
                            if v is not None and col not in rec:
                                rec[col] = v

                    if len(rec) > 3:
                        rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start", "table_type"]).reset_index(drop=True)
    return df


def extract_reporte_mensual(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract monthly market report summary table."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for page in pdf.pages[:2]:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 5:
                        continue

                    # Header detection: find row with column names
                    header = None
                    data_start = 0
                    for ri, trow in enumerate(table[:4]):
                        joined = " ".join(str(c or "") for c in trow).lower()
                        if "concepto" in joined or "(a)" in joined:
                            header = trow
                            data_start = ri + 1
                            break

                    if header is None:
                        continue

                    col_names = [str(c or "").strip().replace("\n", " ")[:50] for c in header]

                    for trow in table[data_start:]:
                        if not trow or not trow[0]:
                            continue
                        label = str(trow[0] or "").strip()
                        if not label or len(label) < 3:
                            continue

                        rec = {
                            "cycle_start": idx_row.get("cycle_start"),
                            "cycle_end": idx_row.get("cycle_end"),
                            "year": idx_row.get("year"),
                            "month_number": idx_row.get("month_number"),
                            "month_name_es": idx_row.get("month_name_es"),
                            "concepto": label.replace("\n", " "),
                        }
                        for ci in range(1, len(trow)):
                            if ci < len(col_names):
                                cname = re.sub(r"[^a-z0-9]+", "_", col_names[ci].lower()).strip("_") or f"col_{ci}"
                            else:
                                cname = f"col_{ci}"
                            rec[cname] = _num(trow[ci])

                        if any(v is not None for k, v in rec.items() if k not in ("concepto", "cycle_start", "cycle_end", "year", "month_number", "month_name_es")):
                            rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start", "year", "month_number"]).reset_index(drop=True)
    return df


def extract_reporte_semanal(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract weekly price report - first page table with daily prices."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            page = pdf.pages[0]
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 4:
                    continue

                header = None
                data_start = 0
                for ri, trow in enumerate(table[:4]):
                    joined = " ".join(str(c or "") for c in trow).lower()
                    if "concepto" in joined:
                        header = trow
                        data_start = ri + 1
                        break

                if header is None:
                    continue

                col_names = [str(c or "").strip().replace("\n", " ")[:30] for c in header]

                for trow in table[data_start:]:
                    if not trow or not trow[0]:
                        continue
                    label = str(trow[0] or "").strip().replace("\n", " ")
                    if not label or len(label) < 3:
                        continue

                    rec = {
                        "cycle_start": idx_row.get("cycle_start"),
                        "cycle_end": idx_row.get("cycle_end"),
                        "year": idx_row.get("year"),
                        "month_number": idx_row.get("month_number"),
                        "concepto": label,
                    }

                    for ci in range(1, min(len(trow), len(col_names))):
                        cname = col_names[ci]
                        if not cname:
                            continue
                        cname_clean = re.sub(r"[^a-z0-9/]+", "_", cname.lower()).strip("_") or f"col_{ci}"
                        rec[cname_clean] = _num(trow[ci])

                    if any(v is not None for k, v in rec.items() if k not in ("concepto", "cycle_start", "cycle_end", "year", "month_number")):
                        rows.append(rec)
                break
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start", "year", "month_number"]).reset_index(drop=True)
    return df


def extract_exportaciones(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract export summary tables from per-cycle export PDFs."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for page in pdf.pages[:6]:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 3:
                        continue

                    header = table[0]
                    col_names = [str(c or "").strip().replace("\n", " ")[:40] for c in header]
                    joined_h = " ".join(col_names).lower()

                    if not ("concepto" in joined_h or "exportaciones" in joined_h or "tonelada" in joined_h):
                        continue

                    for trow in table[1:]:
                        if not trow or not trow[0]:
                            continue
                        label = str(trow[0] or "").strip().replace("\n", " ")
                        if not label or "tonelada" in label.lower():
                            continue
                        rec = {
                            "cycle_start": idx_row.get("cycle_start"),
                            "cycle_end": idx_row.get("cycle_end"),
                            "concepto": label,
                        }
                        for ci in range(1, len(trow)):
                            cname = col_names[ci] if ci < len(col_names) else f"col_{ci}"
                            cname = re.sub(r"[^a-z0-9]+", "_", cname.lower()).strip("_") or f"col_{ci}"
                            rec[cname] = _num(trow[ci])
                        if any(v is not None for k, v in rec.items() if k not in ("concepto", "cycle_start", "cycle_end")):
                            rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start"]).reset_index(drop=True)
    return df


def extract_immex(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract IMMEX sugar flow tables."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for page in pdf.pages[:1]:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 3:
                        continue

                    for trow in table[2:]:
                        if not trow or not trow[0]:
                            continue
                        label = str(trow[0] or "").strip().replace("\n", " ")
                        if not label:
                            continue
                        rec = {
                            "cycle_start": idx_row.get("cycle_start"),
                            "cycle_end": idx_row.get("cycle_end"),
                            "concepto": label,
                        }
                        for ci, suffix in [(1, "total"), (2, "estandar"), (3, "refinada")]:
                            if ci < len(trow):
                                rec[suffix] = _num(trow[ci])
                        if any(v is not None for k, v in rec.items() if k not in ("concepto", "cycle_start", "cycle_end")):
                            rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start"]).reset_index(drop=True)
    return df


def extract_historico(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract the historic reference price / indicators table."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 4:
                        continue
                    # This table has zafras as columns
                    cycle_headers = []
                    for ri, trow in enumerate(table[:3]):
                        for c in trow[2:]:
                            s = str(c or "").strip()
                            m = re.search(r"(\d{4})/(\d{4})", s)
                            if m:
                                cycle_headers.append(s)
                    if not cycle_headers:
                        continue

                    # Find the row with cycle years
                    year_row_idx = None
                    for ri, trow in enumerate(table[:3]):
                        count = sum(1 for c in trow if re.search(r"\d{4}/\d{4}", str(c or "")))
                        if count >= 3:
                            year_row_idx = ri
                            break
                    if year_row_idx is None:
                        continue

                    cycles = []
                    for c in table[year_row_idx]:
                        s = str(c or "").strip()
                        m = re.search(r"(\d{4})/(\d{4})", s)
                        if m:
                            cycles.append(s)
                        else:
                            cycles.append(None)

                    for trow in table[year_row_idx + 1:]:
                        if not trow or not trow[0]:
                            continue
                        variable = str(trow[0] or "").strip().replace("\n", " ")
                        unit = str(trow[1] or "").strip().replace("\n", " ") if len(trow) > 1 else ""
                        for ci in range(2, len(trow)):
                            if ci >= len(cycles) or cycles[ci] is None:
                                continue
                            v = _num(trow[ci])
                            if v is None:
                                continue
                            m = re.search(r"(\d{4})/(\d{4})", cycles[ci])
                            rows.append({
                                "cycle": cycles[ci],
                                "cycle_start": int(m.group(1)),
                                "cycle_end": int(m.group(2)),
                                "variable": variable,
                                "unit": unit,
                                "value": v,
                            })
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Pivot: one row per cycle, one column per variable
    try:
        pivot = df.pivot_table(index=["cycle", "cycle_start", "cycle_end"],
                               columns="variable", values="value", aggfunc="first")
        pivot = pivot.reset_index()
        pivot.columns.name = None
        pivot = pivot.sort_values("cycle_start").reset_index(drop=True)
        return pivot
    except Exception:
        return df.sort_values(["cycle_start"]).reset_index(drop=True)


def extract_balances_mundiales(section_df: pd.DataFrame, pdf_dir: Path) -> pd.DataFrame:
    """Extract world balance tables."""
    rows = []
    pdfs = section_df[section_df["is_pdf"] == True].copy()  # noqa: E712

    for _, idx_row in pdfs.iterrows():
        pdf = _open_pdf(idx_row, pdf_dir)
        if pdf is None:
            continue
        try:
            for page in pdf.pages[:3]:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 3:
                        continue

                    header = table[0]
                    col_names = [str(c or "").strip().replace("\n", " ")[:40] for c in header]

                    for trow in table[1:]:
                        if not trow or not trow[0]:
                            continue
                        label = str(trow[0] or "").strip().replace("\n", " ")
                        if not label or len(label) < 2:
                            continue
                        rec = {
                            "cycle_start": idx_row.get("cycle_start"),
                            "cycle_end": idx_row.get("cycle_end"),
                            "concepto": label,
                            "link_text": idx_row.get("link_text", ""),
                        }
                        for ci in range(1, len(trow)):
                            cname = col_names[ci] if ci < len(col_names) else f"col_{ci}"
                            cname = re.sub(r"[^a-z0-9]+", "_", cname.lower()).strip("_") or f"col_{ci}"
                            rec[cname] = _num(trow[ci])
                        if any(v is not None for k, v in rec.items() if k not in ("concepto", "cycle_start", "cycle_end", "link_text")):
                            rows.append(rec)
        finally:
            pdf.close()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["cycle_start"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Section dispatch
# ---------------------------------------------------------------------------

SECTION_CONFIG = [
    {
        "section_pattern": "BALANCE NACIONAL DE AZÚCAR",
        "exclude_pattern": "EDULCORANTES|ESTIMADO",
        "extractor": extract_balance_azucar,
        "excel_name": "01_balance_nacional_azucar.xlsx",
        "label": "Balance Nacional de Azucar",
    },
    {
        "section_pattern": "BALANCE NACIONAL DE EDULCORANTES",
        "exclude_pattern": None,
        "extractor": extract_balance_edulcorantes,
        "excel_name": "02_balance_nacional_edulcorantes.xlsx",
        "label": "Balance Nacional de Edulcorantes",
    },
    {
        "section_pattern": "BALANCE NACIONAL DE AZÚCAR Y EDULCORANTES ESTIMADO",
        "exclude_pattern": None,
        "extractor": extract_balance_estimado,
        "excel_name": "03_balance_estimado.xlsx",
        "label": "Balance Estimado (Azucar + Edulcorantes)",
    },
    {
        "section_pattern": "REPORTE MENSUAL DEL MERCADO",
        "exclude_pattern": None,
        "extractor": extract_reporte_mensual,
        "excel_name": "04_reporte_mensual_mercado.xlsx",
        "label": "Reporte Mensual del Mercado",
    },
    {
        "section_pattern": "REPORTE SEMANAL DE PRECIOS",
        "exclude_pattern": None,
        "extractor": extract_reporte_semanal,
        "excel_name": "05_reporte_semanal_precios.xlsx",
        "label": "Reporte Semanal de Precios",
    },
    {
        "section_pattern": "BALANCES MUNDIALES",
        "exclude_pattern": None,
        "extractor": extract_balances_mundiales,
        "excel_name": "06_balances_mundiales.xlsx",
        "label": "Balances Mundiales",
    },
    {
        "section_pattern": "EXPORTACIONES DE AZÚCAR POR CICLO",
        "exclude_pattern": None,
        "extractor": extract_exportaciones,
        "excel_name": "07_exportaciones_ciclo.xlsx",
        "label": "Exportaciones de Azucar por Ciclo",
    },
    {
        "section_pattern": "FLUJOS DE AZÚCAR DE LAS IMMEX",
        "exclude_pattern": None,
        "extractor": extract_immex,
        "excel_name": "08_flujos_immex.xlsx",
        "label": "Flujos de Azucar IMMEX",
    },
    {
        "section_pattern": "HISTÓRICO DEL PRECIO DE REFERENCIA",
        "exclude_pattern": None,
        "extractor": extract_historico,
        "excel_name": "09_historico_precio_referencia.xlsx",
        "label": "Historico Precio de Referencia",
    },
]

QUICK_LABELS = {
    "Balance Nacional de Azucar",
    "Balance Nacional de Edulcorantes",
    "Balance Estimado (Azucar + Edulcorantes)",
    "Historico Precio de Referencia",
}


def _match_section(section_name: str, pattern: str, exclude: Optional[str]) -> bool:
    if not re.search(pattern, section_name, re.I):
        return False
    if exclude and re.search(exclude, section_name, re.I):
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("CONADESUCA - Download & Extract All Reports to Excel")
    print("=" * 64)

    quick = "--quick" in sys.argv or "quick" in sys.argv
    skip_dl = "--skip-download" in sys.argv or "skip-download" in sys.argv

    if not Path(INDEX_CSV).exists():
        print(f"Index file not found: {INDEX_CSV}")
        print("Run conadesuca_politica_comercial_scraper.py first.")
        return

    index_df = pd.read_csv(INDEX_CSV)
    print(f"Loaded index: {len(index_df)} documents\n")

    EXCEL_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    configs = SECTION_CONFIG
    if quick:
        configs = [c for c in SECTION_CONFIG if c["label"] in QUICK_LABELS]
        print(f"QUICK mode: processing {len(configs)} sections\n")

    for cfg in configs:
        pattern = cfg["section_pattern"]
        exclude = cfg.get("exclude_pattern")
        mask = index_df["section"].apply(lambda s: _match_section(s, pattern, exclude))
        section_df = index_df[mask].copy()

        if section_df.empty:
            print(f"[SKIP] {cfg['label']}: no matching rows in index")
            continue

        n_pdf = (section_df["is_pdf"] == True).sum()  # noqa: E712
        print(f"[{cfg['label']}] {len(section_df)} rows, {n_pdf} PDFs")

        # Download
        if not skip_dl:
            print(f"  Downloading PDFs...")
            dl = _download_section(section_df, PDF_DIR)
            print(f"  {dl}/{n_pdf} PDFs on disk")

        # Extract
        print(f"  Extracting tables...")
        try:
            result_df = cfg["extractor"](section_df, PDF_DIR)
        except Exception as e:
            print(f"  ERROR during extraction: {e}")
            result_df = pd.DataFrame()

        if result_df.empty:
            print(f"  No data extracted")
            continue

        # Write Excel
        out_path = EXCEL_DIR / cfg["excel_name"]
        _write_excel(result_df, out_path)
        print()

    print(f"\nDone. Excel files in: {EXCEL_DIR}/")


if __name__ == "__main__":
    main()
