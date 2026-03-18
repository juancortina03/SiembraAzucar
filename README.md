# SNIIM Sugar Price Scraper (Mexico)

Scrapes **daily national average sugar prices** from the Mexican SNIIM (Sistema Nacional de Información e Integración de Mercados) for use in price prediction.

## Data sources

- **Azúcar Estándar**: [prod=156](https://www.economia-sniim.gob.mx/AzucarMesPorDia.asp?Cons=D&prod=156&dqMesMes=2&dqAnioMes=2026&Formato=Nor&submit=Ver+Resultados)
- **Azúcar Refinada**: [prod=155](https://www.economia-sniim.gob.mx/AzucarMesPorDia.asp?Cons=D&prod=155&dqMesMes=2&dqAnioMes=2026&Formato=Nor&submit=Ver+Resultados)

Prices are **precio frecuente por bulto de 50 kg** in centrales de abasto (national daily average).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Full scrape (2000–current):**  
Takes several minutes (~0.5 s per request).

```bash
python sniim_sugar_scraper.py
```

**Quick test (e.g. 2024–2025 only):**

```bash
python sniim_sugar_scraper.py quick
```

Output:

- Prints DataFrame info, head, tail, and summary by `product_type`.
- Saves **`sniim_sugar_prices.csv`** and **`sniim_sugar_prices.xlsx`** (downloadable Excel) with columns: `date`, `price`, `product_type`, `year`, `month`.

## DataFrame columns

| Column        | Description                          |
|---------------|--------------------------------------|
| `date`        | Trading date                         |
| `price`       | National average price (MXN per 50 kg bulto) |
| `product_type`| `estandar` or `refinada`             |
| `year`, `month` | For convenience                    |

You can pivot or filter by `product_type` for separate Estándar/Refinada series when building a prediction model.

---

## CONADESUCA Balance Nacional de Azúcar Scraper

Scrapes **monthly sugar balance index** (and optionally balance table data) from [CONADESUCA](https://www.gob.mx/conadesuca/documentos/politica-comercial-balance-nacional-de-azucar-2024-2025?state=published) for multiple crop cycles. Data is logged **by month** and intended for **machine learning** (e.g. predicting price together with SNIIM series).

### What it does

- **Index scrape**: Discovers cycles from **2015-2016 through 2024-2025** and collects one row per month with `cycle_start`, `cycle_end`, `year`, `month_number`, `month_name_es`, `pdf_url`. Supports alternate URL paths (e.g. 2017/18 uses `/es/documentos/...`, 2015/16 and 2016/17 use `balances-azucareros-zafra`).
- **Optional**: Download PDFs (`--download`) and/or extract balance tables from PDFs (`--extract`) into ML-ready columns: `inventario_inicial`, `produccion_nacional`, `importaciones`, `oferta_total`, `exportaciones`, `ventas_immex`, `consumo_nacional_aparente`, `demanda_total`, `inventario_final`, `inventario_optimo`, plus `month_start_date` for joining with SNIIM daily data.

### Usage

```bash
# Index only (all cycles, logs by month) → conadesuca_balance_index.csv
python conadesuca_balance_scraper.py

# Quick run (fewer cycles)
python conadesuca_balance_scraper.py quick

# Also download PDFs to conadesuca_balance_pdfs/
python conadesuca_balance_scraper.py download

# Also extract balance data from PDFs → conadesuca_balance_by_month.csv
python conadesuca_balance_scraper.py extract
```

For extraction you need PDFs on disk (run with `download` first, or place PDFs in `conadesuca_balance_pdfs/`). Install pdfplumber: `pip install pdfplumber`.

### Output files

| File | Description |
|------|-------------|
| `conadesuca_balance_index.csv` / **`.xlsx`** | One row per month: cycle, year, month, pdf_url (for all years found). Excel is downloadable. |
| `conadesuca_balance_by_month.csv` / **`.xlsx`** | One row per month with numeric balance fields for ML (requires `--extract` and PDFs). Excel when extraction runs. |
| `conadesuca_balance_pdfs/` | Downloaded PDFs (optional, created with `--download`). |

Use `month_start_date` and `year`/`month_number` to join balance data with SNIIM prices when building price prediction models.
