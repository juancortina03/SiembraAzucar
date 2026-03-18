"""
Sugar Focars -- Agricultural Intelligence Dashboard (Dash)
==========================================================
Interactive Dash dashboard with:
  - Price history & KPIs
  - ML model comparison
  - Forecast with confidence bands
  - Scenario-based prediction (user inputs balance variables)
  - Feature analysis

Run:  py dashboard.py
"""

import base64
import io
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table, no_update, ctx
from sugar_price_model import EnsembleModel

# ---------------------------------------------------------------
# Basic Auth (password-protected access)
# ---------------------------------------------------------------
DASH_USER = os.environ.get("DASH_USER", "admin")
DASH_PASS = os.environ.get("DASH_PASS", "sugar2026")


class _ModelUnpickler(pickle.Unpickler):
    """Remap __main__.EnsembleModel to sugar_price_model.EnsembleModel
    (the model was pickled when sugar_price_model ran as __main__)."""
    def find_class(self, module, name):
        if name == "EnsembleModel":
            return EnsembleModel
        return super().find_class(module, name)

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
RESULTS_DIR = Path("model_results")
PRICES_CSV = "sniim_sugar_prices.csv"

PRODUCT_LABELS = {
    "estandar": "Azucar Estandar",
    "refinada": "Azucar Refinada",
}

COLOR_ACTUAL = "#2d6a4f"
COLOR_FORECAST = "#b8860b"

MODEL_COLORS = {
    "ridge_regression": "#40916c",
    "random_forest": "#b8860b",
    "gradient_boosting": "#a0522d",
}

AGRO_PALETTE = ["#2d6a4f", "#b8860b", "#a0522d", "#40916c", "#6b8f3c", "#d4a843"]

BALANCE_LABELS = {
    "produccion": "Produccion Nacional (ton/ano)",
    "importaciones": "Importaciones (ton/ano)",
    "exportaciones_totales": "Exportaciones Totales (ton/ano)",
    "consumo_nacional_aparente": "Consumo Nacional Aparente (ton/ano)",
    "inventario_inicial": "Inventario Inicial (ton)",
    "inventario_final": "Inventario Final (ton)",
    "oferta_total": "Oferta Total (ton/ano)",
    "demanda_total": "Demanda Total (ton/ano)",
}

LEGEND_FONT = dict(color="#000000", size=11)

AGRO_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f6f2",
    font=dict(family="Inter, sans-serif", color="#000000", size=12),
    title_font=dict(size=16, color="#1a3a2a"),
    margin=dict(t=60, b=40, l=50, r=20),
    xaxis_gridcolor="rgba(45, 106, 79, 0.12)",
    xaxis_zerolinecolor="rgba(45, 106, 79, 0.2)",
    xaxis_showgrid=True,
    xaxis_tickfont=dict(color="#000000", size=11),
    xaxis_title_font=dict(color="#000000", size=12),
    yaxis_gridcolor="rgba(45, 106, 79, 0.12)",
    yaxis_zerolinecolor="rgba(45, 106, 79, 0.2)",
    yaxis_showgrid=True,
    yaxis_tickfont=dict(color="#000000", size=11),
    yaxis_title_font=dict(color="#000000", size=12),
)

PAGE_OPTIONS = [
    ("overview", "Resumen Ejecutivo"),
    ("history", "Tendencias de Mercado"),
    ("ml", "Precision del Modelo"),
    ("forecast", "Proyeccion de Precios"),
    ("scenario", "Analisis de Escenarios"),
    ("features", "Factores Clave"),
    ("market", "Inteligencia de Mercado"),
]

# ---------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------
def load_raw_prices():
    df = pd.read_csv(PRICES_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_model_outputs(product: str):
    prefix = product
    data = {}
    for key, fname in [
        ("metrics", f"{prefix}_metrics.csv"),
        ("predictions", f"{prefix}_predictions.csv"),
        ("feature_importance", f"{prefix}_feature_importance.csv"),
        ("forecast", f"{prefix}_forecast.csv"),
        ("featured", f"{prefix}_featured.csv"),
        ("monthly_data", f"{prefix}_monthly_data.csv"),
        ("monthly_forecast", f"{prefix}_monthly_forecast.csv"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            parse = ["date"] if "date" in key or key in ("predictions", "forecast", "featured") else []
            try:
                data[key] = pd.read_csv(p, parse_dates=parse if parse else False)
            except Exception:
                data[key] = pd.read_csv(p)

    for key, fname in [
        ("summary", f"{prefix}_summary.json"),
        ("latest_balance", f"{prefix}_latest_balance.json"),
        ("feature_cols", f"models/{prefix}_feature_cols.json"),
    ]:
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f:
                data[key] = json.load(f)

    model_path = RESULTS_DIR / "models" / f"{prefix}_best_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            data["model"] = _ModelUnpickler(f).load()

    scaler_path = RESULTS_DIR / "models" / f"{prefix}_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            data["scaler"] = pickle.load(f)

    weights_path = RESULTS_DIR / "models" / f"{prefix}_feat_weights.pkl"
    if weights_path.exists():
        with open(weights_path, "rb") as f:
            data["feat_weights"] = pickle.load(f)

    balance_path = RESULTS_DIR / "balance_monthly.csv"
    if balance_path.exists():
        data["balance"] = pd.read_csv(balance_path)

    external_path = RESULTS_DIR / "external_monthly.csv"
    if external_path.exists():
        data["external"] = pd.read_csv(external_path)

    return data


def format_price(val):
    return f"${val:,.2f} MXN"

def format_tons(val):
    return f"{val:,.0f}"


# ---------------------------------------------------------------
# Logo
# ---------------------------------------------------------------
LOGO_PATH = Path("logo.png")

def get_logo_src():
    if LOGO_PATH.exists():
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    return None

LOGO_SRC = get_logo_src()


# ---------------------------------------------------------------
# Reusable chart helpers
# ---------------------------------------------------------------
def make_balance_chart(balance_df):
    fig = go.Figure()
    period = balance_df["year"].astype(str) + "-" + balance_df["month_number"].astype(str).str.zfill(2)
    for col, color, name in [
        ("produccion", "#2d6a4f", "Produccion"),
        ("importaciones", "#b8860b", "Importaciones"),
        ("exportaciones_totales", "#a0522d", "Exportaciones"),
        ("consumo_nacional_aparente", "#6b8f3c", "Consumo Nac. Aparente"),
    ]:
        if col in balance_df.columns:
            fig.add_trace(go.Scatter(
                x=period, y=balance_df[col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
    fig.update_layout(
        **AGRO_LAYOUT, height=400,
        title="Componentes del Balance Azucarero (Mensual)",
        xaxis_title="Mes", yaxis_title="Toneladas",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
    )
    return fig


def kpi_card(label, value, delta=None, delta_positive=None):
    children = [
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ]
    if delta is not None:
        cls = "kpi-delta"
        if delta_positive is True:
            cls += " positive"
        elif delta_positive is False:
            cls += " negative"
        children.append(html.Div(delta, className=cls))
    return html.Div(children, className="kpi-card")


# ---------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------
app = Dash(
    __name__,
    title="Sugar Focars | Inteligencia Agricola",
    suppress_callback_exceptions=True,
)
server = app.server  # Flask server exposed for gunicorn


# -- Basic Auth middleware --
@server.before_request
def _basic_auth():
    if os.environ.get("DASH_NOAUTH") == "1":
        return
    from flask import request, Response
    auth = request.authorization
    if not auth or auth.username != DASH_USER or auth.password != DASH_PASS:
        return Response(
            "Login required.", 401,
            {"WWW-Authenticate": 'Basic realm="Sugar Focars"'},
        )

# -- Header --
logo_el = html.Img(src=LOGO_SRC, style={"maxHeight": "38px"}) if LOGO_SRC else html.Span("", style={"fontSize": "1.5rem"})

header = html.Div([
    html.Div([
        html.Div([
            logo_el,
            html.Span("Inteligencia Agricola", className="brand"),
        ], className="logo-section"),
        dcc.Dropdown(
            id="product-select",
            options=[{"label": v, "value": k} for k, v in PRODUCT_LABELS.items()],
            value="estandar",
            clearable=False,
            style={"width": "220px", "color": "#1a3a2a"},
        ),
        html.Button(
            "\u2B07 Descargar Excel",
            id="btn-download-excel",
            className="download-btn",
        ),
    ], className="header-bar"),
    html.Div("Fuentes: SNIIM, CONADESUCA, FRED & Banxico  |  Modelos Predictivos de Precio", className="header-subtitle"),
])

# -- Navigation --
nav_bar = html.Div(id="nav-bar", className="nav-bar")

# -- Layout --
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Download(id="download-excel"),
    header,
    nav_bar,
    html.Div(id="page-content", className="main-content"),
    html.Div([
        html.P("SUGAR FOCARS", className="brand-name"),
        html.P("Inteligencia Agricola \u2022 Fuentes: SNIIM + CONADESUCA \u2022 Modelos Predictivos de Precio", className="brand-sub"),
    ], className="footer"),
])


# ---------------------------------------------------------------
# Navigation callback
# ---------------------------------------------------------------
@callback(
    Output("nav-bar", "children"),
    Input("url", "pathname"),
)
def render_nav(pathname):
    current = (pathname or "/overview").strip("/") or "overview"
    buttons = []
    for pid, label in PAGE_OPTIONS:
        cls = "nav-btn active" if pid == current else "nav-btn"
        buttons.append(dcc.Link(label, href=f"/{pid}", className=cls))
    return buttons


# ---------------------------------------------------------------
# Page router
# ---------------------------------------------------------------
@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("product-select", "value"),
)
def render_page(pathname, product):
    page = (pathname or "/overview").strip("/") or "overview"
    raw_df = load_raw_prices()
    model_data = load_model_outputs(product)
    product_df = raw_df[raw_df["product_type"] == product].copy()
    has_model = "summary" in model_data

    if page == "overview":
        return page_overview(product, product_df, model_data, has_model, raw_df)
    elif page == "history":
        return page_history(product, product_df, raw_df)
    elif page == "ml":
        return page_ml(product, product_df, model_data, has_model)
    elif page == "forecast":
        return page_forecast(product, product_df, model_data, has_model)
    elif page == "scenario":
        return page_scenario(product, product_df, model_data, has_model)
    elif page == "features":
        return page_features(product, product_df, model_data, has_model)
    elif page == "market":
        return page_market_intel(product, product_df, model_data, has_model)
    return page_overview(product, product_df, model_data, has_model, raw_df)


# ---------------------------------------------------------------
# PAGE: Overview & KPIs
# ---------------------------------------------------------------
def page_overview(product, product_df, model_data, has_model, raw_df):
    if product_df.empty:
        return html.Div("No price data available.", className="warning-box")

    latest = product_df.iloc[-1]
    prev = product_df.iloc[-2] if len(product_df) > 1 else latest
    price_change = latest["price"] - prev["price"]
    pct_change = (price_change / prev["price"]) * 100 if prev["price"] else 0.0

    month_ago = product_df[product_df["date"] <= (latest["date"] - pd.Timedelta(days=30))]
    year_ago = product_df[product_df["date"] <= (latest["date"] - pd.Timedelta(days=365))]
    month_change = latest["price"] - month_ago.iloc[-1]["price"] if not month_ago.empty else 0
    year_change = latest["price"] - year_ago.iloc[-1]["price"] if not year_ago.empty else 0

    kpi5_label = "Precision del Modelo"
    kpi5_value = f"{model_data['summary']['best_metrics']['R2']:.1%}" if has_model else f"{len(product_df):,}"
    if not has_model:
        kpi5_label = "Registros"

    kpis = html.Div([
        kpi_card("Precio Actual", format_price(latest["price"]),
                 delta=f"{price_change:+.2f} ({pct_change:+.1f}%)",
                 delta_positive=price_change >= 0),
        kpi_card("Fecha", latest["date"].strftime("%Y-%m-%d")),
        kpi_card("Cambio 30 Dias", f"{month_change:+.2f} MXN",
                 delta_positive=month_change >= 0),
        kpi_card("Cambio Anual", f"{year_change:+.2f} MXN",
                 delta_positive=year_change >= 0),
        kpi_card(kpi5_label, kpi5_value),
    ], className="kpi-row")

    # Price chart with MAs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["price"],
        mode="lines", name="Precio",
        line=dict(color=COLOR_ACTUAL, width=2),
        fill="tozeroy", fillcolor="rgba(45, 106, 79, 0.07)",
    ))
    ma_50 = product_df["price"].rolling(50).mean()
    ma_200 = product_df["price"].rolling(200).mean()
    fig.add_trace(go.Scatter(
        x=product_df["date"], y=ma_50,
        mode="lines", name="MA-50",
        line=dict(color="#b8860b", width=1.5, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=product_df["date"], y=ma_200,
        mode="lines", name="MA-200",
        line=dict(color="#a0522d", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **AGRO_LAYOUT, height=450,
        title="Historial de Precios con Promedios Moviles",
        xaxis_title="Fecha", yaxis_title="Precio (MXN/ton)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
    )

    stats = html.Div([
        html.H2("Estadisticas", className="section-title"),
        html.P([html.Strong("Minimo: "), format_price(product_df["price"].min())]),
        html.P([html.Strong("Maximo: "), format_price(product_df["price"].max())]),
        html.P([html.Strong("Promedio: "), format_price(product_df["price"].mean())]),
        html.P([html.Strong("Mediana: "), format_price(product_df["price"].median())]),
        html.P([html.Strong("Desv. Est.: "), f"{product_df['price'].std():.2f}"]),
        html.P([html.Strong("Registros: "), f"{len(product_df):,}"]),
        html.P([html.Strong("Desde: "), product_df["date"].min().strftime("%Y-%m-%d")]),
        html.P([html.Strong("Hasta: "), product_df["date"].max().strftime("%Y-%m-%d")]),
    ], className="stats-box")

    chart_row = html.Div([
        html.Div(dcc.Graph(figure=fig), className="col-main"),
        html.Div(stats, className="col-side"),
    ], className="two-col")

    sections = [
        html.H1(f"Resumen Ejecutivo -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.P(
            "Vision general del mercado azucarero mexicano: indicadores clave de precio, "
            "tendencias historicas, balance nacional de oferta y demanda, y distribucion anual de precios.",
            className="page-desc",
        ),
        kpis,
        html.Hr(className="section-sep"),
        chart_row,
    ]

    # Balance chart
    balance_df = model_data.get("balance")
    if balance_df is not None and not balance_df.empty:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Balance Azucarero (Fundamentales Mensuales)", className="section-title"),
            dcc.Graph(figure=make_balance_chart(balance_df)),
        ])

    # Annual boxplot
    product_df_copy = product_df.copy()
    product_df_copy["year_str"] = product_df_copy["date"].dt.year.astype(str)
    fig_box = px.box(
        product_df_copy, x="year_str", y="price",
        labels={"year_str": "Ano", "price": "Precio (MXN/ton)"},
        height=400,
        color_discrete_sequence=["#40916c"],
    )
    fig_box.update_layout(**AGRO_LAYOUT, xaxis_tickangle=-45, legend=dict(font=LEGEND_FONT))
    sections.extend([
        html.H2("Distribucion Anual de Precios", className="section-title"),
        dcc.Graph(figure=fig_box),
    ])

    return html.Div(sections)


# ---------------------------------------------------------------
# PAGE: Price History
# ---------------------------------------------------------------
def page_history(product, product_df, raw_df):
    if product_df.empty:
        return html.Div("No data available.", className="warning-box")

    date_min = product_df["date"].min()
    date_max = product_df["date"].max()

    # Daily line
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=product_df["date"], y=product_df["price"],
        mode="lines", name="Precio Diario",
        line=dict(color=COLOR_ACTUAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(45, 106, 79, 0.07)",
    ))
    fig_daily.update_layout(**AGRO_LAYOUT, height=500,
                            xaxis_title="Fecha", yaxis_title="Precio (MXN/ton)",
                            legend=dict(font=LEGEND_FONT))

    # OHLC
    monthly = product_df.set_index("date")["price"].resample("ME").agg(
        open="first", high="max", low="min", close="last"
    ).dropna()

    fig_ohlc = go.Figure(data=[go.Candlestick(
        x=monthly.index,
        open=monthly["open"], high=monthly["high"],
        low=monthly["low"], close=monthly["close"],
        increasing_line_color="#2d6a4f",
        decreasing_line_color="#a0522d",
    )])
    fig_ohlc.update_layout(**AGRO_LAYOUT, title="OHLC Mensual", height=500,
                            xaxis_rangeslider_visible=False, legend=dict(font=LEGEND_FONT))

    # Returns
    returns = product_df["price"].pct_change().dropna() * 100
    fig_ret = px.histogram(returns, nbins=80,
                            labels={"value": "Retorno Diario (%)", "count": "Frecuencia"},
                            height=400,
                            color_discrete_sequence=["#40916c"])
    fig_ret.update_layout(**AGRO_LAYOUT, title="Distribucion de Retornos Diarios (%)", legend=dict(font=LEGEND_FONT))

    # Both products
    fig_both = px.line(raw_df, x="date", y="price", color="product_type",
                        labels={"price": "Precio (MXN/ton)", "date": "Fecha", "product_type": "Tipo"},
                        height=400,
                        color_discrete_map={"estandar": "#2d6a4f", "refinada": "#b8860b"})
    fig_both.update_layout(**AGRO_LAYOUT, legend=dict(font=LEGEND_FONT))

    return html.Div([
        html.H1(f"Tendencias de Mercado -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.P(
            "Evolucion historica de precios del azucar. Explore la tendencia diaria, el resumen mensual "
            "(apertura, maximo, minimo y cierre), y la distribucion de cambios de precio. "
            "Incluye comparacion entre azucar estandar y refinada.",
            className="page-desc",
        ),
        dcc.Tabs([
            dcc.Tab(label="Linea Diaria", children=[dcc.Graph(figure=fig_daily)],
                    style={"fontFamily": "Inter"}, selected_style={"fontFamily": "Inter", "fontWeight": "600"}),
            dcc.Tab(label="OHLC Mensual", children=[dcc.Graph(figure=fig_ohlc)],
                    style={"fontFamily": "Inter"}, selected_style={"fontFamily": "Inter", "fontWeight": "600"}),
            dcc.Tab(label="Distribucion de Retornos", children=[dcc.Graph(figure=fig_ret)],
                    style={"fontFamily": "Inter"}, selected_style={"fontFamily": "Inter", "fontWeight": "600"}),
        ], colors={"border": "#e2ddd4", "primary": "#2d6a4f", "background": "#f1ede6"}),
        html.Hr(className="section-sep"),
        html.H2("Estandar vs Refinada", className="section-title"),
        dcc.Graph(figure=fig_both),
    ])


# ---------------------------------------------------------------
# PAGE: ML Model Comparison
# ---------------------------------------------------------------
def page_ml(product, product_df, model_data, has_model):
    if not has_model:
        return html.Div("No se encontraron resultados del modelo. Ejecute `py sugar_price_model.py` primero.",
                         className="warning-box")

    summary = model_data["summary"]
    metrics_df = model_data.get("metrics", pd.DataFrame())
    preds_df = model_data.get("predictions", pd.DataFrame())

    bm = summary["best_metrics"]
    ret_r2 = bm.get("Return_R2", bm.get("R2", 0))

    sections = [
        html.H1(f"Precision del Modelo -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.P(
            "Evaluamos tres modelos predictivos para encontrar el mas preciso. Cada modelo aprende "
            "la relacion entre los fundamentales del mercado (produccion, consumo, inventario) y el precio "
            "del azucar. Se validan con datos historicos reales para asegurar que las predicciones sean confiables.",
            className="page-desc",
        ),
        html.Div(
            f"Mejor Modelo: {summary['best_model']} -- Error Promedio: {bm['MAE']:.2f} MXN/ton "
            f"({bm['MAPE']:.2f}% del precio), Precision: {ret_r2:.1%}",
            className="success-box",
        ),
    ]

    # Overfit check
    train_m = summary.get("train_metrics", {})
    if train_m:
        train_ret_r2 = train_m.get("Return_R2", 0)
        gap = abs(train_ret_r2 - ret_r2)
        if gap < 0.05:
            sections.append(html.Div(
                f"Control de calidad: El modelo es consistente entre entrenamiento ({train_ret_r2:.1%}) "
                f"y prueba ({ret_r2:.1%}) -- diferencia minima ({gap:.1%})",
                className="info-box"))
        else:
            sections.append(html.Div(
                f"Atencion: Diferencia notable entre entrenamiento ({train_ret_r2:.1%}) "
                f"y prueba ({ret_r2:.1%}) -- requiere monitoreo",
                className="warning-box"))

    n_outliers = summary.get("outliers_removed", 0)
    if n_outliers:
        sections.append(html.Div(
            f"Limpieza de datos: {n_outliers} valores atipicos extremos fueron ajustados para mejorar la precision",
            className="info-box"))

    bal_feats = summary.get("balance_features_used", [])
    if bal_feats:
        sections.append(html.Div(f"Variables del mercado utilizadas: {', '.join(bal_feats)}", className="info-box"))

    # Metrics table
    if not metrics_df.empty:
        if "CV_MAE" in metrics_df.columns:
            sections.append(html.Div(
                "Los modelos se evaluan con datos historicos reales que no vieron durante el entrenamiento",
                className="info-box"))

        sections.append(html.H2("Indicadores de Precision", className="section-title"))

        fmt = {"MAE": ".2f", "RMSE": ".2f", "MAPE_pct": ".2f",
               "CV_MAE": ".2f", "CV_MAE_std": ".2f"}
        for c in ["R2_price", "Return_R2", "Change_R2", "Train_MAE", "Train_Return_R2", "Overfit_Ratio"]:
            fmt[c] = ".4f"

        display_df = metrics_df.copy()
        for col in display_df.columns:
            if col in fmt:
                display_df[col] = display_df[col].apply(
                    lambda x, f=fmt[col]: format(x, f) if pd.notna(x) else "-"
                )

        sections.append(
            dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in display_df.columns],
                style_header={
                    "backgroundColor": "#f8f6f2", "color": "#000",
                    "fontWeight": "600", "border": "1px solid #c4beb2",
                    "fontFamily": "Inter, sans-serif", "textAlign": "center",
                },
                style_cell={
                    "backgroundColor": "#ffffff", "color": "#000",
                    "border": "1px solid #e2ddd4", "textAlign": "center",
                    "fontFamily": "Inter, sans-serif", "fontSize": "0.85rem",
                    "padding": "8px 12px",
                },
                style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
            )
        )

        # Bar charts
        fig_mae = px.bar(metrics_df, x="model", y="MAE", color="model",
                          height=350, title="Error Promedio por Modelo (menor = mejor)",
                          color_discrete_sequence=AGRO_PALETTE)
        fig_mae.update_layout(**AGRO_LAYOUT, showlegend=False)

        r2_col = "R2_price" if "R2_price" in metrics_df.columns else "R2"
        fig_r2 = px.bar(metrics_df, x="model", y=r2_col, color="model",
                          height=350, title="Precision por Modelo (mayor = mejor)",
                          color_discrete_sequence=AGRO_PALETTE)
        fig_r2.update_layout(**AGRO_LAYOUT, showlegend=False)

        sections.append(html.Div([
            html.Div(dcc.Graph(figure=fig_mae), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=fig_r2), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}))

    # Predictions vs Actual
    if not preds_df.empty:
        actual_col = "actual_target" if "actual_target" in preds_df.columns else "actual"
        skip_cols = {"date", "actual", "actual_today", "actual_target"}
        model_cols = [c for c in preds_df.columns if c not in skip_cols]

        sections.append(html.H2("Predicciones vs Precio Real", className="section-title"))

        sections.append(dcc.Dropdown(
            id="ml-model-select",
            options=[{"label": c.replace("_", " ").title(), "value": c} for c in model_cols],
            value=model_cols,
            multi=True,
            style={"marginBottom": "12px"},
        ))
        sections.append(html.Div(id="ml-predictions-chart-container"))
        sections.append(dcc.Store(id="ml-product-store", data=product))

    # Residuals
    if not preds_df.empty:
        actual_col = "actual_target" if "actual_target" in preds_df.columns else "actual"
        best_daily_name = summary.get("best_daily_model", summary["best_model"])
        best_col = best_daily_name.replace("Daily ", "").lower().replace(" ", "_")
        if best_col in preds_df.columns:
            residuals = preds_df[actual_col] - preds_df[best_col]

            sections.append(html.H2("Analisis de Errores de Prediccion", className="section-title"))

            fig_res = px.scatter(x=preds_df["date"], y=residuals,
                                  labels={"x": "Fecha", "y": "Error (MXN)"},
                                  height=350, title="Errores de Prediccion en el Tiempo",
                                  color_discrete_sequence=["#b8860b"])
            fig_res.update_layout(**AGRO_LAYOUT, legend=dict(font=LEGEND_FONT))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#6b7f6e", opacity=0.5)

            fig_rh = px.histogram(residuals, nbins=60,
                                   labels={"value": "Error", "count": "Frecuencia"},
                                   height=350, title="Distribucion de Errores",
                                   color_discrete_sequence=["#40916c"])
            fig_rh.update_layout(**AGRO_LAYOUT, legend=dict(font=LEGEND_FONT))

            sections.append(html.Div([
                html.Div(dcc.Graph(figure=fig_res), style={"flex": "1"}),
                html.Div(dcc.Graph(figure=fig_rh), style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px"}))

    return html.Div(sections)


# ---------------------------------------------------------------
# PAGE: Forecast
# ---------------------------------------------------------------
def page_forecast(product, product_df, model_data, has_model):
    if not has_model:
        return html.Div("Ejecute el pipeline ML primero: `py sugar_price_model.py`", className="warning-box")

    forecast_df = model_data.get("forecast", pd.DataFrame())
    summary = model_data["summary"]

    if forecast_df.empty:
        return html.Div("No se encontraron datos de pronostico.", className="warning-box")

    forecast_days = summary.get("forecast_days", 90)
    monthly_r2 = summary.get("monthly_metrics", {}).get("R2", summary["best_metrics"].get("R2", 0))

    sections = [
        html.H1(f"Proyeccion de Precios -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.Div(
            f"Modelo: {summary['best_model']} -- Error promedio: {summary['best_metrics']['MAE']:.2f} MXN/ton "
            f"({summary['best_metrics']['MAPE']:.2f}% del precio), Precision: {monthly_r2:.1%}",
            className="info-box",
        ),
        html.P(
            "Proyeccion de precios basada en los fundamentales del mercado azucarero: produccion, consumo, "
            "inventario, oferta y demanda. Las bandas sombreadas muestran el rango de confianza de la "
            "prediccion (50% y 80%), indicando donde es mas probable que se ubique el precio real.",
            className="page-desc",
        ),
    ]

    # Monthly forecast table
    monthly_fc = model_data.get("monthly_forecast", pd.DataFrame())
    if not monthly_fc.empty:
        sections.append(html.H2("Proyeccion Mensual de Precios", className="section-title"))
        mf_display = monthly_fc.copy()
        mf_display["period"] = mf_display["year"].astype(int).astype(str) + "-" + mf_display["month"].astype(int).astype(str).str.zfill(2)
        mf_display["predicted_price"] = mf_display["predicted_price"].apply(lambda x: f"${x:,.2f}")
        sections.append(
            dash_table.DataTable(
                data=mf_display[["period", "predicted_price"]].to_dict("records"),
                columns=[
                    {"name": "Periodo", "id": "period"},
                    {"name": "Precio Predicho", "id": "predicted_price"},
                ],
                style_header={
                    "backgroundColor": "#f8f6f2", "color": "#000",
                    "fontWeight": "600", "border": "1px solid #c4beb2",
                    "fontFamily": "Inter, sans-serif", "textAlign": "center",
                },
                style_cell={
                    "backgroundColor": "#ffffff", "color": "#000",
                    "border": "1px solid #e2ddd4", "textAlign": "center",
                    "fontFamily": "Inter, sans-serif", "padding": "8px 12px",
                },
                style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px",
                              "overflow": "hidden", "maxWidth": "400px", "margin": "0 auto"},
            )
        )

    # Slider for history days
    sections.append(html.Div([
        html.Label("Dias de historial a mostrar", style={"fontWeight": "600", "color": "#1a3a2a", "fontSize": "0.85rem"}),
        dcc.Slider(id="forecast-history-slider", min=30, max=365, value=180, step=10,
                   marks={30: "30", 90: "90", 180: "180", 365: "365"},
                   tooltip={"placement": "bottom"}),
    ], style={"maxWidth": "500px", "margin": "16px 0"}))

    # Forecast chart placeholder (updated by callback)
    sections.append(html.Div(id="forecast-chart-container"))
    sections.append(dcc.Store(id="forecast-product-store", data=product))

    forecast_months = summary.get("forecast_months", 6)

    # KPIs
    end_price = forecast_df["predicted_price"].iloc[-1]
    diff = end_price - product_df["price"].iloc[-1]
    sections.append(html.Div([
        kpi_card("Precio Actual", format_price(product_df["price"].iloc[-1])),
        kpi_card(f"Precio Proyectado ({forecast_months}m)", format_price(end_price),
                 delta=f"{diff:+.2f} MXN", delta_positive=diff >= 0),
        kpi_card("Minimo Esperado", format_price(forecast_df["predicted_price"].min())),
        kpi_card("Maximo Esperado", format_price(forecast_df["predicted_price"].max())),
    ], className="kpi-row"))

    # Forecast table
    fc_display = forecast_df.copy()
    fc_display["predicted_price"] = fc_display["predicted_price"].apply(lambda x: f"${x:,.2f}")
    fc_display["date"] = fc_display["date"].dt.strftime("%Y-%m-%d")
    cols_show = ["date", "predicted_price"]
    for ci_col in ["ci_lower_80", "ci_upper_80", "ci_lower_50", "ci_upper_50"]:
        if ci_col in fc_display.columns:
            fc_display[ci_col] = fc_display[ci_col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
            cols_show.append(ci_col)

    sections.append(
        dash_table.DataTable(
            data=fc_display[cols_show].to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols_show],
            style_header={
                "backgroundColor": "#f8f6f2", "color": "#000",
                "fontWeight": "600", "border": "1px solid #c4beb2",
                "fontFamily": "Inter, sans-serif", "textAlign": "center",
            },
            style_cell={
                "backgroundColor": "#ffffff", "color": "#000",
                "border": "1px solid #e2ddd4", "textAlign": "center",
                "fontFamily": "Inter, sans-serif", "fontSize": "0.82rem",
                "padding": "6px 10px",
            },
            style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
            page_size=15,
        )
    )

    # ==================================================================
    # VARIABLES BEHIND THE PROJECTION
    # ==================================================================
    monthly_data = model_data.get("monthly_data", pd.DataFrame())
    balance_df = model_data.get("balance")
    external_df = model_data.get("external")

    if not monthly_data.empty:
        md = monthly_data.copy()
        md["period"] = md["year"].astype(int).astype(str) + "-" + md["month"].astype(int).astype(str).str.zfill(2)

        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Variables que Impulsan la Proyeccion", className="section-title"),
            html.P(
                "Estas son las variables fundamentales y de mercado que el modelo utiliza para calcular "
                "la proyeccion de precio. La linea punteada indica el ultimo valor utilizado para proyectar.",
                className="page-desc",
            ),
        ])

        # --- Production vs Consumption ---
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Scatter(
            x=md["period"], y=md["produccion"],
            mode="lines", name="Produccion",
            line=dict(color="#2d6a4f", width=2),
            fill="tozeroy", fillcolor="rgba(45, 106, 79, 0.06)",
        ))
        fig_pc.add_trace(go.Scatter(
            x=md["period"], y=md["consumo_nacional_aparente"],
            mode="lines", name="Consumo Nacional Aparente",
            line=dict(color="#b8860b", width=2),
        ))
        fig_pc.add_trace(go.Scatter(
            x=md["period"], y=md["avg_price"],
            mode="lines", name="Precio Promedio",
            line=dict(color="#a0522d", width=1.5, dash="dot"),
            yaxis="y2",
        ))
        fig_pc.update_layout(
            **AGRO_LAYOUT, height=450,
            title="Produccion vs Consumo Nacional & Precio",
            xaxis_title="Mes", yaxis_title="Toneladas",
            yaxis2=dict(title="Precio (MXN/ton)", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#a0522d"), title_font=dict(color="#a0522d")),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_pc))

        # --- Inventory & Supply/Demand ---
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(
            x=md["period"], y=md["oferta_total"],
            name="Oferta Total", marker_color="#2d6a4f", opacity=0.7,
        ))
        fig_inv.add_trace(go.Bar(
            x=md["period"], y=md["demanda_total"],
            name="Demanda Total", marker_color="#a0522d", opacity=0.7,
        ))
        fig_inv.add_trace(go.Scatter(
            x=md["period"], y=md["inventario_final"],
            mode="lines+markers", name="Inventario Final",
            line=dict(color="#b8860b", width=2.5),
            marker=dict(size=3),
            yaxis="y2",
        ))
        fig_inv.update_layout(
            **AGRO_LAYOUT, height=450, barmode="group",
            title="Oferta vs Demanda & Inventario Final",
            xaxis_title="Mes", yaxis_title="Toneladas",
            yaxis2=dict(title="Inventario Final (ton)", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#b8860b"), title_font=dict(color="#b8860b")),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_inv))

        # --- Derived Ratios ---
        if "supply_demand_ratio" in md.columns and "inventory_months" in md.columns:
            fig_ratios = go.Figure()
            fig_ratios.add_trace(go.Scatter(
                x=md["period"], y=md["supply_demand_ratio"],
                mode="lines", name="Oferta / Demanda",
                line=dict(color="#2d6a4f", width=2),
            ))
            fig_ratios.add_trace(go.Scatter(
                x=md["period"], y=md["demand_pressure"],
                mode="lines", name="Presion de Demanda",
                line=dict(color="#a0522d", width=2),
                yaxis="y2",
            ))
            fig_ratios.update_layout(
                **AGRO_LAYOUT, height=380,
                title="Indicadores Derivados: Oferta/Demanda & Presion de Demanda",
                xaxis_title="Mes", yaxis_title="Ratio O/D",
                yaxis2=dict(title="Presion de Demanda", overlaying="y", side="right", showgrid=False,
                            tickfont=dict(color="#a0522d"), title_font=dict(color="#a0522d")),
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
            )
            sections.append(dcc.Graph(figure=fig_ratios))

        # --- Imports & Exports ---
        fig_trade = go.Figure()
        fig_trade.add_trace(go.Bar(
            x=md["period"], y=md["importaciones"],
            name="Importaciones", marker_color="#40916c",
        ))
        fig_trade.add_trace(go.Bar(
            x=md["period"], y=md["exportaciones_totales"],
            name="Exportaciones Totales", marker_color="#d4a843",
        ))
        if "net_exports" in md.columns:
            fig_trade.add_trace(go.Scatter(
                x=md["period"], y=md["net_exports"],
                mode="lines+markers", name="Exportaciones Netas",
                line=dict(color="#a0522d", width=2, dash="dash"),
                marker=dict(size=3),
            ))
        fig_trade.update_layout(
            **AGRO_LAYOUT, height=400, barmode="group",
            title="Comercio Exterior: Importaciones, Exportaciones & Balance Neto",
            xaxis_title="Mes", yaxis_title="Toneladas",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_trade))

    # --- External Market Variables (Futures, FX, WTI) ---
    if external_df is not None and not external_df.empty:
        ext = external_df.copy()
        ext_month_col = "month" if "month" in ext.columns else "month_number"
        ext["period"] = ext["year"].astype(int).astype(str) + "-" + ext[ext_month_col].astype(int).astype(str).str.zfill(2)

        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Variables de Mercado Externo en la Proyeccion", className="section-title"),
            html.P(
                "El modelo tambien incorpora datos internacionales: futuros de azucar ICE, "
                "tipo de cambio USD/MXN y precio del petroleo WTI.",
                className="page-desc",
            ),
        ])

        # ICE Futures
        ice11 = ext.dropna(subset=["ice_no11"])
        ice16 = ext.dropna(subset=["ice_no16"])
        if not ice11.empty or not ice16.empty:
            fig_ice = go.Figure()
            if not ice11.empty:
                fig_ice.add_trace(go.Scatter(
                    x=ice11["period"], y=ice11["ice_no11"],
                    mode="lines", name="ICE No.11 (Mundo)",
                    line=dict(color="#2d6a4f", width=2),
                    fill="tozeroy", fillcolor="rgba(45, 106, 79, 0.06)",
                ))
            if not ice16.empty:
                fig_ice.add_trace(go.Scatter(
                    x=ice16["period"], y=ice16["ice_no16"],
                    mode="lines", name="ICE No.16 (EE.UU.)",
                    line=dict(color="#b8860b", width=2),
                ))
            fig_ice.update_layout(
                **AGRO_LAYOUT, height=400,
                title="Futuros de Azucar Utilizados en la Proyeccion",
                xaxis_title="Mes", yaxis_title="Centavos por Libra (USD)",
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
            )
            sections.append(dcc.Graph(figure=fig_ice))

        # USD/MXN + WTI
        fx = ext.dropna(subset=["usd_mxn"])
        wti = ext.dropna(subset=["wti"])
        if not fx.empty or not wti.empty:
            fig_fx = go.Figure()
            if not fx.empty:
                fig_fx.add_trace(go.Scatter(
                    x=fx["period"], y=fx["usd_mxn"],
                    mode="lines", name="USD/MXN",
                    line=dict(color="#2d6a4f", width=2),
                ))
            if not wti.empty:
                fig_fx.add_trace(go.Scatter(
                    x=wti["period"], y=wti["wti"],
                    mode="lines", name="WTI (USD/bbl)",
                    line=dict(color="#a0522d", width=2),
                    yaxis="y2",
                ))
            fig_fx.update_layout(
                **AGRO_LAYOUT, height=400,
                title="Tipo de Cambio & Petroleo en la Proyeccion",
                xaxis_title="Mes", yaxis_title="MXN por USD",
                yaxis2=dict(title="USD/bbl", overlaying="y", side="right", showgrid=False,
                            tickfont=dict(color="#a0522d"), title_font=dict(color="#a0522d")),
                xaxis_tickangle=-45,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
            )
            sections.append(dcc.Graph(figure=fig_fx))

    # --- KPIs of current variable values used in projection ---
    latest_balance = model_data.get("latest_balance", {})
    if latest_balance:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Valores Actuales Utilizados en la Proyeccion", className="section-title"),
            html.P(
                "Estos son los valores mas recientes de cada variable que el modelo utiliza "
                "como base para calcular la proyeccion de precio.",
                className="page-desc",
            ),
        ])
        var_kpis = []
        for key, label in [
            ("produccion", "Produccion"), ("consumo_nacional_aparente", "Consumo Nac."),
            ("inventario_final", "Inv. Final"), ("oferta_total", "Oferta Total"),
            ("demanda_total", "Demanda Total"), ("importaciones", "Importaciones"),
            ("exportaciones_totales", "Exportaciones"),
        ]:
            val = latest_balance.get(key, 0)
            if val:
                var_kpis.append(kpi_card(label, format_tons(val)))
        if var_kpis:
            sections.append(html.Div(var_kpis, className="kpi-row"))

        ext_kpis = []
        for key, label, fmt_str in [
            ("usd_mxn", "USD/MXN", "{:.2f}"), ("ice_no11", "ICE No.11", "{:.2f} c/lb"),
            ("ice_no16", "ICE No.16", "{:.2f} c/lb"), ("wti", "WTI", "${:.2f} USD"),
        ]:
            val = latest_balance.get(key, 0)
            if val:
                ext_kpis.append(kpi_card(label, fmt_str.format(val)))
        if ext_kpis:
            sections.append(html.Div(ext_kpis, className="kpi-row"))

    return html.Div(sections)


# Callback: update forecast chart when slider changes
@callback(
    Output("forecast-chart-container", "children"),
    Input("forecast-history-slider", "value"),
    State("forecast-product-store", "data"),
    prevent_initial_call=False,
)
def update_forecast_chart(history_days, product):
    raw_df = load_raw_prices()
    model_data = load_model_outputs(product)
    product_df = raw_df[raw_df["product_type"] == product].copy()
    forecast_df = model_data.get("forecast", pd.DataFrame())
    summary = model_data.get("summary", {})

    if forecast_df.empty or not summary:
        return html.Div("No forecast data.", className="warning-box")

    recent = product_df.tail(history_days or 180)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["price"],
        mode="lines", name="Historico",
        line=dict(color=COLOR_ACTUAL, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["predicted_price"],
        mode="lines", name="Pronostico",
        line=dict(color=COLOR_FORECAST, width=2.5, dash="dash"),
    ))

    if "ci_upper_80" in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["ci_upper_80"], forecast_df["ci_lower_80"][::-1]]),
            fill="toself", fillcolor="rgba(184, 134, 11, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% IC", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["ci_upper_50"], forecast_df["ci_lower_50"][::-1]]),
            fill="toself", fillcolor="rgba(184, 134, 11, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="50% IC", showlegend=True,
        ))
    else:
        mae = summary["best_metrics"]["MAE"]
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([
                forecast_df["predicted_price"] + mae,
                (forecast_df["predicted_price"] - mae)[::-1],
            ]),
            fill="toself", fillcolor="rgba(184, 134, 11, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"+/-MAE ({mae:.0f})", showlegend=True,
        ))

    forecast_months = summary.get("forecast_months", 6)
    fig.update_layout(
        **AGRO_LAYOUT, height=550,
        xaxis_title="Fecha", yaxis_title="Precio (MXN/ton)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        title=f"{summary['best_metrics']['MAPE']:.1f}% error promedio -- Pronostico a {forecast_months} meses",
    )
    return dcc.Graph(figure=fig)


# Primary variables (production & national balance) -- emphasized in UI
PRIMARY_SCENARIO_VARS = [
    ("produccion", "Produccion Nacional (ton/ano)"),
    ("consumo_nacional_aparente", "Consumo Nacional Aparente (ton/ano)"),
    ("inventario_final", "Inventario Final (ton)"),
]
SECONDARY_SCENARIO_VARS = [
    ("importaciones", "Importaciones (ton/ano)"),
    ("exportaciones_totales", "Exportaciones Totales (ton/ano)"),
    ("inventario_inicial", "Inventario Inicial (ton)"),
]

EXTERNAL_SCENARIO_VARS = [
    ("usd_mxn", "Tipo de Cambio USD/MXN"),
    ("ice_no11", "Azucar ICE No.11 (cents/lb)"),
    ("ice_no16", "Azucar ICE No.16 (cents/lb)"),
    ("wti", "Petroleo WTI (USD/bbl)"),
    ("brl_usd", "Tipo de Cambio BRL/USD"),
]

DERIVED_RATIO_LABELS = {
    "oferta_total": "Oferta Total (ton/ano)",
    "demanda_total": "Demanda Total (ton/ano)",
    "supply_demand_ratio": "Oferta / Demanda",
    "inventory_months": "Meses de Inventario",
    "demand_pressure": "Presion de Demanda",
    "production_share": "Participacion Produccion",
    "inventory_to_consumption": "Inventario / Consumo",
    "net_exports": "Exportaciones Netas (ton/ano)",
    "excess_supply": "Exceso de Oferta (ton/ano)",
}


# ---------------------------------------------------------------
# PAGE: Scenario Predictor
# ---------------------------------------------------------------
def page_scenario(product, product_df, model_data, has_model):
    if not has_model or "model" not in model_data:
        return html.Div("Ejecute el pipeline ML primero: `py sugar_price_model.py`", className="warning-box")

    latest_balance = model_data.get("latest_balance", {})
    balance_df = model_data.get("balance")

    # --- Aggregate balance to annual totals ---
    # Only pure flow variables are summed; oferta_total/demanda_total are computed
    FLOW_COLS = ["produccion", "importaciones", "exportaciones_totales",
                 "consumo_nacional_aparente"]
    STOCK_COLS_FIRST = ["inventario_inicial"]   # first month of year
    STOCK_COLS_LAST  = ["inventario_final"]     # last month of year

    annual_balance = None
    if balance_df is not None and not balance_df.empty:
        _bdf = balance_df.copy()
        _bdf["year"] = _bdf["year"].astype(int)
        _agg = {}
        for c in FLOW_COLS:
            if c in _bdf.columns:
                _agg[c] = "sum"
        _ann = _bdf.groupby("year").agg(_agg).reset_index()
        # Stock cols: inventario_inicial = first month, inventario_final = last month
        for c in STOCK_COLS_FIRST:
            if c in _bdf.columns:
                _first = _bdf.sort_values("month_number").groupby("year")[c].first().reset_index()
                _ann = _ann.merge(_first, on="year", how="left")
        for c in STOCK_COLS_LAST:
            if c in _bdf.columns:
                _last = _bdf.sort_values("month_number").groupby("year")[c].last().reset_index()
                _ann = _ann.merge(_last, on="year", how="left")
        # Compute oferta_total and demanda_total from annual components
        _ann["oferta_total"] = _ann.get("inventario_inicial", 0) + _ann.get("produccion", 0) + _ann.get("importaciones", 0)
        _ann["demanda_total"] = _ann.get("consumo_nacional_aparente", 0) + _ann.get("exportaciones_totales", 0)
        # Recompute derived ratios from annual totals
        _ann["supply_demand_ratio"] = _ann["oferta_total"] / _ann["demanda_total"].replace(0, 1)
        _ann["inventory_months"] = _ann["inventario_final"] / (_ann["consumo_nacional_aparente"].replace(0, 1) / 12)
        annual_balance = _ann

    # Latest annual balance for slider defaults
    latest_annual = {}
    if annual_balance is not None and not annual_balance.empty:
        _row = annual_balance.iloc[-1]
        for c in FLOW_COLS + STOCK_COLS_FIRST + STOCK_COLS_LAST + ["oferta_total", "demanda_total"]:
            if c in _row.index:
                latest_annual[c] = float(_row[c])
        latest_annual["supply_demand_ratio"] = float(_row.get("supply_demand_ratio", 1))
        latest_annual["inventory_months"] = float(_row.get("inventory_months", 0))

    sections = [
        html.H1(f"Analisis de Escenarios -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.P(
            "Simule diferentes condiciones anuales de mercado para ver como afectarian el precio del azucar. "
            "Ajuste produccion, consumo, inventario y condiciones internacionales a nivel anual, y compare "
            "el resultado contra la proyeccion base. Los indicadores derivados se calculan automaticamente.",
            className="page-desc",
        ),
    ]

    # ---- Annual Balance History Table ----
    if annual_balance is not None and not annual_balance.empty:
        sections.extend([
            html.H2("Balance Azucarero Nacional Anual", className="section-title"),
        ])

        bal_recent = annual_balance.tail(10).copy().reset_index(drop=True)
        bal_recent["Ano"] = bal_recent["year"].astype(int).astype(str)

        display_cols = [
            ("Ano", "Ano"),
            ("produccion", "Produccion"),
            ("consumo_nacional_aparente", "Consumo Nac."),
            ("inventario_final", "Inv. Final"),
            ("oferta_total", "Oferta Total"),
            ("demanda_total", "Demanda Total"),
            ("importaciones", "Importaciones"),
            ("exportaciones_totales", "Exportaciones"),
            ("supply_demand_ratio", "Oferta/Demanda"),
            ("inventory_months", "Meses Inv."),
        ]

        bal_display = pd.DataFrame()
        for col_id, col_name in display_cols:
            if col_id in bal_recent.columns:
                if col_id in ("supply_demand_ratio", "inventory_months"):
                    bal_display[col_name] = bal_recent[col_id].apply(lambda x: f"{x:.2f}")
                elif col_id == "Ano":
                    bal_display[col_name] = bal_recent[col_id].values
                else:
                    bal_display[col_name] = bal_recent[col_id].apply(lambda x: f"{x:,.0f}")

        sections.append(
            dash_table.DataTable(
                data=bal_display.to_dict("records"),
                columns=[{"name": c, "id": c} for c in bal_display.columns],
                style_header={
                    "backgroundColor": "#f8f6f2", "color": "#000",
                    "fontWeight": "600", "border": "1px solid #c4beb2",
                    "fontFamily": "Inter, sans-serif", "textAlign": "center",
                    "fontSize": "0.78rem",
                },
                style_cell={
                    "backgroundColor": "#ffffff", "color": "#000",
                    "border": "1px solid #e2ddd4", "textAlign": "center",
                    "fontFamily": "Inter, sans-serif", "fontSize": "0.78rem",
                    "padding": "6px 8px",
                },
                style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
                style_data_conditional=[
                    {"if": {"column_id": "Produccion"},
                     "backgroundColor": "rgba(45, 106, 79, 0.06)", "fontWeight": "600"},
                    {"if": {"column_id": "Consumo Nac."},
                     "backgroundColor": "rgba(184, 134, 11, 0.06)", "fontWeight": "600"},
                    {"if": {"column_id": "Inv. Final"},
                     "backgroundColor": "rgba(160, 82, 45, 0.06)", "fontWeight": "600"},
                ],
            )
        )

        # Annual Production vs Consumption chart
        period = bal_recent["Ano"]
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Bar(x=period, y=bal_recent["produccion"], name="Produccion",
                                marker_color="#2d6a4f"))
        fig_pc.add_trace(go.Bar(x=period, y=bal_recent["consumo_nacional_aparente"], name="Consumo Nacional",
                                marker_color="#b8860b"))
        fig_pc.add_trace(go.Scatter(x=period, y=bal_recent["inventario_final"], name="Inventario Final",
                                    mode="lines+markers", line=dict(color="#a0522d", width=2.5),
                                    yaxis="y2"))
        fig_pc.update_layout(
            **AGRO_LAYOUT, height=380, barmode="group",
            title="Produccion vs Consumo Nacional e Inventario (Anual)",
            xaxis_title="Ano", yaxis_title="Toneladas",
            yaxis2=dict(title="Inventario Final (ton)", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(color="#a0522d"),
                        title_font=dict(color="#a0522d")),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_pc))

    sections.append(html.Hr(className="section-sep"))

    # ---- Current annual balance KPIs ----
    _last_year_label = str(int(annual_balance.iloc[-1]["year"])) if annual_balance is not None and not annual_balance.empty else "N/A"
    sections.append(html.H2(f"Balance Anual ({_last_year_label})", className="section-title"))
    bal_kpis = []
    for key, label in list(BALANCE_LABELS.items())[:5]:  # Show top 5
        val = latest_annual.get(key, 0)
        bal_kpis.append(kpi_card(label.split("(")[0].strip(), format_tons(val)))
    sections.append(html.Div(bal_kpis, className="kpi-row"))

    # Derived ratios KPIs (annual)
    ratio_kpis = []
    for key in ["supply_demand_ratio", "inventory_months"]:
        val = latest_annual.get(key, 0)
        ratio_kpis.append(kpi_card(DERIVED_RATIO_LABELS[key], f"{val:.2f}"))
    sections.append(html.Div(ratio_kpis, className="kpi-row"))

    sections.append(html.Hr(className="section-sep"))

    # ---- PRIMARY VARIABLES (Production & National Balance) ----
    sections.extend([
        html.H2("Variables Principales (Anuales)", className="section-title"),
        html.Div(
            "Estos son los factores anuales con mayor impacto en el precio. "
            "La produccion y el consumo nacional son los principales determinantes. "
            "Los valores representan totales anuales.",
            className="info-box",
        ),
    ])

    def make_slider(key, label, annual_vals):
        current_val = annual_vals.get(key, 0)
        if current_val == 0:
            min_val, max_val, default_val = 0, 50_000_000, 0
        else:
            min_val = 0
            max_val = int(current_val * 3)
            default_val = int(current_val)
        step = max(1, int(max_val / 200))
        return html.Div([
            html.Label(label, style={"fontWeight": "600", "color": "#1a3a2a", "fontSize": "0.85rem"}),
            html.Div(f"Actual anual: {format_tons(current_val)}",
                     style={"fontSize": "0.75rem", "color": "#6b7f6e", "marginBottom": "4px"}),
            dcc.Slider(
                id=f"scenario-{key}",
                min=min_val, max=max_val, value=default_val, step=step,
                marks={min_val: format_tons(min_val),
                       default_val: {"label": format_tons(default_val), "style": {"color": "#2d6a4f", "fontWeight": "600"}},
                       max_val: format_tons(max_val)},
                tooltip={"placement": "bottom"},
            ),
        ], style={"marginBottom": "18px"})

    primary_sliders = []
    for key, label in PRIMARY_SCENARIO_VARS:
        primary_sliders.append(make_slider(key, label, latest_annual))

    # Two columns for primary
    half_p = (len(primary_sliders) + 1) // 2
    sections.append(html.Div([
        html.Div(primary_sliders[:half_p], style={"flex": "1"}),
        html.Div(primary_sliders[half_p:], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "24px"}))

    # Hidden sliders for oferta_total and demanda_total (computed, not user-set)
    sections.append(html.Div(
        dcc.Slider(id="scenario-oferta_total", min=0, max=1, value=0),
        style={"display": "none"},
    ))
    sections.append(html.Div(
        dcc.Slider(id="scenario-demanda_total", min=0, max=1, value=0),
        style={"display": "none"},
    ))

    # ---- SECONDARY VARIABLES ----
    sections.extend([
        html.Hr(className="section-sep"),
        html.H2("Variables Secundarias (Anuales)", className="section-title"),
        html.P("Comercio exterior e inventario inicial (totales anuales).", className="page-desc"),
    ])

    secondary_sliders = []
    for key, label in SECONDARY_SCENARIO_VARS:
        secondary_sliders.append(make_slider(key, label, latest_annual))

    sections.append(html.Div(secondary_sliders, style={"maxWidth": "700px"}))

    # ---- EXTERNAL MARKET VARIABLES ----
    has_external = any(latest_balance.get(k) for k, _ in EXTERNAL_SCENARIO_VARS)

    # Custom ranges for external variables
    EXT_RANGES = {
        "usd_mxn": (10, 30, 0.1),      # MXN per USD
        "ice_no11": (5, 50, 0.5),       # cents/lb
        "ice_no16": (10, 80, 0.5),      # cents/lb
        "wti": (20, 150, 1),            # USD/bbl
        "brl_usd": (2, 8, 0.05),        # BRL per USD
    }

    if has_external:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Variables de Mercado Externo (Promedios Anuales)", className="section-title"),
            html.P(
                "Condiciones del mercado internacional: tipo de cambio, precio internacional del azucar "
                "y petroleo. Estos valores representan promedios anuales esperados.",
                className="page-desc",
            ),
        ])

        external_sliders = []
        for key, label in EXTERNAL_SCENARIO_VARS:
            current_val = latest_balance.get(key, 0)
            if current_val == 0:
                # Hidden slider with default value so callbacks work
                sections.append(html.Div(
                    dcc.Slider(id=f"scenario-{key}", min=0, max=1, value=0),
                    style={"display": "none"},
                ))
                continue
            lo, hi, step = EXT_RANGES.get(key, (0, current_val * 2, current_val / 100))
            lo = min(lo, current_val * 0.5)
            hi = max(hi, current_val * 1.5)
            external_sliders.append(html.Div([
                html.Label(label, style={"fontWeight": "600", "color": "#1a3a2a", "fontSize": "0.85rem"}),
                html.Div(f"Actual: {current_val:.2f}",
                         style={"fontSize": "0.75rem", "color": "#6b7f6e", "marginBottom": "4px"}),
                dcc.Slider(
                    id=f"scenario-{key}",
                    min=round(lo, 2), max=round(hi, 2), value=round(current_val, 2), step=step,
                    marks={
                        round(lo, 2): f"{lo:.1f}",
                        round(current_val, 2): {"label": f"{current_val:.2f}", "style": {"color": "#2d6a4f", "fontWeight": "600"}},
                        round(hi, 2): f"{hi:.1f}",
                    },
                    tooltip={"placement": "bottom"},
                ),
            ], style={"marginBottom": "18px"}))

        if external_sliders:
            half_e = (len(external_sliders) + 1) // 2
            sections.append(html.Div([
                html.Div(external_sliders[:half_e], style={"flex": "1"}),
                html.Div(external_sliders[half_e:], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "24px"}))
    else:
        # No external data -- create hidden sliders so callbacks don't break
        for key, _ in EXTERNAL_SCENARIO_VARS:
            sections.append(html.Div(
                dcc.Slider(id=f"scenario-{key}", min=0, max=1, value=0),
                style={"display": "none"},
            ))

    # ---- Derived ratios preview ----
    sections.extend([
        html.Hr(className="section-sep"),
        html.H2("Indicadores Calculados del Escenario (Anuales)", className="section-title"),
        html.P("Estos indicadores se calculan automaticamente a partir de las variables anuales del escenario.",
               className="page-desc"),
        html.Div(id="scenario-derived-ratios"),
    ])

    # Horizon selector + button
    sections.extend([
        html.Hr(className="section-sep"),
        html.Div([
            html.Label("Horizonte de pronostico (anos)", style={"fontWeight": "600", "color": "#1a3a2a", "fontSize": "0.85rem"}),
            dcc.Dropdown(
                id="scenario-horizon",
                options=[{"label": f"{y} {'ano' if y == 1 else 'anos'}", "value": y} for y in [1, 2, 3, 4, 5]],
                value=1,
                clearable=False,
                style={"width": "180px"},
            ),
        ], style={"marginBottom": "16px"}),
        html.Button("Ejecutar Pronostico de Escenario", id="scenario-run-btn", className="btn-primary",
                     style={"maxWidth": "400px"}),
        dcc.Loading(
            html.Div(id="scenario-results"),
            type="circle", color="#2d6a4f",
        ),
    ])

    # Store product info for the callback
    sections.append(dcc.Store(id="scenario-product-store", data=product))

    return html.Div(sections)


# Callback: update derived ratios preview when sliders change
@callback(
    Output("scenario-derived-ratios", "children"),
    [Input(f"scenario-{key}", "value") for key, _ in PRIMARY_SCENARIO_VARS + SECONDARY_SCENARIO_VARS],
)
def update_derived_ratios(*vals):
    all_vars = PRIMARY_SCENARIO_VARS + SECONDARY_SCENARIO_VARS
    scenario = {all_vars[i][0]: (vals[i] or 0) for i in range(len(all_vars))}

    prod = scenario.get("produccion", 0)
    imp = scenario.get("importaciones", 0)
    exp = scenario.get("exportaciones_totales", 0)
    cna = scenario.get("consumo_nacional_aparente", 1) or 1
    inv_i = scenario.get("inventario_inicial", 0)
    inv_f = scenario.get("inventario_final", 0)
    # Compute oferta_total and demanda_total from components
    ot = inv_i + prod + imp
    dt_ = cna + exp

    ratios = {
        "oferta_total": ot,
        "demanda_total": dt_,
        "supply_demand_ratio": ot / dt_ if dt_ else 1,
        "inventory_months": inv_f / (cna / 12) if cna else 0,
        "demand_pressure": (cna + exp) / (prod + imp + inv_i) if (prod + imp + inv_i) else 1,
        "production_share": prod / ot if ot else 0,
        "inventory_to_consumption": inv_f / cna if cna else 1,
        "net_exports": exp - imp,
        "excess_supply": (prod + imp + inv_i) - (cna + exp),
    }

    cards = []
    for key, label in DERIVED_RATIO_LABELS.items():
        val = ratios.get(key, 0)
        if key in ("net_exports", "excess_supply", "oferta_total", "demanda_total"):
            display = format_tons(val)
        else:
            display = f"{val:.3f}"
        cards.append(kpi_card(label, display))

    return html.Div(cards, className="kpi-row")


_ALL_SCENARIO_KEYS = [k for k, _ in PRIMARY_SCENARIO_VARS + SECONDARY_SCENARIO_VARS + EXTERNAL_SCENARIO_VARS]

@callback(
    Output("scenario-results", "children"),
    Input("scenario-run-btn", "n_clicks"),
    State("scenario-product-store", "data"),
    *[State(f"scenario-{key}", "value") for key in _ALL_SCENARIO_KEYS],
    State("scenario-horizon", "value"),
    prevent_initial_call=True,
)
def run_scenario(n_clicks, product, *args):
    if not n_clicks:
        return no_update

    scenario_years = args[-1] or 1
    scenario_months = int(scenario_years) * 12
    annual_scenario = {_ALL_SCENARIO_KEYS[i]: (args[i] or 0) for i in range(len(_ALL_SCENARIO_KEYS))}

    raw_df = load_raw_prices()
    model_data = load_model_outputs(product)
    product_df = raw_df[raw_df["product_type"] == product].copy()

    model = model_data["model"]
    scaler = model_data.get("scaler", None)
    feature_cols = model_data.get("feature_cols", [])
    monthly_df = model_data.get("monthly_data", pd.DataFrame())
    bal_df = model_data.get("balance")
    ext_df = model_data.get("external")
    latest_balance = model_data.get("latest_balance", {})

    # Convert annual scenario values to monthly for the model
    # Flow variables: divide by 12; stock variables & external: keep as-is
    _FLOW_KEYS = {"produccion", "importaciones", "exportaciones_totales",
                   "consumo_nacional_aparente"}
    monthly_scenario = {}
    for k, v in annual_scenario.items():
        if k in _FLOW_KEYS:
            monthly_scenario[k] = v / 12.0
        else:
            monthly_scenario[k] = v
    # Compute monthly oferta_total and demanda_total from components
    _prod_m = monthly_scenario.get("produccion", 0)
    _imp_m = monthly_scenario.get("importaciones", 0)
    _inv_i = monthly_scenario.get("inventario_inicial", 0)
    _cna_m = monthly_scenario.get("consumo_nacional_aparente", 0)
    _exp_m = monthly_scenario.get("exportaciones_totales", 0)
    monthly_scenario["oferta_total"] = _inv_i + _prod_m + _imp_m
    monthly_scenario["demanda_total"] = _cna_m + _exp_m

    from sugar_price_model import forecast_monthly

    lpd = product_df["date"].max()
    lap = float(product_df["price"].iloc[-1])

    scenario_daily, scenario_monthly_fc = forecast_monthly(
        model, scaler, feature_cols, monthly_df,
        months_ahead=scenario_months, scenario=monthly_scenario,
        latest_price_date=lpd, latest_actual_price=lap,
        balance_df=bal_df, external_df=ext_df,
    )
    baseline_daily, baseline_monthly_fc = forecast_monthly(
        model, scaler, feature_cols, monthly_df,
        months_ahead=scenario_months, scenario=None,
        latest_price_date=lpd, latest_actual_price=lap,
        balance_df=bal_df, external_df=ext_df,
    )

    # Aggregate monthly forecasts to annual averages
    baseline_monthly_fc["year"] = baseline_monthly_fc["year"].astype(int)
    scenario_monthly_fc["year"] = scenario_monthly_fc["year"].astype(int)

    baseline_annual = baseline_monthly_fc.groupby("year").agg(
        predicted_price=("predicted_price", "mean"),
    ).reset_index()
    scenario_annual = scenario_monthly_fc.groupby("year").agg(
        predicted_price=("predicted_price", "mean"),
    ).reset_index()

    # Comparison table (annual)
    comp = baseline_annual[["year", "predicted_price"]].copy()
    comp.columns = ["Ano", "Precio Base"]
    comp["Precio Escenario"] = scenario_annual["predicted_price"].values
    comp["Diferencia"] = comp["Precio Escenario"] - comp["Precio Base"]
    comp_raw_base = comp["Precio Base"].copy()
    comp_raw_scen = comp["Precio Escenario"].copy()
    comp_raw_diff = comp["Diferencia"].copy()
    comp["Ano"] = comp["Ano"].astype(int).astype(str)
    comp["Precio Base"] = comp["Precio Base"].apply(lambda x: f"${x:,.2f}")
    comp["Precio Escenario"] = comp["Precio Escenario"].apply(lambda x: f"${x:,.2f}")
    comp["Diferencia"] = comp["Diferencia"].apply(lambda x: f"{x:+,.2f}")

    # Chart -- show daily trajectory with annual averages
    recent = product_df.tail(365)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["price"],
        mode="lines", name="Historico",
        line=dict(color=COLOR_ACTUAL, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=baseline_daily["date"], y=baseline_daily["predicted_price"],
        mode="lines", name="Pronostico Base",
        line=dict(color="#40916c", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=scenario_daily["date"], y=scenario_daily["predicted_price"],
        mode="lines", name="Pronostico Escenario",
        line=dict(color="#b8860b", width=2.5),
    ))

    # Add annual average markers
    for _, row in baseline_annual.iterrows():
        mid_date = pd.Timestamp(year=int(row["year"]), month=7, day=1)
        fig.add_trace(go.Scatter(
            x=[mid_date], y=[row["predicted_price"]],
            mode="markers", marker=dict(size=12, color="#40916c", symbol="diamond"),
            name=f"Promedio Base {int(row['year'])}", showlegend=False,
        ))
    for _, row in scenario_annual.iterrows():
        mid_date = pd.Timestamp(year=int(row["year"]), month=7, day=1)
        fig.add_trace(go.Scatter(
            x=[mid_date], y=[row["predicted_price"]],
            mode="markers", marker=dict(size=12, color="#b8860b", symbol="diamond"),
            name=f"Promedio Escenario {int(row['year'])}", showlegend=False,
        ))

    fig.update_layout(
        **AGRO_LAYOUT, height=500,
        xaxis_title="Fecha", yaxis_title="Precio (MXN/ton)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        title="Comparacion Anual: Escenario vs Proyeccion Base",
    )

    # KPIs
    current_price = product_df["price"].iloc[-1]
    baseline_end = comp_raw_base.iloc[-1]
    scenario_end = comp_raw_scen.iloc[-1]

    # Compute annual latest_balance for changes table (same aggregation as page_scenario)
    _FLOW_KEYS_LIST = list(_FLOW_KEYS)
    _annual_latest = {}
    if bal_df is not None and not bal_df.empty:
        _bdf2 = bal_df.copy()
        _bdf2["year"] = _bdf2["year"].astype(int)
        _agg2 = {c: "sum" for c in _FLOW_KEYS_LIST if c in _bdf2.columns}
        _ann2 = _bdf2.groupby("year").agg(_agg2).reset_index()
        for c in ["inventario_inicial"]:
            if c in _bdf2.columns:
                _f = _bdf2.sort_values("month_number").groupby("year")[c].first().reset_index()
                _ann2 = _ann2.merge(_f, on="year", how="left")
        for c in ["inventario_final"]:
            if c in _bdf2.columns:
                _l = _bdf2.sort_values("month_number").groupby("year")[c].last().reset_index()
                _ann2 = _ann2.merge(_l, on="year", how="left")
        # Compute oferta_total and demanda_total from components
        _ann2["oferta_total"] = _ann2.get("inventario_inicial", 0) + _ann2.get("produccion", 0) + _ann2.get("importaciones", 0)
        _ann2["demanda_total"] = _ann2.get("consumo_nacional_aparente", 0) + _ann2.get("exportaciones_totales", 0)
        if not _ann2.empty:
            _lr = _ann2.iloc[-1]
            for c in _FLOW_KEYS_LIST + ["inventario_inicial", "inventario_final", "oferta_total", "demanda_total"]:
                if c in _lr.index:
                    _annual_latest[c] = float(_lr[c])

    # Compute annual oferta_total/demanda_total for the scenario too
    _scen_prod = annual_scenario.get("produccion", 0)
    _scen_imp = annual_scenario.get("importaciones", 0)
    _scen_inv_i = annual_scenario.get("inventario_inicial", 0)
    _scen_cna = annual_scenario.get("consumo_nacional_aparente", 0)
    _scen_exp = annual_scenario.get("exportaciones_totales", 0)
    annual_scenario["oferta_total"] = _scen_inv_i + _scen_prod + _scen_imp
    annual_scenario["demanda_total"] = _scen_cna + _scen_exp

    # Changes table (annual values)
    changes = []
    for key, label in BALANCE_LABELS.items():
        old = _annual_latest.get(key, 0)
        new = annual_scenario.get(key, old)
        pct = ((new - old) / old) * 100 if old != 0 else 0
        changes.append({
            "Variable": label.split("(")[0].strip(),
            "Actual Anual": format_tons(old),
            "Escenario Anual": format_tons(new),
            "Cambio %": f"{pct:+.1f}%",
        })

    return html.Div([
        html.H2("Resultados del Escenario (Pronostico Anual)", className="section-title"),
        dash_table.DataTable(
            data=comp[["Ano", "Precio Base", "Precio Escenario", "Diferencia"]].to_dict("records"),
            columns=[{"name": c, "id": c} for c in ["Ano", "Precio Base", "Precio Escenario", "Diferencia"]],
            style_header={
                "backgroundColor": "#f8f6f2", "color": "#000",
                "fontWeight": "600", "border": "1px solid #c4beb2",
                "fontFamily": "Inter, sans-serif", "textAlign": "center",
            },
            style_cell={
                "backgroundColor": "#ffffff", "color": "#000",
                "border": "1px solid #e2ddd4", "textAlign": "center",
                "fontFamily": "Inter, sans-serif", "padding": "8px 12px",
            },
            style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
        ),
        dcc.Graph(figure=fig),
        html.Div([
            kpi_card("Precio Actual", format_price(current_price)),
            kpi_card("Promedio Anual Base", format_price(baseline_end),
                     delta=f"{baseline_end - current_price:+.2f} MXN",
                     delta_positive=baseline_end >= current_price),
            kpi_card("Promedio Anual Escenario", format_price(scenario_end),
                     delta=f"{scenario_end - current_price:+.2f} MXN",
                     delta_positive=scenario_end >= current_price),
            kpi_card("Escenario vs Base", f"{scenario_end - baseline_end:+.2f} MXN",
                     delta_positive=scenario_end >= baseline_end),
        ], className="kpi-row"),
        html.H2("Cambios del Escenario vs Balance Anual Actual", className="section-title"),
        dash_table.DataTable(
            data=changes,
            columns=[{"name": c, "id": c} for c in ["Variable", "Actual Anual", "Escenario Anual", "Cambio %"]],
            style_header={
                "backgroundColor": "#f8f6f2", "color": "#000",
                "fontWeight": "600", "border": "1px solid #c4beb2",
                "fontFamily": "Inter, sans-serif", "textAlign": "center",
            },
            style_cell={
                "backgroundColor": "#ffffff", "color": "#000",
                "border": "1px solid #e2ddd4", "textAlign": "center",
                "fontFamily": "Inter, sans-serif", "padding": "8px 12px",
            },
            style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
        ),
    ])


# ---------------------------------------------------------------
# PAGE: Feature Analysis
# ---------------------------------------------------------------
def page_features(product, product_df, model_data, has_model):
    if not has_model:
        return html.Div("Ejecute el pipeline ML primero: `py sugar_price_model.py`", className="warning-box")

    fi_df = model_data.get("feature_importance", pd.DataFrame())
    featured_df = model_data.get("featured", pd.DataFrame())
    balance_df = model_data.get("balance")

    sections = [
        html.H1(f"Factores Clave -- {PRODUCT_LABELS[product]}", className="page-title"),
        html.P(
            "Que factores del mercado tienen mayor influencia en el precio del azucar? "
            "La grafica de importancia muestra cuanto peso tiene cada variable en la prediccion. "
            "Las correlaciones indican si una variable sube o baja junto con el precio.",
            className="page-desc",
        ),
    ]

    if not fi_df.empty:
        sections.append(html.H2("Importancia de cada Factor en la Prediccion", className="section-title"))

        model_options = fi_df["model"].unique().tolist()
        sections.append(dcc.Dropdown(
            id="fi-model-select",
            options=[{"label": m, "value": m} for m in model_options],
            value=model_options[0] if model_options else None,
            clearable=False,
            style={"maxWidth": "400px", "marginBottom": "12px"},
        ))
        sections.append(html.Div(id="fi-chart-container"))
        sections.append(dcc.Store(id="fi-product-store", data=product))

    if balance_df is not None and not balance_df.empty:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Datos del Balance Azucarero (Mensual)", className="section-title"),
            dcc.Graph(figure=make_balance_chart(balance_df)),
        ])

        # Supply vs Demand
        fig_sd = go.Figure()
        period = balance_df["year"].astype(str) + "-" + balance_df["month_number"].astype(str).str.zfill(2)
        fig_sd.add_trace(go.Bar(x=period, y=balance_df["oferta_total"], name="Oferta Total", marker_color="#2d6a4f"))
        fig_sd.add_trace(go.Bar(x=period, y=balance_df["demanda_total"], name="Demanda Total", marker_color="#a0522d"))
        fig_sd.update_layout(
            **AGRO_LAYOUT, height=400, barmode="group",
            title="Oferta vs Demanda (Mensual)", xaxis_title="Mes", yaxis_title="Toneladas",
            xaxis_tickangle=-45, legend=dict(font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_sd))

    if not featured_df.empty:
        sections.append(html.H2("Relacion de cada Factor con el Precio", className="section-title"))
        numeric_cols = featured_df.select_dtypes(include=[np.number]).columns.tolist()
        if "price" in numeric_cols:
            corrs = featured_df[numeric_cols].corr()["price"].drop(
                ["price", "target"], errors="ignore"
            ).sort_values()

            agro_colorscale = [
                [0, "#a0522d"], [0.25, "#d4a76a"], [0.5, "#f1ede6"],
                [0.75, "#74c69d"], [1, "#2d6a4f"],
            ]

            fig_corr = go.Figure()
            fig_corr.add_trace(go.Bar(
                x=corrs.values, y=corrs.index, orientation="h",
                marker=dict(color=corrs.values, colorscale=agro_colorscale,
                             cmin=-1, cmax=1,
                             colorbar=dict(title="Corr")),
            ))
            fig_corr.update_layout(
                **AGRO_LAYOUT, height=max(400, len(corrs) * 18),
                title="Correlacion de Variables con Precio", xaxis_title="Correlacion de Pearson",
                legend=dict(font=LEGEND_FONT),
            )
            sections.append(dcc.Graph(figure=fig_corr))

    return html.Div(sections)


# ---------------------------------------------------------------
# EXCEL REPORT PATHS
# ---------------------------------------------------------------
EXCEL_DIR = Path("excel_reports")


# ---------------------------------------------------------------
# PAGE: Market Intelligence (CONADESUCA + Futures)
# ---------------------------------------------------------------
def page_market_intel(product, product_df, model_data, has_model):
    sections = [
        html.H1("Inteligencia de Mercado", className="page-title"),
        html.P(
            "Informacion consolidada de CONADESUCA: balance nacional de azucar, balance estimado, "
            "precios de referencia historicos, y datos de mercado externo (futuros ICE No.11 / No.16, "
            "tipo de cambio, petroleo WTI).",
            className="page-desc",
        ),
    ]

    # ==================================================================
    # 1. EXTERNAL MARKET / FUTURES (ICE No.11, No.16, WTI, FX)
    # ==================================================================
    ext_df = model_data.get("external")
    if ext_df is not None and not ext_df.empty:
        ext = ext_df.copy()
        month_col = "month" if "month" in ext.columns else "month_number"
        ext["period"] = ext["year"].astype(int).astype(str) + "-" + ext[month_col].astype(int).astype(str).str.zfill(2)

        sections.append(html.H2("Futuros de Azucar & Mercado Externo", className="section-title"))
        sections.append(html.Div(
            "Precios internacionales de futuros de azucar (ICE No.11 mundo, ICE No.16 EE.UU.), "
            "tipo de cambio USD/MXN, petroleo WTI y tipo de cambio BRL/USD. Fuente: FRED & Banxico.",
            className="info-box",
        ))

        # KPIs for latest external data
        latest_ext = ext.dropna(subset=["ice_no11"]).iloc[-1] if ext["ice_no11"].notna().any() else ext.iloc[-1]
        ext_kpis = []
        if pd.notna(latest_ext.get("ice_no11", None)):
            ext_kpis.append(kpi_card("ICE No.11 (Mundo)", f"{latest_ext['ice_no11']:.2f} c/lb"))
        if pd.notna(latest_ext.get("ice_no16", None)):
            ext_kpis.append(kpi_card("ICE No.16 (EE.UU.)", f"{latest_ext['ice_no16']:.2f} c/lb"))
        if pd.notna(latest_ext.get("usd_mxn", None)):
            ext_kpis.append(kpi_card("USD/MXN", f"{latest_ext['usd_mxn']:.2f}"))
        if pd.notna(latest_ext.get("wti", None)):
            ext_kpis.append(kpi_card("WTI Petroleo", f"${latest_ext['wti']:.2f} USD"))
        if pd.notna(latest_ext.get("brl_usd", None)):
            ext_kpis.append(kpi_card("BRL/USD", f"{latest_ext['brl_usd']:.2f}"))
        if ext_kpis:
            sections.append(html.Div(ext_kpis, className="kpi-row"))

        # ICE Futures chart
        fig_ice = go.Figure()
        ice11 = ext.dropna(subset=["ice_no11"])
        ice16 = ext.dropna(subset=["ice_no16"])
        if not ice11.empty:
            fig_ice.add_trace(go.Scatter(
                x=ice11["period"], y=ice11["ice_no11"],
                mode="lines", name="ICE No.11 (Mundo)",
                line=dict(color="#2d6a4f", width=2),
            ))
        if not ice16.empty:
            fig_ice.add_trace(go.Scatter(
                x=ice16["period"], y=ice16["ice_no16"],
                mode="lines", name="ICE No.16 (EE.UU.)",
                line=dict(color="#b8860b", width=2),
            ))
        fig_ice.update_layout(
            **AGRO_LAYOUT, height=420,
            title="Futuros de Azucar: ICE No.11 vs No.16",
            xaxis_title="Mes", yaxis_title="Centavos por Libra (USD)",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_ice))

        # FX + WTI chart
        fig_fx = go.Figure()
        fx = ext.dropna(subset=["usd_mxn"])
        if not fx.empty:
            fig_fx.add_trace(go.Scatter(
                x=fx["period"], y=fx["usd_mxn"],
                mode="lines", name="USD/MXN",
                line=dict(color="#2d6a4f", width=2),
            ))
        wti = ext.dropna(subset=["wti"])
        if not wti.empty:
            fig_fx.add_trace(go.Scatter(
                x=wti["period"], y=wti["wti"],
                mode="lines", name="WTI (USD/bbl)",
                line=dict(color="#a0522d", width=2),
                yaxis="y2",
            ))
        fig_fx.update_layout(
            **AGRO_LAYOUT, height=420,
            title="Tipo de Cambio USD/MXN & Petroleo WTI",
            xaxis_title="Mes", yaxis_title="MXN por USD",
            yaxis2=dict(title="USD/bbl", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#a0522d"), title_font=dict(color="#a0522d")),
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_fx))

    # ==================================================================
    # 2. CONADESUCA BALANCE NACIONAL (monthly from balance_monthly.csv)
    # ==================================================================
    balance_df = model_data.get("balance")
    if balance_df is not None and not balance_df.empty:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Balance Nacional de Azucar (CONADESUCA)", className="section-title"),
            html.Div(
                "Datos mensuales del balance nacional de azucar publicado por CONADESUCA. "
                "Incluye produccion, consumo, importaciones, exportaciones e inventarios.",
                className="info-box",
            ),
        ])

        bal = balance_df.copy()
        bal["period"] = bal["year"].astype(int).astype(str) + "-" + bal["month_number"].astype(int).astype(str).str.zfill(2)

        # Latest balance KPIs
        latest_row = bal.iloc[-1]
        bal_kpis = [
            kpi_card("Produccion", format_tons(latest_row.get("produccion", 0))),
            kpi_card("Consumo Nac. Aparente", format_tons(latest_row.get("consumo_nacional_aparente", 0))),
            kpi_card("Inventario Final", format_tons(latest_row.get("inventario_final", 0))),
            kpi_card("Oferta Total", format_tons(latest_row.get("oferta_total", 0))),
            kpi_card("Demanda Total", format_tons(latest_row.get("demanda_total", 0))),
        ]
        sections.append(html.Div(bal_kpis, className="kpi-row"))

        # Ratio KPIs
        ratio_kpis = []
        for key in ["supply_demand_ratio", "inventory_months", "demand_pressure", "net_exports"]:
            if key in latest_row.index and pd.notna(latest_row[key]):
                lbl = DERIVED_RATIO_LABELS.get(key, key)
                val = latest_row[key]
                fmt = format_tons(val) if key == "net_exports" else f"{val:.2f}"
                ratio_kpis.append(kpi_card(lbl, fmt))
        if ratio_kpis:
            sections.append(html.Div(ratio_kpis, className="kpi-row"))

        # Balance components chart
        sections.append(dcc.Graph(figure=make_balance_chart(bal)))

        # Supply vs Demand bar chart
        fig_sd = go.Figure()
        fig_sd.add_trace(go.Bar(x=bal["period"], y=bal["oferta_total"], name="Oferta Total", marker_color="#2d6a4f"))
        fig_sd.add_trace(go.Bar(x=bal["period"], y=bal["demanda_total"], name="Demanda Total", marker_color="#a0522d"))
        fig_sd.update_layout(
            **AGRO_LAYOUT, height=400, barmode="group",
            title="Oferta vs Demanda (Historico Mensual)",
            xaxis_title="Mes", yaxis_title="Toneladas",
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
        )
        sections.append(dcc.Graph(figure=fig_sd))

        # Inventory months chart
        if "inventory_months" in bal.columns:
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(
                x=bal["period"], y=bal["inventory_months"],
                mode="lines+markers", name="Meses de Inventario",
                line=dict(color="#b8860b", width=2),
                marker=dict(size=3),
                fill="tozeroy", fillcolor="rgba(184, 134, 11, 0.08)",
            ))
            fig_inv.update_layout(
                **AGRO_LAYOUT, height=350,
                title="Meses de Inventario (Cobertura de Consumo)",
                xaxis_title="Mes", yaxis_title="Meses",
                xaxis_tickangle=-45,
                legend=dict(font=LEGEND_FONT),
            )
            sections.append(dcc.Graph(figure=fig_inv))

        # Full balance data table (last 24 months)
        sections.append(html.H2("Detalle del Balance (Ultimos 24 Meses)", className="section-title"))
        bal_recent = bal.tail(24).copy().reset_index(drop=True)
        display_cols = [
            ("period", "Periodo"),
            ("produccion", "Produccion"),
            ("importaciones", "Importaciones"),
            ("exportaciones_totales", "Exportaciones"),
            ("consumo_nacional_aparente", "Consumo Nac."),
            ("inventario_inicial", "Inv. Inicial"),
            ("inventario_final", "Inv. Final"),
            ("oferta_total", "Oferta Total"),
            ("demanda_total", "Demanda Total"),
            ("supply_demand_ratio", "O/D Ratio"),
            ("inventory_months", "Meses Inv."),
        ]
        bal_display = pd.DataFrame()
        for col_id, col_name in display_cols:
            if col_id in bal_recent.columns:
                if col_id in ("supply_demand_ratio", "inventory_months", "demand_pressure"):
                    bal_display[col_name] = bal_recent[col_id].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                elif col_id == "period":
                    bal_display[col_name] = bal_recent[col_id].values
                else:
                    bal_display[col_name] = bal_recent[col_id].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
        sections.append(
            dash_table.DataTable(
                data=bal_display.to_dict("records"),
                columns=[{"name": c, "id": c} for c in bal_display.columns],
                style_header={
                    "backgroundColor": "#f8f6f2", "color": "#000",
                    "fontWeight": "600", "border": "1px solid #c4beb2",
                    "fontFamily": "Inter, sans-serif", "textAlign": "center",
                    "fontSize": "0.78rem",
                },
                style_cell={
                    "backgroundColor": "#ffffff", "color": "#000",
                    "border": "1px solid #e2ddd4", "textAlign": "center",
                    "fontFamily": "Inter, sans-serif", "fontSize": "0.78rem",
                    "padding": "6px 8px",
                },
                style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
                page_size=24,
            )
        )

    # ==================================================================
    # 3. CONADESUCA BALANCE NACIONAL POR CICLO (from excel_reports)
    # ==================================================================
    bal_xlsx = EXCEL_DIR / "01_balance_nacional_azucar.xlsx"
    if bal_xlsx.exists():
        try:
            df_bal_ciclo = pd.read_excel(bal_xlsx)
            # Filter to monthly rows and current/recent cycles
            df_mensual = df_bal_ciclo[df_bal_ciclo["type"] == "mensual"].copy()
            if not df_mensual.empty:
                sections.extend([
                    html.Hr(className="section-sep"),
                    html.H2("Balance por Ciclo Azucarero (CONADESUCA)", className="section-title"),
                    html.Div(
                        "Comparacion de ciclos azucareros (octubre a septiembre). "
                        "Datos directos de los reportes de politica comercial de CONADESUCA.",
                        className="info-box",
                    ),
                ])

                # Cycle-over-cycle production comparison
                cycles = df_mensual[["cycle_start", "cycle_end"]].drop_duplicates()
                recent_cycles = cycles.tail(4)

                fig_cycle = go.Figure()
                month_names = ["Oct", "Nov", "Dic", "Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep"]
                colors = ["#2d6a4f", "#b8860b", "#a0522d", "#40916c"]
                for i, (_, row) in enumerate(recent_cycles.iterrows()):
                    cycle_data = df_mensual[
                        (df_mensual["cycle_start"] == row["cycle_start"]) &
                        (df_mensual["cycle_end"] == row["cycle_end"])
                    ].sort_values("month_number")
                    label = f"{int(row['cycle_start'])}/{int(row['cycle_end'])}"
                    prod = cycle_data["produccion"].values
                    months = month_names[:len(prod)]
                    fig_cycle.add_trace(go.Scatter(
                        x=months, y=prod,
                        mode="lines+markers", name=label,
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=5),
                    ))
                fig_cycle.update_layout(
                    **AGRO_LAYOUT, height=420,
                    title="Produccion por Ciclo Azucarero (Comparativo)",
                    xaxis_title="Mes del Ciclo", yaxis_title="Toneladas",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
                )
                sections.append(dcc.Graph(figure=fig_cycle))

                # Cycle summary table: aggregate per cycle
                cycle_summary = df_mensual.groupby(["cycle_start", "cycle_end"]).agg(
                    produccion_total=("produccion", "sum"),
                    consumo_total=("consumo_nacional_aparente", "sum"),
                    export_total=("exportaciones_totales", "sum"),
                    import_total=("importaciones", "sum"),
                ).reset_index()
                cycle_summary["ciclo"] = cycle_summary["cycle_start"].astype(int).astype(str) + "/" + cycle_summary["cycle_end"].astype(int).astype(str)
                cycle_summary = cycle_summary.sort_values("cycle_start")

                cs_display = pd.DataFrame()
                cs_display["Ciclo"] = cycle_summary["ciclo"]
                cs_display["Produccion"] = cycle_summary["produccion_total"].apply(lambda x: f"{x:,.0f}")
                cs_display["Consumo"] = cycle_summary["consumo_total"].apply(lambda x: f"{x:,.0f}")
                cs_display["Exportaciones"] = cycle_summary["export_total"].apply(lambda x: f"{x:,.0f}")
                cs_display["Importaciones"] = cycle_summary["import_total"].apply(lambda x: f"{x:,.0f}")

                sections.append(
                    dash_table.DataTable(
                        data=cs_display.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in cs_display.columns],
                        style_header={
                            "backgroundColor": "#f8f6f2", "color": "#000",
                            "fontWeight": "600", "border": "1px solid #c4beb2",
                            "fontFamily": "Inter, sans-serif", "textAlign": "center",
                        },
                        style_cell={
                            "backgroundColor": "#ffffff", "color": "#000",
                            "border": "1px solid #e2ddd4", "textAlign": "center",
                            "fontFamily": "Inter, sans-serif", "fontSize": "0.82rem",
                            "padding": "8px 12px",
                        },
                        style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
                    )
                )
        except Exception:
            pass

    # ==================================================================
    # 4. BALANCE ESTIMADO (CONADESUCA)
    # ==================================================================
    bal_est_xlsx = EXCEL_DIR / "03_balance_estimado.xlsx"
    if bal_est_xlsx.exists():
        try:
            df_est = pd.read_excel(bal_est_xlsx)
            if not df_est.empty and "oferta_total" in df_est.columns:
                sections.extend([
                    html.Hr(className="section-sep"),
                    html.H2("Balance Estimado (CONADESUCA)", className="section-title"),
                    html.Div(
                        "Estimaciones oficiales de CONADESUCA del balance azucarero para ciclos actuales y futuros.",
                        className="info-box",
                    ),
                ])

                est_display = pd.DataFrame()
                est_display["Ciclo"] = df_est["cycle_start"].astype(int).astype(str) + "/" + df_est["cycle_end"].astype(int).astype(str)
                for col, lbl in [("oferta_total", "Oferta Total"), ("produccion", "Produccion"),
                                  ("importaciones", "Importaciones"), ("demanda_total", "Demanda Total"),
                                  ("exportaciones", "Exportaciones"), ("consumo_nacional_aparente", "Consumo Nac."),
                                  ("inventario_optimo", "Inv. Optimo")]:
                    if col in df_est.columns:
                        est_display[lbl] = df_est[col].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) and x != 0 else "-"
                        )

                if len(est_display.columns) > 1:
                    sections.append(
                        dash_table.DataTable(
                            data=est_display.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in est_display.columns],
                            style_header={
                                "backgroundColor": "#f8f6f2", "color": "#000",
                                "fontWeight": "600", "border": "1px solid #c4beb2",
                                "fontFamily": "Inter, sans-serif", "textAlign": "center",
                            },
                            style_cell={
                                "backgroundColor": "#ffffff", "color": "#000",
                                "border": "1px solid #e2ddd4", "textAlign": "center",
                                "fontFamily": "Inter, sans-serif", "fontSize": "0.82rem",
                                "padding": "8px 12px",
                            },
                            style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
                        )
                    )
        except Exception:
            pass

    # ==================================================================
    # 5. HISTORICO PRECIO DE REFERENCIA (CONADESUCA)
    # ==================================================================
    hist_xlsx = EXCEL_DIR / "09_historico_precio_referencia.xlsx"
    if hist_xlsx.exists():
        try:
            df_hist = pd.read_excel(hist_xlsx)
            if not df_hist.empty and "cycle" in df_hist.columns:
                sections.extend([
                    html.Hr(className="section-sep"),
                    html.H2("Historico de Precio de Referencia (CONADESUCA)", className="section-title"),
                    html.Div(
                        "Precio de referencia del azucar base estandar, produccion, cana industrializada, "
                        "y KARBE nacional por ciclo azucarero, segun reportes de CONADESUCA.",
                        className="info-box",
                    ),
                ])

                # Price reference chart
                price_col = [c for c in df_hist.columns if "referencia" in c.lower()]
                mayoreo_col = [c for c in df_hist.columns if "mayoreo" in c.lower()]
                prod_col = [c for c in df_hist.columns if "producida" in c.lower()]

                if price_col:
                    fig_ref = go.Figure()
                    fig_ref.add_trace(go.Bar(
                        x=df_hist["cycle"], y=df_hist[price_col[0]],
                        name="Precio Referencia Estandar",
                        marker_color="#2d6a4f",
                    ))
                    if mayoreo_col:
                        fig_ref.add_trace(go.Bar(
                            x=df_hist["cycle"], y=df_hist[mayoreo_col[0]],
                            name="Precio Mayoreo 23 Mercados",
                            marker_color="#b8860b",
                        ))
                    fig_ref.update_layout(
                        **AGRO_LAYOUT, height=400, barmode="group",
                        title="Precio de Referencia por Ciclo Azucarero",
                        xaxis_title="Ciclo", yaxis_title="MXN / Tonelada",
                        xaxis_tickangle=-45,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT),
                    )
                    sections.append(dcc.Graph(figure=fig_ref))

                if prod_col:
                    fig_prod = go.Figure()
                    fig_prod.add_trace(go.Bar(
                        x=df_hist["cycle"], y=df_hist[prod_col[0]],
                        name="Azucar Producida (ton)",
                        marker_color="#40916c",
                    ))
                    fig_prod.update_layout(
                        **AGRO_LAYOUT, height=380,
                        title="Produccion de Azucar por Ciclo",
                        xaxis_title="Ciclo", yaxis_title="Toneladas",
                        xaxis_tickangle=-45,
                        legend=dict(font=LEGEND_FONT),
                    )
                    sections.append(dcc.Graph(figure=fig_prod))

                # Full table
                hist_display = df_hist.copy()
                for c in hist_display.columns:
                    if hist_display[c].dtype in [np.float64, np.int64]:
                        hist_display[c] = hist_display[c].apply(
                            lambda x: f"{x:,.2f}" if pd.notna(x) else "-"
                        )
                sections.append(
                    dash_table.DataTable(
                        data=hist_display.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in hist_display.columns],
                        style_header={
                            "backgroundColor": "#f8f6f2", "color": "#000",
                            "fontWeight": "600", "border": "1px solid #c4beb2",
                            "fontFamily": "Inter, sans-serif", "textAlign": "center",
                            "fontSize": "0.75rem",
                        },
                        style_cell={
                            "backgroundColor": "#ffffff", "color": "#000",
                            "border": "1px solid #e2ddd4", "textAlign": "center",
                            "fontFamily": "Inter, sans-serif", "fontSize": "0.75rem",
                            "padding": "6px 8px",
                        },
                        style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px",
                                      "overflow": "hidden", "overflowX": "auto"},
                        page_size=15,
                    )
                )
        except Exception:
            pass

    # ==================================================================
    # 6. EXTERNAL DATA TABLE (full monthly series)
    # ==================================================================
    if ext_df is not None and not ext_df.empty:
        sections.extend([
            html.Hr(className="section-sep"),
            html.H2("Datos de Mercado Externo (Serie Completa)", className="section-title"),
        ])
        ext_table = ext_df.copy()
        month_col = "month" if "month" in ext_table.columns else "month_number"
        ext_table["Periodo"] = ext_table["year"].astype(int).astype(str) + "-" + ext_table[month_col].astype(int).astype(str).str.zfill(2)
        for c in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"]:
            if c in ext_table.columns:
                ext_table[c] = ext_table[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        ext_show = ext_table[["Periodo"] + [c for c in ["usd_mxn", "ice_no11", "ice_no16", "wti", "brl_usd"] if c in ext_table.columns]]
        col_labels = {"Periodo": "Periodo", "usd_mxn": "USD/MXN", "ice_no11": "ICE No.11", "ice_no16": "ICE No.16", "wti": "WTI", "brl_usd": "BRL/USD"}
        sections.append(
            dash_table.DataTable(
                data=ext_show.to_dict("records"),
                columns=[{"name": col_labels.get(c, c), "id": c} for c in ext_show.columns],
                style_header={
                    "backgroundColor": "#f8f6f2", "color": "#000",
                    "fontWeight": "600", "border": "1px solid #c4beb2",
                    "fontFamily": "Inter, sans-serif", "textAlign": "center",
                },
                style_cell={
                    "backgroundColor": "#ffffff", "color": "#000",
                    "border": "1px solid #e2ddd4", "textAlign": "center",
                    "fontFamily": "Inter, sans-serif", "fontSize": "0.82rem",
                    "padding": "6px 10px",
                },
                style_table={"border": "1px solid #2d6a4f", "borderRadius": "8px", "overflow": "hidden"},
                page_size=18,
                sort_action="native",
            )
        )

    if len(sections) <= 2:
        sections.append(html.Div(
            "No se encontraron datos de mercado. Ejecute los scrapers y el pipeline ML primero.",
            className="warning-box",
        ))

    return html.Div(sections)


# Callback: update ML predictions chart when model selection changes
@callback(
    Output("ml-predictions-chart-container", "children"),
    Input("ml-model-select", "value"),
    State("ml-product-store", "data"),
    prevent_initial_call=False,
)
def update_ml_predictions_chart(selected_models, product):
    model_data = load_model_outputs(product)
    preds_df = model_data.get("predictions", pd.DataFrame())
    if preds_df.empty or not selected_models:
        return html.Div("Seleccione al menos un modelo.", className="warning-box")

    actual_col = "actual_target" if "actual_target" in preds_df.columns else "actual"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds_df["date"], y=preds_df[actual_col],
        mode="lines", name="Real",
        line=dict(color=COLOR_ACTUAL, width=2),
    ))
    for i, col in enumerate(selected_models):
        if col in preds_df.columns:
            fig.add_trace(go.Scatter(
                x=preds_df["date"], y=preds_df[col],
                mode="lines", name=col.replace("_", " ").title(),
                line=dict(color=AGRO_PALETTE[i % len(AGRO_PALETTE)], width=1.5, dash="dot"),
            ))
    fig.update_layout(**AGRO_LAYOUT, height=500,
                      xaxis_title="Fecha", yaxis_title="Precio (MXN/ton)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)", font=LEGEND_FONT))
    return dcc.Graph(figure=fig)


# Callback: update feature importance chart when model selection changes
@callback(
    Output("fi-chart-container", "children"),
    Input("fi-model-select", "value"),
    State("fi-product-store", "data"),
    prevent_initial_call=False,
)
def update_fi_chart(selected_model, product):
    model_data = load_model_outputs(product)
    fi_df = model_data.get("feature_importance", pd.DataFrame())
    if fi_df.empty or not selected_model:
        return html.Div("No feature importance data.", className="warning-box")

    fi_model = fi_df[fi_df["model"] == selected_model].sort_values("importance", ascending=True).tail(20)
    bal_feat_names = list(BALANCE_LABELS.keys()) + [
        "net_exports", "supply_demand_ratio", "inventory_to_consumption", "production_share",
    ]
    fi_model = fi_model.copy()
    fi_model["category"] = fi_model["feature"].apply(
        lambda x: "Balance/Fundamental" if x in bal_feat_names else "Tecnico/Precio"
    )

    fig_fi = px.bar(
        fi_model, x="importance", y="feature", orientation="h",
        color="category", height=550,
        title=f"Top 20 Variables -- {selected_model}",
        color_discrete_map={"Balance/Fundamental": "#b8860b", "Tecnico/Precio": "#2d6a4f"},
    )
    fig_fi.update_layout(**AGRO_LAYOUT, yaxis_title="", xaxis_title="Importancia", legend=dict(font=LEGEND_FONT))
    return dcc.Graph(figure=fig_fi)


# ---------------------------------------------------------------
# Excel Download callback
# ---------------------------------------------------------------
@callback(
    Output("download-excel", "data"),
    Input("btn-download-excel", "n_clicks"),
    State("product-select", "value"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, product):
    if not n_clicks:
        return no_update

    raw_df = load_raw_prices()
    product_df = raw_df[raw_df["product_type"] == product].copy()
    model_data = load_model_outputs(product)
    label = PRODUCT_LABELS.get(product, product)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Daily prices
        if not product_df.empty:
            export_prices = product_df[["date", "price", "product_type"]].copy()
            export_prices["date"] = export_prices["date"].dt.strftime("%Y-%m-%d")
            export_prices.to_excel(writer, sheet_name="Precios Diarios", index=False)

        # Sheet 2: Forecast
        if "forecast" in model_data:
            fc = model_data["forecast"].copy()
            if "date" in fc.columns:
                fc["date"] = pd.to_datetime(fc["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            fc.to_excel(writer, sheet_name="Pronostico", index=False)

        # Sheet 3: Monthly forecast
        if "monthly_forecast" in model_data:
            mf = model_data["monthly_forecast"].copy()
            mf.to_excel(writer, sheet_name="Pronostico Mensual", index=False)

        # Sheet 4: Model predictions (actual vs predicted)
        if "predictions" in model_data:
            pred = model_data["predictions"].copy()
            if "date" in pred.columns:
                pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            pred.to_excel(writer, sheet_name="Predicciones Modelo", index=False)

        # Sheet 5: Model metrics
        if "metrics" in model_data:
            model_data["metrics"].to_excel(writer, sheet_name="Metricas Modelo", index=False)

        # Sheet 6: Feature importance
        if "feature_importance" in model_data:
            model_data["feature_importance"].to_excel(writer, sheet_name="Importancia Variables", index=False)

        # Sheet 7: Balance data
        if "balance" in model_data:
            model_data["balance"].to_excel(writer, sheet_name="Balance Azucarero", index=False)

        # Sheet 8: External market data
        if "external" in model_data:
            model_data["external"].to_excel(writer, sheet_name="Datos Mercado Externo", index=False)

    buf.seek(0)
    return dcc.send_bytes(buf.getvalue(), f"sugar_focars_{product}.xlsx")


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
