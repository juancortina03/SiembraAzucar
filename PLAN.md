# Migrate Dashboard from Streamlit to Dash

## Approach
Rewrite `dashboard.py` as a Dash app preserving all 6 pages, styling, charts, and functionality.

## Steps
1. Create `dashboard.py` with Dash framework (replacing Streamlit)
2. Keep all existing Plotly charts (they work directly in Dash)
3. Multi-page layout via `dcc.Location` + callbacks
4. Replicate the agro theme via CSS stylesheet
5. Port all 6 pages: Overview, Price History, ML Comparison, Forecast, Scenario Predictor, Feature Analysis
6. Keep same data loading logic
7. Add `dash` to requirements.txt
