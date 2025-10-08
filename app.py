import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from services.news_service import fetch_news_finnhub, filter_news_by_country

# --- Configuración base ---
st.set_page_config(page_title="Dashboard Macro por País", layout="wide")

# Rutas por defecto
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "macro_data.db"
DEFAULT_PARQUET_PATH = DATA_DIR / "macro_data.parquet"

MULTIPLES_SET = {
    'P/E Ratio', 'P/B Ratio', 'EV/Sales', 'P/Sales', 'EV/EBITDA', 'ROE', 'Index Price', 'Price'
}


# =========================
# Utilidades de Base de Datos / Archivos
# =========================

def get_connection(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def detect_data_source() -> tuple[str, Path | None]:
    """Devuelve (kind, path) donde kind in {sqlite, parquet, none}."""
    if DEFAULT_DB_PATH.exists():
        return ("sqlite", DEFAULT_DB_PATH)
    if DEFAULT_PARQUET_PATH.exists():
        return ("parquet", DEFAULT_PARQUET_PATH)
    return ("none", None)


def get_countries_sqlite(conn: sqlite3.Connection) -> List[str]:
    q = "SELECT DISTINCT country FROM macro_data ORDER BY 1"
    return [r[0] for r in conn.execute(q).fetchall()]


def get_metrics_sqlite(conn: sqlite3.Connection, country: str | None = None) -> List[str]:
    if country:
        q = "SELECT DISTINCT metric FROM macro_data WHERE country = ? ORDER BY 1"
        rows = conn.execute(q, (country,)).fetchall()
    else:
        q = "SELECT DISTINCT metric FROM macro_data ORDER BY 1"
        rows = conn.execute(q).fetchall()
    return [r[0] for r in rows]


def get_date_bounds_sqlite(conn: sqlite3.Connection, country: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    q = "SELECT MIN(DATE), MAX(DATE) FROM macro_data WHERE country = ?"
    mn, mx = conn.execute(q, (country,)).fetchone()
    mn_dt = pd.to_datetime(mn)
    mx_dt = pd.to_datetime(mx)
    return mn_dt, mx_dt


def load_macro_data_sqlite(conn: sqlite3.Connection, country: str, metrics: List[str], d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    country_str = str(country)
    metrics_str = [str(m) for m in list(metrics)]
    d1_str = pd.to_datetime(d1).strftime('%Y-%m-%d')
    d2_str = pd.to_datetime(d2).strftime('%Y-%m-%d')

    placeholders = ",".join(["?"] * len(metrics_str))
    q = f"""
        SELECT DATE, metric, value
        FROM macro_data
        WHERE country = ?
          AND DATE >= ? AND DATE <= ?
          AND metric IN ({placeholders})
        ORDER BY DATE
    """
    params = [country_str, d1_str, d2_str, *metrics_str]
    df = pd.read_sql(q, conn, params=params)
    # Deduplicate any duplicated columns defensively (e.g., 'DATE')
    if hasattr(df, 'columns') and getattr(df.columns, 'duplicated', None) is not None:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    if df.empty:
        return df
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    if df['value'].dtype == 'object':
        v0 = pd.to_numeric(df['value'], errors='coerce')
        if v0.notna().sum() < max(1, int(0.5 * len(v0))):
            v1 = pd.to_numeric(df['value'].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
            df['value'] = v1
        else:
            df['value'] = v0
    else:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.sort_values(['metric', 'DATE'])
    df['value'] = df.groupby('metric')['value'].ffill()
    df = df.dropna(subset=['DATE', 'value'])
    df = df.drop_duplicates()
    return df


def load_macro_data_parquet(path: Path, country: str, metrics: List[str], d1: pd.Timestamp, d2: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Deduplicate any duplicated columns defensively (e.g., 'DATE')
    if hasattr(df, 'columns') and getattr(df.columns, 'duplicated', None) is not None:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    if df.empty:
        return df
    # Expect columns: country, DATE, metric, value
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df[(df['country'] == country) & (df['DATE'] >= d1) & (df['DATE'] <= d2) & (df['metric'].isin(metrics))]
    if df.empty:
        return df
    df = df[['DATE', 'metric', 'value']].copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.sort_values(['metric', 'DATE'])
    df['value'] = df.groupby('metric')['value'].ffill()
    df = df.dropna(subset=['DATE', 'value']).drop_duplicates()
    return df


# =========================
# Cálculos
# =========================

def compute_yoy(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return df_long
    tmp = df_long.copy().dropna(subset=['DATE'])
    tmp['DATE_M'] = pd.to_datetime(tmp['DATE'], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    rows = []
    for m in tmp['metric'].unique():
        t = tmp.loc[tmp['metric'] == m, ['DATE_M', 'metric', 'value']].dropna(subset=['value']).sort_values('DATE_M')
        if t.empty:
            continue
        t = t.drop_duplicates(subset=['DATE_M'], keep='last')
        idx = pd.period_range(t['DATE_M'].min().to_period('M'), t['DATE_M'].max().to_period('M'), freq='M').to_timestamp('M')
        t = t.set_index('DATE_M').reindex(idx)
        t['metric'] = m
        t['value'] = pd.to_numeric(t['value'], errors='coerce').ffill()
        t = t.reset_index().rename(columns={'index': 'DATE'})
        rows.append(t[['DATE', 'metric', 'value']])
    if not rows:
        return pd.DataFrame(columns=['DATE', 'metric', 'value', 'yoy'])
    work = pd.concat(rows, ignore_index=True)
    work = work.sort_values(['metric', 'DATE'])
    work['yoy'] = work.groupby('metric')['value'].pct_change(12)
    return work.dropna(subset=['yoy'])


def latest_snapshot(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    latest_dates = df_long.groupby('metric')['DATE'].max().reset_index()
    merged = df_long.merge(latest_dates, on=['metric', 'DATE'], how='inner')
    snap_rows = []
    for m in merged['metric'].unique():
        ser = df_long[df_long['metric'] == m].sort_values('DATE').set_index('DATE')['value']
        last_date = ser.index.max()
        last_val = ser.loc[last_date]
        chg_1m = np.nan
        chg_12m = np.nan
        one_m_ago = last_date - pd.DateOffset(months=1)
        twelve_m_ago = last_date - pd.DateOffset(months=12)
        ser_month = ser[ser.index <= last_date]
        try:
            v1 = ser_month.loc[:one_m_ago].iloc[-1]
            chg_1m = (last_val - v1) / abs(v1) if v1 != 0 else np.nan
        except Exception:
            pass
        try:
            v12 = ser_month.loc[:twelve_m_ago].iloc[-1]
            chg_12m = (last_val - v12) / abs(v12) if v12 != 0 else np.nan
        except Exception:
            pass
        snap_rows.append({'metric': m, 'last_date': last_date, 'last_value': last_val, 'chg_1m': chg_1m, 'chg_12m': chg_12m})
    return pd.DataFrame(snap_rows)


def corr_heatmap(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    wide = df_long.pivot_table(index='DATE', columns='metric', values='value')
    if wide.shape[1] < 2:
        return pd.DataFrame()
    return wide.corr(min_periods=12)


def compute_zscores_table(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame()
    tmp = df_long.copy().dropna(subset=['DATE'])
    tmp['DATE_M'] = pd.to_datetime(tmp['DATE'], errors='coerce').dt.to_period('M').dt.to_timestamp('M')
    rows = []
    for m in tmp['metric'].unique():
        t = tmp.loc[tmp['metric'] == m, ['DATE_M', 'metric', 'value']].dropna(subset=['value']).sort_values('DATE_M')
        if t.empty:
            continue
        t = t.drop_duplicates(subset=['DATE_M'], keep='last')
        idx = pd.period_range(t['DATE_M'].min().to_period('M'), t['DATE_M'].max().to_period('M'), freq='M').to_timestamp('M')
        t = t.set_index('DATE_M').reindex(idx)
        t['metric'] = m
        t['value'] = pd.to_numeric(t['value'], errors='coerce').ffill()
        t = t.reset_index().rename(columns={'index': 'DATE'})
        ser = t.set_index('DATE')['value'].dropna()
        if ser.empty:
            continue
        last_date = ser.index.max()
        last_val = ser.iloc[-1]
        def z_of(months: int | None):
            s = ser if months is None else ser.tail(months)
            if len(s) < 6:
                return np.nan
            mu = s.mean()
            sd = s.std(ddof=1)
            return (last_val - mu) / sd if sd and not np.isclose(sd, 0) else np.nan
        rows.append({
            'metric': m,
            'last_date': last_date,
            'last_value': last_val,
            'z_1y': z_of(12),
            'z_3y': z_of(36),
            'z_5y': z_of(60),
            'z_hist': z_of(None),
        })
    dfz = pd.DataFrame(rows)
    if not dfz.empty:
        dfz = dfz.sort_values('metric')
    return dfz


def load_metric_series_sqlite(conn: sqlite3.Connection, country: str, metric: str) -> pd.DataFrame:
    q = """
        SELECT DATE, value
        FROM macro_data
        WHERE country = ? AND metric = ?
        ORDER BY DATE
    """
    df = pd.read_sql(q, conn, params=[country, metric])
    if df.empty:
        return df
    # Deduplicate potential duplicated columns
    if hasattr(df, 'columns') and getattr(df.columns, 'duplicated', None) is not None:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['DATE', 'value']).drop_duplicates().sort_values('DATE')
    return df[['DATE', 'value']]


def compute_index_summary(df_idx: pd.DataFrame) -> dict:
    if df_idx.empty:
        return {}
    s = df_idx.sort_values('DATE').set_index('DATE')['value'].dropna()
    if s.empty:
        return {}
    last_date = s.index.max()
    last_value = float(s.loc[last_date])
    # YTD using last value of previous year end as base
    year_start = pd.Timestamp(year=last_date.year, month=1, day=1)
    try:
        base_ytd = s.loc[:(year_start - pd.Timedelta(days=1))].iloc[-1]
        ytd = (last_value - base_ytd) / abs(base_ytd)
    except Exception:
        ytd = np.nan
    def cal_ret(year: int):
        try:
            y_start = pd.Timestamp(year=year, month=1, day=1)
            y_end = pd.Timestamp(year=year, month=12, day=31)
            base = s.loc[:(y_start - pd.Timedelta(days=1))].iloc[-1]
            endv = s.loc[:y_end].iloc[-1]
            return (float(endv) - float(base)) / abs(float(base))
        except Exception:
            return np.nan
    # Volatility annualized using monthly returns over last 36 months
    try:
        sm = s.resample('M').last().dropna()
        rets = sm.pct_change().dropna()
        if len(rets) >= 6:
            rets = rets.tail(36)
            vol_ann = float(rets.std(ddof=1) * np.sqrt(12))
        else:
            vol_ann = np.nan
    except Exception:
        vol_ann = np.nan
    return {
        'last_value': last_value,
        'ytd': ytd,
        'ret_2024': cal_ret(2024),
        'ret_2023': cal_ret(2023),
        'vol_ann': vol_ann,
    }


# =========================
# Noticias + Perplexity (opcional)
# =========================

def filter_recent_unique_news(items: List[dict], days: int = 14, max_total: int = 20) -> List[dict]:
    if not items:
        return []
    cutoff = datetime.utcnow() - timedelta(days=days)
    def parse_dt(x):
        try:
            if isinstance(x, (int, float)):
                return datetime.utcfromtimestamp(int(x))
            return pd.to_datetime(x, errors='coerce').to_pydatetime()
        except Exception:
            return None
    filtered = []
    for it in items:
        dt_raw = it.get('datetime')
        dt_parsed = parse_dt(dt_raw)
        if dt_parsed and dt_parsed >= cutoff:
            filtered.append(it)
    seen = set()
    out = []
    for it in sorted(filtered, key=lambda r: r.get('datetime') or 0, reverse=True):
        title = (it.get('headline') or it.get('title') or '').strip().casefold()
        if not title or title in seen:
            continue
        seen.add(title)
        out.append(it)
        if len(out) >= max_total:
            break
    return out


def perplexity_country_brief(api_key: str, country: str, news: List[dict]) -> str:
    if not api_key:
        return "[Configura PERPLEXITY_API_KEY en .streamlit/secrets.toml o variable de entorno]"
    bullets = []
    for it in news[:12]:
        title = (it.get('headline') or it.get('title') or '').strip()
        source = (it.get('source') or '').strip()
        url = (it.get('url') or '').strip()
        dt_raw = it.get('datetime')
        try:
            if isinstance(dt_raw, (int, float)):
                dt_txt = datetime.utcfromtimestamp(int(dt_raw)).strftime('%Y-%m-%d')
            else:
                dt_txt = str(pd.to_datetime(dt_raw, errors='coerce').date())
        except Exception:
            dt_txt = ''
        if title:
            bullets.append(f"- [{dt_txt}] {title} ({source}) {url}")
    news_block = "\n".join(bullets) if bullets else "- (sin titulares previos)"
    endpoint = os.environ.get('PPLX_ENDPOINT', 'https://api.perplexity.ai/chat/completions')
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        models_try = [
            "sonar-pro",
            "sonar",
            "sonar-reasoning",
        ]
        last_err = None
        for mdl in models_try:
            payload = {
                "model": mdl,
                "messages": [
                    {"role": "system", "content": "Eres un analista macro y de mercados. Resume impacto en FX, bonos y equity por país. IMPORTANTE: Siempre responde en español, sin importar el idioma de las noticias fuente."},
                    {"role": "user", "content": f"País: {country}\nNoticias recientes:\n{news_block}\n\nElabora un brief de 6-10 viñetas con insights accionables. Responde SIEMPRE en español."}
                ],
                "temperature": 0.4,
            }
            r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            if r.status_code >= 400:
                # Capture server-provided error message for diagnostics
                try:
                    err_txt = r.text.strip()
                except Exception:
                    err_txt = str(r.status_code)
                last_err = f"{r.status_code} {err_txt} (model={mdl})"
                continue
            try:
                data = r.json()
                content = (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                if content:
                    return content
            except Exception as je:
                last_err = f"Invalid JSON response: {je}"
                continue
        return f"[Perplexity error: {last_err or 'Unknown error'}]"
    except Exception as e:
        return f"[Perplexity error: {e}]"


# =========================
# UI
# =========================

def main():
    st.title("Macro por País")
    st.caption("Fuente: base local macro_data.db. Noticias: Finnhub + Perplexity.")

    kind, path = detect_data_source()
    if kind == "none":
        st.warning("No se encontró `data/macro_data.db` ni `data/macro_data.parquet`. Usa scripts/extract_data.py para traer un subset.")
        st.stop()

    # Cargar listas
    if kind == "sqlite":
        with get_connection(path) as conn:
            countries = get_countries_sqlite(conn)
    else:
        # parquet: leer países desde archivo
        df_all = pd.read_parquet(path, columns=["country"]).dropna()
        countries = sorted(df_all['country'].unique().tolist())

    if not countries:
        st.error("No hay países en la fuente de datos.")
        st.stop()

    # Sidebar
    with st.sidebar:
        country = st.selectbox("País", countries, index=0)
        # Métricas por país
        if kind == "sqlite":
            with get_connection(path) as conn:
                metrics_all = get_metrics_sqlite(conn, country)
                d1_default, d2_default = get_date_bounds_sqlite(conn, country)
        else:
            dfc = pd.read_parquet(path, columns=["country", "metric", "DATE"]).dropna()
            metrics_all = sorted(dfc[dfc['country'] == country]['metric'].unique().tolist())
            sub = dfc[dfc['country'] == country]
            d1_default, d2_default = pd.to_datetime(sub['DATE']).min(), pd.to_datetime(sub['DATE']).max()

        metrics = st.multiselect("Métricas", options=metrics_all, default=metrics_all[: min(6, len(metrics_all))])
        d1 = st.date_input("Desde", value=d1_default.date() if pd.notna(d1_default) else None)
        d2 = st.date_input("Hasta", value=d2_default.date() if pd.notna(d2_default) else None)
        max_charts = st.slider("Cantidad máxima de gráficos", 1, 12, 6)
        st.markdown("---")
        st.subheader("Noticias e IA")
        days_back = st.slider("Días a consultar (Finnhub)", min_value=3, max_value=45, value=14)
        show_news = st.checkbox("Mostrar titulares Finnhub", value=True)
        show_ai = st.checkbox("Mostrar briefing IA", value=True)

    d1_ts = pd.to_datetime(d1) if d1 else d1_default
    d2_ts = pd.to_datetime(d2) if d2 else d2_default

    # Carga de datos según backend
    if kind == "sqlite":
        with get_connection(path) as conn:
            df_long = load_macro_data_sqlite(conn, country, metrics, d1_ts, d2_ts)
    else:
        df_long = load_macro_data_parquet(path, country, metrics, d1_ts, d2_ts)

    # Mostrar todo en un flujo continuo, sin selector de modo
    if df_long.empty:
        st.warning("No hay datos para la selección actual.")
    else:
        # Series individuales por variable (compactas) + z-scores al lado
        st.subheader(f"Series por variable — {country}")
        zs_cards = compute_zscores_table(df_long)
        for metric in metrics[:max_charts]:
            sub = df_long[df_long['metric'] == metric].sort_values('DATE')
            if sub.empty:
                continue
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = px.line(sub, x='DATE', y='value', title=f"{metric}")
                fig.update_layout(yaxis_title='Valor', xaxis_title='Fecha', height=420)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                row = None
                if not zs_cards.empty:
                    r = zs_cards[zs_cards['metric'] == metric]
                    row = r.iloc[0] if not r.empty else None
                st.markdown("**Z-scores**")
                if row is not None:
                    st.markdown(
                        f"- 1 año: {row['z_1y']:.2f}\n\n"
                        f"- 3 años: {row['z_3y']:.2f}\n\n"
                        f"- 5 años: {row['z_5y']:.2f}\n\n"
                        f"- Histórico: {row['z_hist']:.2f}"
                    )
                else:
                    st.caption("Sin datos suficientes para z-scores")

        # Correlaciones
        st.subheader("Correlación entre variables (período seleccionado)")
        corr = corr_heatmap(df_long)
        if not corr.empty:
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale='RdBu', zmin=-1, zmax=1, colorbar=dict(title='corr')
            ))
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.caption("Selecciona al menos 2 variables para ver la correlación.")

        # Snapshot últimos valores
        st.subheader("Snapshot último dato")
        snap = latest_snapshot(df_long)
        if not snap.empty:
            snap_disp = snap.copy()
            snap_disp['last_date'] = snap_disp['last_date'].dt.strftime('%Y-%m-%d')
            for c in ['chg_1m', 'chg_12m']:
                if c in snap_disp.columns:
                    snap_disp[c] = (snap_disp[c] * 100).round(2)
            st.dataframe(snap_disp.rename(columns={'last_date': 'fecha', 'last_value': 'valor', 'chg_1m': '% 1m', 'chg_12m': '% 12m'}), use_container_width=True)

        # Z-scores por ventanas
        st.subheader("Desviaciones estándar (z-score) por ventana")
        zs = compute_zscores_table(df_long)
        if not zs.empty:
            zs_disp = zs.copy()
            zs_disp['last_date'] = zs_disp['last_date'].dt.strftime('%Y-%m-%d')
            for c in ['z_1y', 'z_3y', 'z_5y', 'z_hist']:
                if c in zs_disp.columns:
                    zs_disp[c] = zs_disp[c].round(2)
            st.dataframe(zs_disp.rename(columns={'last_date': 'fecha', 'last_value': 'valor'}), use_container_width=True)

        # Múltiplos
        sel_is_multiples = [m for m in metrics if m in MULTIPLES_SET]
        if sel_is_multiples:
            st.subheader("Múltiplos de mercado")
            df_mult = df_long[df_long['metric'].isin(MULTIPLES_SET)].copy()
            if not df_mult.empty:
                fig_mult = px.line(df_mult, x='DATE', y='value', color='metric', title=f"Múltiplos seleccionados ({country})")
                fig_mult.update_layout(yaxis_title='Valor', xaxis_title='Fecha')
                st.plotly_chart(fig_mult, use_container_width=True)

        # Noticias + IA
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader(f"Noticias recientes - {country}")
            news = []
            if show_news:
                finnhub_key = st.secrets.get('FINNHUB_API_KEY', os.getenv('FINNHUB_API_KEY', ''))
                if finnhub_key:
                    end_dt = pd.Timestamp.today().normalize()
                    start_dt = end_dt - pd.Timedelta(days=int(days_back))
                    items = fetch_news_finnhub(country, api_key=finnhub_key, max_items=100)
                    items = filter_news_by_country(items, country)
                    news_recent = []
                    for it in items:
                        ts = it.get('datetime')
                        try:
                            ts_parsed = pd.to_datetime(ts, unit='s', errors='coerce') if isinstance(ts, (int, float)) else pd.to_datetime(ts, errors='coerce')
                        except Exception:
                            ts_parsed = pd.NaT
                        if pd.notna(ts_parsed) and (start_dt <= ts_parsed <= end_dt):
                            news_recent.append(it)
                    news = filter_recent_unique_news(news_recent, days=days_back, max_total=20)
                    st.caption(f"Ventana noticias: {start_dt.date()} → {end_dt.date()} ({len(news)} items)")
                    if not news:
                        st.write("No se encontraron noticias relevantes en el período seleccionado.")
                    else:
                        for it in news:
                            dt_txt = it.get('datetime')
                            try:
                                if isinstance(dt_txt, (int, float)):
                                    dt_fmt = datetime.utcfromtimestamp(int(dt_txt)).strftime('%Y-%m-%d')
                                else:
                                    dt_fmt = str(pd.to_datetime(dt_txt, errors='coerce').date())
                            except Exception:
                                dt_fmt = ""
                            st.markdown(f"- [{it.get('headline','')}]({it.get('url','')}) — {it.get('source','')} {dt_fmt}")
                            # Pequeño resumen debajo de cada titular si está disponible
                            summary = (it.get('summary') or it.get('description') or '').strip()
                            if summary:
                                short = summary if len(summary) <= 220 else (summary[:217] + '…')
                                st.caption(short)
                else:
                    st.caption("Configura FINNHUB_API_KEY para ver titulares.")
            else:
                st.caption("Titulares ocultos.")
        with col2:
            st.subheader("Briefing IA: clima político y perspectivas")
            pplx_key = st.secrets.get('PERPLEXITY_API_KEY', os.getenv('PERPLEXITY_API_KEY', ''))
            if show_ai and pplx_key:
                briefing = perplexity_country_brief(pplx_key, country, news)
                if not briefing or briefing.strip() == "[Sin respuesta]":
                    st.caption("Sin respuesta de IA. Revisa la clave PERPLEXITY_API_KEY o intenta nuevamente.")
                else:
                    st.write(briefing)
            elif not show_ai:
                st.caption("IA desactivada.")
            else:
                st.caption("Configura PERPLEXITY_API_KEY para ver el briefing.")

        # Índice: precio y retornos
        st.markdown("---")
        st.subheader("Índice del país: precio y retornos")
        index_metric_candidates = ['Index Price', 'Price']
        chosen_index_metric = next((m for m in index_metric_candidates if m in metrics_all), None)
        if chosen_index_metric:
            if kind == "sqlite":
                with get_connection(path) as conn:
                    df_idx = load_metric_series_sqlite(conn, country, chosen_index_metric)
            else:
                # Parquet path: derive from df_long filtered
                df_idx = df_long[df_long['metric'] == chosen_index_metric][['DATE', 'value']].copy().sort_values('DATE')
            if not df_idx.empty:
                fig_idx = px.line(df_idx, x='DATE', y='value', title=f"{chosen_index_metric} — {country}")
                fig_idx.update_layout(yaxis_title='Nivel', xaxis_title='Fecha')
                st.plotly_chart(fig_idx, use_container_width=True)
                idx_stats = compute_index_summary(df_idx)
                if idx_stats:
                    st.markdown(
                        f"- Último valor: {idx_stats['last_value']:.2f}\n\n"
                        f"- YTD: {idx_stats['ytd']*100:.2f}%\n\n"
                        f"- 2024: {idx_stats['ret_2024']*100:.2f}% — 2023: {idx_stats['ret_2023']*100:.2f}%\n\n"
                        f"- Volatilidad anualizada (36m): {idx_stats['vol_ann']*100:.2f}%"
                    )
        else:
            st.caption("No se detectó una métrica de índice ('Index Price' o 'Price') para este país.")


if __name__ == "__main__":
    main()