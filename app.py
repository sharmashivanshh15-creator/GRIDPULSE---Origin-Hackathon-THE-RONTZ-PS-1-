from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from xgboost import XGBRegressor


# ============================================================
# GRIDPULSE — SELF-CONTAINED STREAMLIT APP
# ============================================================

st.set_page_config(
    page_title="GRIDPULSE",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "demand_weather.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_xgb.json"
FALLBACK_FORECAST = BASE_DIR / "data" / "forecast_168h.csv"

CAPACITY_MW = 8000.0
WATCH = 0.70
HIGH = 0.85
CRITICAL = 0.95

# No blue.
BG = "#0D100E"
PANEL = "#171B18"
PANEL2 = "#1D221E"
BORDER = "#30372F"
TEXT = "#F3F0E7"
MUTED = "#999D93"
GREEN = "#5ED38A"
GOLD = "#E5B85C"
CORAL = "#EF7272"
VIOLET = "#A394F5"
OLIVE = "#9BB56F"

# IMPORTANT: this order must exactly match the trained XGBoost model.
FEATURES = [
    "temperature",
    "humidity",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
    "cooling_degree",
    "heating_degree",
]


# ============================================================
# STYLE
# ============================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {BG};
        }}

        [data-testid="stSidebar"] {{
            background: #111511;
            border-right: 1px solid {BORDER};
        }}

        [data-testid="stSidebar"] * {{
            color: {TEXT};
        }}

        .block-container {{
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3 {{
            color: {TEXT} !important;
            letter-spacing: -0.025em;
        }}

        p {{
            color: {MUTED};
        }}

        [data-testid="stMetric"] {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 15px;
        }}

        [data-testid="stMetricLabel"] {{
            color: {MUTED};
        }}

        [data-testid="stMetricValue"] {{
            color: {TEXT};
        }}

        .hero-title {{
            color: {TEXT};
            font-size: 3rem;
            line-height: 1;
            font-weight: 850;
            letter-spacing: -0.05em;
        }}

        .hero-sub {{
            color: {MUTED};
            margin-top: 8px;
            font-size: 1rem;
        }}

        .eyebrow {{
            color: {MUTED};
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .15em;
            margin-bottom: 5px;
        }}

        .note {{
            color: {MUTED};
            font-size: .78rem;
        }}

        hr {{
            border-color: {BORDER};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_history():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"timestamp", "demand_mw", "temperature", "humidity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ["demand_mw", "temperature", "humidity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "demand_mw", "temperature", "humidity"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.reset_index(drop=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    # GitHub web uploads can leave leading CR/BOM/whitespace in a JSON
    # model file. XGBoost expects the first byte to be the JSON opening brace.
    raw = MODEL_PATH.read_bytes()
    raw = raw.lstrip(b"\xef\xbb\xbf\r\n\t ")

    if not raw.startswith(b"{"):
        preview = raw[:80].decode("utf-8", errors="replace")
        raise ValueError(
            "The uploaded XGBoost model file is not valid JSON. "
            f"First bytes: {preview!r}"
        )

    import tempfile

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp:
        tmp.write(raw)
        clean_path = tmp.name

    try:
        model = XGBRegressor()
        model.load_model(clean_path)
        return model
    finally:
        try:
            Path(clean_path).unlink()
        except OSError:
            pass


# ============================================================
# FEATURES
# ============================================================

def make_features(df):
    x = df.copy()

    x["hour"] = x["timestamp"].dt.hour
    x["day_of_week"] = x["timestamp"].dt.dayofweek
    x["day_of_year"] = x["timestamp"].dt.dayofyear
    x["month"] = x["timestamp"].dt.month
    x["is_weekend"] = (x["day_of_week"] >= 5).astype(int)

    x["hour_sin"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x["hour"] / 24)
    x["dow_sin"] = np.sin(2 * np.pi * x["day_of_week"] / 7)
    x["dow_cos"] = np.cos(2 * np.pi * x["day_of_week"] / 7)

    x["lag_1"] = x["demand_mw"].shift(1)
    x["lag_24"] = x["demand_mw"].shift(24)
    x["lag_168"] = x["demand_mw"].shift(168)

    x["rolling_mean_24"] = x["demand_mw"].shift(1).rolling(24).mean()
    x["rolling_mean_168"] = x["demand_mw"].shift(1).rolling(168).mean()
    x["rolling_std_24"] = x["demand_mw"].shift(1).rolling(24).std()

    x["cooling_degree"] = np.maximum(x["temperature"] - 24, 0)
    x["heating_degree"] = np.maximum(18 - x["temperature"], 0)

    return x


# ============================================================
# OPEN-METEO WEATHER
# ============================================================

@st.cache_data(ttl=900)
def fetch_weather(hours=168):
    """
    Direct Open-Meteo request.
    No API key and no dependency on any project helper module.
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "hourly": "temperature_2m,relative_humidity_2m,shortwave_radiation",
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()

        hourly = payload.get("hourly", {})
        timestamps = hourly.get("time", [])

        if not timestamps:
            return pd.DataFrame()

        out = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "temperature": hourly.get("temperature_2m", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "solar_radiation": hourly.get("shortwave_radiation", []),
        })

        out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce")
        out["humidity"] = pd.to_numeric(out["humidity"], errors="coerce")
        out["solar_radiation"] = pd.to_numeric(
            out["solar_radiation"], errors="coerce"
        )

        return out.dropna(subset=["timestamp", "temperature", "humidity"]).head(hours)

    except Exception:
        return pd.DataFrame()


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def generate_forecast(history, model, weather):
    if weather.empty:
        return pd.DataFrame()

    work = history[
        ["timestamp", "demand_mw", "temperature", "humidity"]
    ].copy()

    results = []

    for _, w in weather.iterrows():
        ts = pd.Timestamp(w["timestamp"])
        temp = float(w["temperature"])
        humidity = float(w["humidity"])

        new_row = pd.DataFrame([{
            "timestamp": ts,
            "demand_mw": np.nan,
            "temperature": temp,
            "humidity": humidity,
        }])

        temp_work = pd.concat([work, new_row], ignore_index=True)

        # Temporary seed is only used to construct lag features.
        # The predicted value replaces it immediately afterward.
        temp_work.loc[temp_work.index[-1], "demand_mw"] = work["demand_mw"].iloc[-1]

        feat = make_features(temp_work)
        model_features = model.get_booster().feature_names or FEATURES
        missing_features = [c for c in model_features if c not in feat.columns]
        if missing_features:
            raise ValueError(f"Missing model features: {missing_features}")

        current = feat.iloc[[-1]][model_features]

        prediction = float(model.predict(current)[0])
        prediction = max(0.0, prediction)

        temp_work.loc[temp_work.index[-1], "demand_mw"] = prediction

        results.append({
            "timestamp": ts,
            "predicted_mw": prediction,
            "temperature": temp,
            "humidity": humidity,
            "solar_radiation": float(w["solar_radiation"])
            if pd.notna(w["solar_radiation"])
            else 0.0,
        })

        work = temp_work

    return pd.DataFrame(results)


# ============================================================
# RISK HELPERS
# ============================================================

def risk_label(mw):
    ratio = float(mw) / CAPACITY_MW

    if ratio >= CRITICAL:
        return "CRITICAL"
    if ratio >= HIGH:
        return "HIGH"
    if ratio >= WATCH:
        return "WATCH"
    return "NORMAL"


def risk_color(label):
    return {
        "NORMAL": GREEN,
        "WATCH": GOLD,
        "HIGH": GOLD,
        "CRITICAL": CORAL,
    }.get(label, MUTED)


# ============================================================
# PLOTLY HELPERS
# ============================================================

def layout(title=None, height=430):
    return dict(
        title=title,
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            color=TEXT,
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        ),
        margin=dict(l=55, r=25, t=55 if title else 20, b=50),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED),
        ),
        xaxis=dict(
            gridcolor="rgba(243,240,231,.07)",
            linecolor=BORDER,
        ),
        yaxis=dict(
            gridcolor="rgba(243,240,231,.07)",
            linecolor=BORDER,
        ),
    )


def forecast_chart(df, title="Demand forecast"):
    fig = go.Figure()

    if df.empty:
        return fig

    fig.add_hrect(
        y0=0,
        y1=CAPACITY_MW * WATCH,
        fillcolor="rgba(94,211,138,.045)",
        line_width=0,
    )
    fig.add_hrect(
        y0=CAPACITY_MW * WATCH,
        y1=CAPACITY_MW * HIGH,
        fillcolor="rgba(229,184,92,.055)",
        line_width=0,
    )
    fig.add_hrect(
        y0=CAPACITY_MW * HIGH,
        y1=CAPACITY_MW * CRITICAL,
        fillcolor="rgba(229,184,92,.10)",
        line_width=0,
    )
    fig.add_hrect(
        y0=CAPACITY_MW * CRITICAL,
        y1=CAPACITY_MW * 1.10,
        fillcolor="rgba(239,114,114,.12)",
        line_width=0,
    )

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["predicted_mw"],
        mode="lines",
        name="Forecast",
        line=dict(color=GOLD, width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(229,184,92,.07)",
        hovertemplate="%{x}<br>%{y:,.0f} MW<extra></extra>",
    ))

    peak_idx = df["predicted_mw"].idxmax()
    peak = df.loc[peak_idx]

    fig.add_trace(go.Scatter(
        x=[peak["timestamp"]],
        y=[peak["predicted_mw"]],
        mode="markers",
        name="Peak",
        marker=dict(
            color=CORAL if peak["predicted_mw"] >= CAPACITY_MW * HIGH else GOLD,
            size=12,
            line=dict(color=TEXT, width=2),
        ),
        hovertemplate=(
            "Forecast peak<br>%{x}<br>%{y:,.0f} MW<extra></extra>"
        ),
    ))

    fig.add_hline(
        y=CAPACITY_MW,
        line_dash="dash",
        line_color=CORAL,
        line_width=2,
        annotation_text="8,000 MW CAPACITY",
        annotation_font_color=CORAL,
    )

    fig.update_layout(**layout(title, 470))
    fig.update_yaxes(title="Demand (MW)", rangemode="tozero")
    return fig


def utilization_chart(df):
    fig = go.Figure()

    if df.empty:
        return fig

    util = df["predicted_mw"] / CAPACITY_MW * 100

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=util,
        mode="lines",
        name="Utilization",
        line=dict(color=GREEN, width=3),
        fill="tozeroy",
        fillcolor="rgba(94,211,138,.07)",
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ))

    for value, color, label in [
        (70, GOLD, "WATCH"),
        (85, GOLD, "HIGH"),
        (95, CORAL, "CRITICAL"),
    ]:
        fig.add_hline(
            y=value,
            line_dash="dot",
            line_color=color,
            annotation_text=f"{label} {value}%",
            annotation_font_color=color,
        )

    fig.update_layout(**layout("Capacity utilization", 410))
    fig.update_yaxes(
        title="Utilization",
        ticksuffix="%",
        range=[0, 105],
    )
    return fig


def daily_peak_chart(df):
    if df.empty:
        return go.Figure()

    d = df.copy()
    d["date"] = d["timestamp"].dt.date

    daily = d.groupby("date")["predicted_mw"].max().reset_index()

    colors = [
        CORAL if x >= CAPACITY_MW * CRITICAL
        else GOLD if x >= CAPACITY_MW * WATCH
        else GREEN
        for x in daily["predicted_mw"]
    ]

    fig = go.Figure(go.Bar(
        x=daily["date"],
        y=daily["predicted_mw"],
        marker_color=colors,
        hovertemplate="%{x}<br>Peak %{y:,.0f} MW<extra></extra>",
    ))

    fig.add_hline(
        y=CAPACITY_MW,
        line_dash="dash",
        line_color=CORAL,
        annotation_text="8,000 MW",
        annotation_font_color=CORAL,
    )

    fig.update_layout(**layout("Daily peak exposure", 390))
    fig.update_yaxes(title="Peak MW")
    return fig


def daily_average_chart(df):
    if df.empty:
        return go.Figure()

    d = df.copy()
    d["date"] = d["timestamp"].dt.date
    daily = d.groupby("date")["predicted_mw"].mean().reset_index()

    fig = go.Figure(go.Scatter(
        x=daily["date"],
        y=daily["predicted_mw"],
        mode="lines+markers",
        line=dict(color=VIOLET, width=3),
        marker=dict(color=VIOLET, size=8),
        fill="tozeroy",
        fillcolor="rgba(163,148,245,.08)",
        hovertemplate="%{x}<br>%{y:,.0f} MW<extra></extra>",
    ))

    fig.update_layout(**layout("Daily average demand", 390))
    fig.update_yaxes(title="Average MW")
    return fig


def risk_donut(df):
    if df.empty:
        return go.Figure()

    labels = df["predicted_mw"].apply(risk_label)
    counts = labels.value_counts()

    order = ["NORMAL", "WATCH", "HIGH", "CRITICAL"]
    values = [int(counts.get(x, 0)) for x in order]

    fig = go.Figure(go.Pie(
        labels=order,
        values=values,
        hole=.68,
        marker=dict(
            colors=[GREEN, GOLD, "#C8943E", CORAL],
            line=dict(color=PANEL, width=3),
        ),
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value} hours<extra></extra>",
    ))

    fig.update_layout(
        **layout("Risk distribution", 410),
        showlegend=False,
        annotations=[dict(
            text=f"{len(df)}<br>hours",
            x=.5,
            y=.5,
            showarrow=False,
            font=dict(size=20, color=TEXT),
        )],
    )
    return fig


def weather_chart(df):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=.08,
        subplot_titles=("Temperature", "Humidity", "Solar radiation"),
    )

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        mode="lines",
        name="Temperature",
        line=dict(color=VIOLET, width=3),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["humidity"],
        mode="lines",
        name="Humidity",
        line=dict(color=GREEN, width=3),
        fill="tozeroy",
        fillcolor="rgba(94,211,138,.07)",
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["solar_radiation"],
        name="Solar",
        marker_color=GOLD,
    ), row=3, col=1)

    fig.update_layout(**layout("Weather drivers", 720), showlegend=False)
    fig.update_yaxes(title="°C", row=1, col=1)
    fig.update_yaxes(title="%", row=2, col=1)
    fig.update_yaxes(title="W/m²", row=3, col=1)

    return fig


def temp_demand_chart(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["predicted_mw"],
        mode="lines",
        name="Demand",
        line=dict(color=GOLD, width=3),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        mode="lines",
        name="Temperature",
        line=dict(color=VIOLET, width=2.5),
    ), secondary_y=True)

    fig.update_layout(**layout("Temperature vs predicted demand", 430))
    fig.update_yaxes(title="Demand MW", secondary_y=False)
    fig.update_yaxes(title="Temperature °C", secondary_y=True)
    return fig


def gauge(value):
    pct = value / CAPACITY_MW * 100

    if pct >= 95:
        bar = CORAL
    elif pct >= 70:
        bar = GOLD
    else:
        bar = GREEN

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(
            suffix="%",
            font=dict(size=36, color=TEXT),
        ),
        title=dict(
            text="CURRENT CAPACITY LOAD",
            font=dict(size=13, color=MUTED),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor=MUTED,
            ),
            bar=dict(color=bar, thickness=.72),
            bgcolor=PANEL2,
            borderwidth=0,
            steps=[
                dict(range=[0, 70], color="rgba(94,211,138,.07)"),
                dict(range=[70, 85], color="rgba(229,184,92,.08)"),
                dict(range=[85, 95], color="rgba(229,184,92,.14)"),
                dict(range=[95, 100], color="rgba(239,114,114,.15)"),
            ],
        ),
    ))

    fig.update_layout(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        height=280,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


# ============================================================
# INITIALIZE
# ============================================================

try:
    history = load_history()
    model = load_model()
except Exception as exc:
    st.error("GRIDPULSE could not start.")
    st.code(str(exc))
    st.stop()

weather = fetch_weather(168)
forecast = generate_forecast(history, model, weather)

# If Open-Meteo is temporarily unavailable, use the already-generated
# forecast file rather than crashing the dashboard.
if forecast.empty and FALLBACK_FORECAST.exists():
    try:
        forecast = pd.read_csv(FALLBACK_FORECAST)
        forecast.columns = [str(c).strip() for c in forecast.columns]
        forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
        forecast["predicted_mw"] = pd.to_numeric(
            forecast["predicted_mw"], errors="coerce"
        )
        for col in ["temperature", "humidity", "solar_radiation"]:
            if col not in forecast.columns:
                forecast[col] = 0.0
    except Exception:
        forecast = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:8px 0 14px">
            <div style="
                color:{TEXT};
                font-size:1.75rem;
                font-weight:900;
                letter-spacing:-.05em;
            ">GRIDPULSE</div>
            <div style="color:{MUTED};font-size:.78rem">
                Electricity Demand Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Command Center",
            "Demand Forecast",
            "Grid Risk",
            "What-If Lab",
            "Weather",
            "Data & Model",
        ],
    )

    st.divider()

    st.caption("DELHI AGGREGATE")
    st.caption("Capacity · 8,000 MW")
    st.caption("Model · XGBoost")
    st.caption("Weather · Open-Meteo")
    st.caption("No hardcoded forecast values")

    st.divider()
    st.caption(
        "Historical demand is available through "
        f"{history['timestamp'].max().strftime('%d %b %Y %H:%M')}."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div style="padding-bottom:18px">
        <div class="hero-title">GRIDPULSE</div>
        <div class="hero-sub">
            Delhi electricity demand forecasting · capacity exposure · weather intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT STATE
# ============================================================

current = float(history["demand_mw"].iloc[-1])
util = current / CAPACITY_MW * 100
current_risk = risk_label(current)

if not forecast.empty:
    peak_row = forecast.loc[forecast["predicted_mw"].idxmax()]
    peak = float(peak_row["predicted_mw"])
    peak_time = pd.Timestamp(peak_row["timestamp"])
else:
    peak = np.nan
    peak_time = None


# ============================================================
# COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown('<div class="eyebrow">OPERATIONS OVERVIEW</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Latest observed demand", f"{current:,.0f} MW")
    c2.metric("Capacity utilization", f"{util:.1f}%")
    c3.metric(
        "Forecast peak",
        f"{peak:,.0f} MW" if not np.isnan(peak) else "—",
    )
    c4.metric(
        "Peak headroom",
        f"{CAPACITY_MW - peak:,.0f} MW"
        if not np.isnan(peak)
        else "—",
    )

    st.caption(
        "Observed demand comes from the supplied historical dataset. "
        "Forecast weather is refreshed from Open-Meteo; no live SLDC value is presented as current."
    )

    if current_risk == "NORMAL":
        st.success("GRID STATUS · NORMAL")
    elif current_risk in ["WATCH", "HIGH"]:
        st.warning(f"GRID STATUS · {current_risk}")
    else:
        st.error("GRID STATUS · CRITICAL")

    st.plotly_chart(
        forecast_chart(forecast, "Demand trajectory · forecast horizon"),
        use_container_width=True,
    )

    left, right = st.columns([1.7, 1])

    with left:
        st.plotly_chart(
            utilization_chart(forecast),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            gauge(current),
            use_container_width=True,
        )

    a, b = st.columns(2)

    with a:
        st.plotly_chart(
            daily_peak_chart(forecast),
            use_container_width=True,
        )

    with b:
        st.plotly_chart(
            daily_average_chart(forecast),
            use_container_width=True,
        )

    if peak_time is not None:
        st.info(
            f"Forecast peak: {peak:,.0f} MW around "
            f"{peak_time.strftime('%d %b %Y · %H:%M')}."
        )


# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "Demand Forecast":

    st.markdown('<div class="eyebrow">DEMAND FORECAST</div>',
                unsafe_allow_html=True)

    st.title("24-hour operating profile")

    if forecast.empty:
        st.error("No forecast is available.")
        st.stop()

    f24 = forecast.head(24).copy()

    st.plotly_chart(
        forecast_chart(f24, "Next 24 hours"),
        use_container_width=True,
    )

    hourly = f24[
        ["timestamp", "predicted_mw", "temperature", "humidity"]
    ].copy()

    hourly["risk"] = hourly["predicted_mw"].apply(risk_label)
    hourly["capacity_pct"] = hourly["predicted_mw"] / CAPACITY_MW * 100

    hourly.columns = [
        "Time",
        "Forecast MW",
        "Temperature °C",
        "Humidity %",
        "Risk",
        "Capacity %",
    ]

    st.dataframe(
        hourly.style.format({
            "Forecast MW": "{:,.0f}",
            "Temperature °C": "{:.1f}",
            "Humidity %": "{:.0f}",
            "Capacity %": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.title("7-day outlook")

    st.plotly_chart(
        daily_peak_chart(forecast),
        use_container_width=True,
    )

    d = forecast.copy()
    d["date"] = d["timestamp"].dt.date

    daily = d.groupby("date").agg(
        peak_mw=("predicted_mw", "max"),
        average_mw=("predicted_mw", "mean"),
        minimum_mw=("predicted_mw", "min"),
    ).reset_index()

    daily["risk"] = daily["peak_mw"].apply(risk_label)
    daily["headroom_mw"] = CAPACITY_MW - daily["peak_mw"]

    daily.columns = [
        "Date",
        "Peak MW",
        "Average MW",
        "Minimum MW",
        "Risk",
        "Headroom MW",
    ]

    st.dataframe(
        daily.style.format({
            "Peak MW": "{:,.0f}",
            "Average MW": "{:,.0f}",
            "Minimum MW": "{:,.0f}",
            "Headroom MW": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GRID RISK
# ============================================================

elif page == "Grid Risk":

    st.markdown('<div class="eyebrow">GRID RISK ENGINE</div>',
                unsafe_allow_html=True)

    st.title("Capacity exposure")

    if forecast.empty:
        st.error("No forecast is available.")
        st.stop()

    labels = forecast["predicted_mw"].apply(risk_label)

    normal = int((labels == "NORMAL").sum())
    watch = int((labels == "WATCH").sum())
    high = int((labels == "HIGH").sum())
    critical = int((labels == "CRITICAL").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Normal", f"{normal} h")
    c2.metric("Watch", f"{watch} h")
    c3.metric("High", f"{high} h")
    c4.metric("Critical", f"{critical} h")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            utilization_chart(forecast),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            risk_donut(forecast),
            use_container_width=True,
        )

    peak_row = forecast.loc[forecast["predicted_mw"].idxmax()]
    peak = float(peak_row["predicted_mw"])
    peak_pct = peak / CAPACITY_MW * 100
    headroom = CAPACITY_MW - peak

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast peak", f"{peak:,.0f} MW")
    c2.metric("Peak utilization", f"{peak_pct:.1f}%")
    c3.metric("Remaining headroom", f"{headroom:,.0f} MW")

    st.caption(
        f"Peak window: {pd.Timestamp(peak_row['timestamp']).strftime('%d %b %Y · %H:%M')}"
    )

    if peak_pct >= 95:
        st.error("Critical exposure: forecast is at or above the 95% capacity threshold.")
    elif peak_pct >= 85:
        st.warning("High exposure: forecast enters the high-utilization band.")
    elif peak_pct >= 70:
        st.warning("Watch exposure: forecast enters the operational watch band.")
    else:
        st.success("Forecast remains below the watch threshold.")


# ============================================================
# WHAT-IF LAB
# ============================================================

elif page == "What-If Lab":

    st.markdown('<div class="eyebrow">SCENARIO ANALYSIS</div>',
                unsafe_allow_html=True)

    st.title("What-If Lab")

    st.caption(
        "Explore directional effects of weather and demand-management assumptions."
    )

    if forecast.empty:
        st.error("No forecast is available.")
        st.stop()

    temp_change = st.slider(
        "Temperature change (°C)",
        -5.0, 5.0, 0.0, 0.5,
    )

    efficiency = st.slider(
        "Efficiency reduction in demand (%)",
        0.0, 10.0, 0.0, 1.0,
    )

    response = st.slider(
        "Demand-response reduction (%)",
        0.0, 15.0, 0.0, 1.0,
    )

    scenario = forecast.copy()

    # A transparent scenario sensitivity layer.
    # It is intentionally separate from the trained model forecast.
    extra_cooling = (
        np.maximum(scenario["temperature"] + temp_change - 24, 0)
        - np.maximum(scenario["temperature"] - 24, 0)
    )

    scenario["scenario_mw"] = (
        scenario["predicted_mw"]
        + extra_cooling * 70.0
        - scenario["predicted_mw"] * efficiency / 100.0
        - scenario["predicted_mw"] * response / 100.0
    )

    base_peak = float(forecast["predicted_mw"].max())
    scenario_peak = float(scenario["scenario_mw"].max())

    c1, c2, c3 = st.columns(3)
    c1.metric("Base peak", f"{base_peak:,.0f} MW")
    c2.metric(
        "Scenario peak",
        f"{scenario_peak:,.0f} MW",
        f"{scenario_peak - base_peak:+,.0f} MW",
    )
    c3.metric(
        "Scenario headroom",
        f"{CAPACITY_MW - scenario_peak:,.0f} MW",
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast["timestamp"],
        y=forecast["predicted_mw"],
        mode="lines",
        name="Base forecast",
        line=dict(color=MUTED, width=2),
    ))

    fig.add_trace(go.Scatter(
        x=scenario["timestamp"],
        y=scenario["scenario_mw"],
        mode="lines",
        name="Scenario",
        line=dict(color=GOLD, width=3),
        fill="tozeroy",
        fillcolor="rgba(229,184,92,.07)",
    ))

    fig.add_hline(
        y=CAPACITY_MW,
        line_dash="dash",
        line_color=CORAL,
        annotation_text="8,000 MW capacity",
        annotation_font_color=CORAL,
    )

    fig.update_layout(**layout("Base forecast vs scenario", 480))
    fig.update_yaxes(title="Demand MW")

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Scenario values are analytical sensitivity estimates, not replacement "
        "forecasts from a retrained model."
    )


# ============================================================
# WEATHER
# ============================================================

elif page == "Weather":

    st.markdown('<div class="eyebrow">WEATHER INTELLIGENCE</div>',
                unsafe_allow_html=True)

    st.title("Weather drivers")

    if forecast.empty:
        st.error("No weather forecast is available.")
        st.stop()

    st.plotly_chart(
        weather_chart(forecast),
        use_container_width=True,
    )

    st.plotly_chart(
        temp_demand_chart(forecast),
        use_container_width=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Maximum temperature",
        f"{forecast['temperature'].max():.1f} °C",
    )

    c2.metric(
        "Average humidity",
        f"{forecast['humidity'].mean():.0f}%",
    )

    c3.metric(
        "Peak solar radiation",
        f"{forecast['solar_radiation'].max():.0f} W/m²",
    )

    st.caption(
        "Delhi weather coordinates: 28.6139° N, 77.2090° E. "
        "Weather is fetched directly from Open-Meteo."
    )


# ============================================================
# DATA & MODEL
# ============================================================

elif page == "Data & Model":

    st.markdown('<div class="eyebrow">DATA & MODEL</div>',
                unsafe_allow_html=True)

    st.title("Model transparency")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical rows", f"{len(history):,}")
    c2.metric(
        "Dataset start",
        history["timestamp"].min().strftime("%d %b %Y"),
    )
    c3.metric(
        "Dataset through",
        history["timestamp"].max().strftime("%d %b %Y"),
    )
    c4.metric("Model", "XGBoost")

    st.markdown("---")

    hist = history.tail(336)

    fig = go.Figure(go.Scatter(
        x=hist["timestamp"],
        y=hist["demand_mw"],
        mode="lines",
        name="Observed demand",
        line=dict(color=GREEN, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(94,211,138,.07)",
    ))

    fig.add_hline(
        y=CAPACITY_MW,
        line_dash="dash",
        line_color=CORAL,
        annotation_text="8,000 MW",
        annotation_font_color=CORAL,
    )

    fig.update_layout(**layout("Recent historical demand", 450))
    fig.update_yaxes(title="Demand MW")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature groups")

    feature_df = pd.DataFrame({
        "Feature": FEATURES,
        "Group": [
            "Calendar", "Calendar", "Calendar", "Calendar", "Calendar",
            "Cyclical", "Cyclical", "Cyclical", "Cyclical",
            "Historical lag", "Historical lag", "Historical lag",
            "Rolling", "Rolling", "Rolling",
            "Weather", "Weather", "Weather", "Weather",
        ],
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset integrity")

    duplicate_timestamps = int(history["timestamp"].duplicated().sum())
    missing_values = int(history[
        ["timestamp", "demand_mw", "temperature", "humidity"]
    ].isna().sum().sum())

    intervals = history["timestamp"].diff().dropna()
    non_hourly = int((intervals != pd.Timedelta(hours=1)).sum())

    i1, i2, i3 = st.columns(3)
    i1.metric("Duplicate timestamps", duplicate_timestamps)
    i2.metric("Missing values", missing_values)
    i3.metric("Non-hourly intervals", non_hourly)

    if duplicate_timestamps == 0 and missing_values == 0 and non_hourly == 0:
        st.success("Historical dataset passes continuity and completeness checks.")
    else:
        st.warning("Review dataset integrity metrics above.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "GRIDPULSE · Delhi aggregate electricity demand intelligence · "
    f"Rendered {datetime.now().astimezone().strftime('%d %b %Y %H:%M %Z')}"
)
