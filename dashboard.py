import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide"
)

# ── LOAD MODEL AND DATA ───────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    features = joblib.load('feature_cols.pkl')
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('clean_traffic.csv')

model, feature_cols = load_model()
df = load_data()

# ── TITLE ─────────────────────────────────────────────────────
st.title("🚦 Traffic Congestion / Bottleneck Predictor")
st.markdown("Predict traffic congestion levels using a trained **XGBoost ML model** (90.58% accuracy)")
st.divider()

# ── SIDEBAR INPUTS ────────────────────────────────────────────
st.sidebar.header("Input parameters")
st.sidebar.markdown("Adjust the values to get a prediction:")

hour       = st.sidebar.slider("Hour of day", 0, 23, 8)
day        = st.sidebar.selectbox("Day of week",
               [0,1,2,3,4,5,6],
               format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                       'Thursday','Friday','Saturday','Sunday'][x])
month      = st.sidebar.slider("Month", 1, 12, 10)
temp       = st.sidebar.slider("Temperature (°C)", -20, 40, 15)
rain       = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
snow       = st.sidebar.slider("Snowfall (mm)", 0.0, 1.0, 0.0)
clouds     = st.sidebar.slider("Cloud cover (%)", 0, 100, 40)
weather    = st.sidebar.selectbox("Weather condition",
               [0,1,2,3,4,5,6,7,8],
               format_func=lambda x: ['Clear','Clouds','Drizzle','Fog',
                                       'Haze','Mist','Rain','Smoke','Snow'][x])

# Auto-calculate derived features
is_weekend   = 1 if day >= 5 else 0
is_rush_hour = 1 if ((7 <= hour <= 9) or (16 <= hour <= 18)) and not is_weekend else 0
is_holiday   = 0  # default

# ── PREDICTION ────────────────────────────────────────────────
input_data = pd.DataFrame([[
    hour, day, month, is_weekend, is_rush_hour,
    is_holiday, temp, rain, snow, clouds, weather
]], columns=feature_cols)

prediction = model.predict(input_data)[0]
proba       = model.predict_proba(input_data)[0]

labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
colors = {0: "green",  1: "orange",   2: "red"}
label  = labels[prediction]
color  = colors[prediction]

# ── MAIN CONTENT ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div style='font-size:18px;color:gray'>Hour selected</div>"
                f"<div style='font-size:29px;font-weight:600'>{hour}:00</div>",
                unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='font-size:18px;color:gray'>Rush hour?</div>"
                f"<div style='font-size:29px;font-weight:600'>{'Yes' if is_rush_hour else 'No'}</div>",
                unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='font-size:18px;color:gray'>Weekend?</div>"
                f"<div style='font-size:29px;font-weight:600'>{'Yes' if is_weekend else 'No'}</div>",
                unsafe_allow_html=True
)
st.divider()

# Prediction result
pred_col, conf_col = st.columns([1, 1])

with pred_col:
    st.subheader("Predicted congestion level")
    st.markdown(f"<h1 style='color:{color};font-size:3rem'>{label}</h1>",
                unsafe_allow_html=True)
    st.markdown(f"**Confidence scores:**")
    st.write(f"🟢 Low:    {proba[0]*100:.1f}%")
    st.write(f"🟡 Medium: {proba[1]*100:.1f}%")
    st.write(f"🔴 High:   {proba[2]*100:.1f}%")

with conf_col:
    st.subheader("Average traffic by hour")
    hourly = df.groupby('hour')['traffic_volume'].mean()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bar_colors = ['#f87171' if h == hour else '#378ADD' for h in hourly.index]
    ax2.bar(hourly.index, hourly.values, color=bar_colors)
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Avg volume')
    ax2.set_title(f'Your selected hour ({hour}:00) in red')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.divider()

# ── CHARTS ────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    with st.expander("Show prediction confidence breakdown"):
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(['Low','Medium','High'],
                      [p*100 for p in proba],
                      color=['#4ade80','#facc15','#f87171'])
        ax.set_ylabel('Confidence (%)')
        ax.set_title('Prediction confidence')
        ax.set_ylim(0, 100)
        for bar, p in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{p*100:.1f}%', ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with chart_col2:
    with st.expander("Show dataset congestion distribution"):
        counts = df['congestion_level'].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.pie(counts.values,
                labels=counts.index,
                colors=['#f87171','#facc15','#4ade80'],
                autopct='%1.1f%%', startangle=90)
        ax3.set_title('Dataset congestion split')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

# ── FEATURE IMPORTANCE ────────────────────────────────────────
st.subheader("Model information")
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("**Model:** XGBoost Classifier")
    st.markdown("**Training samples:** 38,540")
    st.markdown("**Test accuracy:** 90.58%")
    st.markdown("**F1 Score:** 0.9055")
    st.markdown("**Classes:** Low · Medium · High congestion")

with info_col2:
    st.markdown("**Top predictive features:**")
    st.markdown("1. `hour` — 72% importance")
    st.markdown("2. `is_rush_hour` — 10% importance")
    st.markdown("3. `day` — 6% importance")
    st.markdown("4. `is_weekend` — 4% importance")
    st.markdown("5. `temp_celsius` — 4% importance")

st.divider()
st.caption("Traffic Congestion/Bottleneck Prediction · Powered by XGBoost + Streamlit")