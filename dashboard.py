import streamlit as st
import streamlit.components.v1 as components
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
st.divider()
st.subheader("🚗 Live traffic simulation")
st.markdown("See how congestion builds up in real time — adjust volume and lanes, add an accident to create a bottleneck.")

st.divider()
st.subheader("🗺️ Chicago Loop — interactive congestion map")
st.markdown("Real road segments from Chicago city data · adjust lanes and incidents per segment · model predicts updated congestion")

import folium
from streamlit_folium import st_folium

df_chicago = pd.read_csv('chicago_small.csv')
df_chicago.columns = df_chicago.columns.str.strip()
df_chicago = df_chicago.reset_index(drop=True)

# ── SEGMENT CONTROLS IN SIDEBAR ───────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("### Chicago map controls")

selected_street = st.sidebar.selectbox(
    "Select road segment",
    df_chicago['STREET'].tolist(),
    key='chicago_street'
)

seg_idx = df_chicago[df_chicago['STREET'] == selected_street].index[0]

sim_lanes    = st.sidebar.slider("Number of lanes", 1, 4, 2, key='chi_lanes')
sim_accident = st.sidebar.toggle("Accident on this segment", value=False, key='chi_acc')
sim_hour     = st.sidebar.slider("Simulate hour of day", 0, 23, hour, key='chi_hour')

# ── COMPUTE SIMULATED SPEED FOR SELECTED SEGMENT ──────────────
base_speed = df_chicago.loc[seg_idx, 'CURRENT_SPEED']

# Lane reduction slows traffic proportionally
lane_factor  = sim_lanes / 2.0
# Accident cuts speed drastically
acc_factor   = 0.25 if sim_accident else 1.0
# Rush hour (7-9am, 4-6pm) reduces speed
rush         = (7 <= sim_hour <= 9) or (16 <= sim_hour <= 18)
rush_factor  = 0.6 if rush else 1.0

simulated_speed = max(2, round(base_speed * lane_factor * acc_factor * rush_factor))

# ── USE ML MODEL TO PREDICT CONGESTION FOR THIS SEGMENT ───────
seg_input = pd.DataFrame([[
    sim_hour,
    day,
    month,
    1 if day >= 5 else 0,
    1 if rush and day < 5 else 0,
    0,
    temp,
    rain,
    snow,
    clouds,
    weather
]], columns=feature_cols)

seg_pred  = model.predict(seg_input)[0]
seg_proba = model.predict_proba(seg_input)[0]
seg_label = {0:'Low', 1:'Medium', 2:'High'}[seg_pred]
seg_color_map = {0:'#4ade80', 1:'#facc15', 2:'#f87171'}
seg_color = seg_color_map[seg_pred]

# ── SHOW PREDICTION FOR SELECTED SEGMENT ──────────────────────
pred_col1, pred_col2, pred_col3 = st.columns(3)
with pred_col1:
    st.metric("Segment", selected_street)
with pred_col2:
    st.metric("Simulated speed", f"{simulated_speed} mph")
with pred_col3:
    st.metric("ML prediction", seg_label)

if sim_accident:
    st.warning(f"Accident detected on {selected_street} — speed reduced to {simulated_speed} mph · congestion spreading to nearby segments")
if rush:
    st.info(f"Rush hour active at {sim_hour}:00 — expect higher congestion on all segments")

# ── SPEED COLOR HELPERS ───────────────────────────────────────
def speed_color(speed, is_selected=False, is_accident=False):
    if is_accident: return '#e879f9'
    if speed >= 25: return '#4ade80'
    elif speed >= 18: return '#facc15'
    else: return '#f87171'

def congestion_label(speed):
    if speed >= 25: return 'Free flow'
    elif speed >= 18: return 'Moderate'
    else: return 'Congested'

# ── BUILD UPDATED MAP ─────────────────────────────────────────
m2 = folium.Map(
    location=[41.886, -87.628],
    zoom_start=15,
    tiles='CartoDB dark_matter'
)
m2.fit_bounds([[41.875, -87.640], [41.900, -87.620]])

for idx, row in df_chicago.iterrows():
    is_selected = (idx == seg_idx)
    is_accident = is_selected and sim_accident

    # Apply simulation to selected segment
    if is_selected:
        display_speed = simulated_speed
    else:
        # Accident on selected segment spreads congestion to neighbors
        if sim_accident:
            display_speed = max(5, round(row['CURRENT_SPEED'] * 0.7))
        else:
            display_speed = row['CURRENT_SPEED']

    color = speed_color(display_speed, is_selected, is_accident)
    label = congestion_label(display_speed)
    weight = 10 if is_selected else 6
    opacity = 1.0 if is_selected else 0.85

    folium.PolyLine(
        locations=[
            [row['START_LATITUDE'], row['START_LONGITUDE']],
            [row['END_LATITUDE'],   row['END_LONGITUDE']]
        ],
        color=color,
        weight=weight,
        opacity=opacity,
        tooltip=(
            f"<b style='font-size:13px'>{row['STREET']}</b>"
            f"{'  ⭐ SELECTED' if is_selected else ''}"
            f"{'  🚨 ACCIDENT' if is_accident else ''}<br>"
            f"{row['FROM_STREET']} → {row['TO_STREET']}<br>"
            f"Speed: <b>{display_speed} mph</b><br>"
            f"Status: <b>{label}</b>"
        )
    ).add_to(m2)

    # Start node
    folium.CircleMarker(
        location=[row['START_LATITUDE'], row['START_LONGITUDE']],
        radius=7 if is_selected else 5,
        color='#fff' if is_selected else '#111',
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=1,
        tooltip=f"{row['STREET']} @ {row['FROM_STREET']} — {display_speed} mph"
    ).add_to(m2)

    # End node
    folium.CircleMarker(
        location=[row['END_LATITUDE'], row['END_LONGITUDE']],
        radius=7 if is_selected else 5,
        color='#fff' if is_selected else '#111',
        weight=2,
        fill=True,
        fill_color=color,
        fill_opacity=1,
        tooltip=f"{row['STREET']} @ {row['TO_STREET']} — {display_speed} mph"
    ).add_to(m2)

    # Street label
    mid_lat = (row['START_LATITUDE'] + row['END_LATITUDE']) / 2
    mid_lon = (row['START_LONGITUDE'] + row['END_LONGITUDE']) / 2
    folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(
            html=f"""<div style="
                font-size:{'11' if is_selected else '9'}px;
                font-weight:700;color:#fff;white-space:nowrap;
                text-shadow:0 0 4px #000,0 0 4px #000;">
                {row['STREET']}{'  *' if is_selected else ''}
            </div>""",
            icon_size=(100, 16),
            icon_anchor=(50, -6)
        )
    ).add_to(m2)

# Legend
legend_html = f"""
<div style="position:fixed;bottom:24px;left:24px;z-index:1000;
     background:#1a1a18;padding:12px 16px;border-radius:10px;
     border:1px solid #444;font-family:sans-serif;min-width:200px">
  <div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:8px">
    Chicago Loop · {sim_hour}:00
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#4ade80;font-size:16px">&#9632;</span> Free flow (25+ mph)
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#facc15;font-size:16px">&#9632;</span> Moderate (18–25 mph)
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#f87171;font-size:16px">&#9632;</span> Congested (under 18 mph)
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#e879f9;font-size:16px">&#9632;</span> Incident
  </div>
  <div style="font-size:11px;color:#666;margin-top:8px;
       border-top:1px solid #333;padding-top:8px">
    Selected: <b style="color:#fff">{selected_street}</b><br>
    ML prediction: <b style="color:{seg_color}">{seg_label} congestion</b>
  </div>
</div>
"""
m2.get_root().html.add_child(folium.Element(legend_html))

# ── RENDER MAP + STATS SIDE BY SIDE ───────────────────────────
map_col, stat_col = st.columns([2, 1])

with map_col:
    st_folium(m2, width=550, height=440)

with stat_col:
    st.markdown("**All segment speeds**")
    for idx, row in df_chicago.iterrows():
        is_selected = (idx == seg_idx)
        is_accident = is_selected and sim_accident
        spd = simulated_speed if is_selected else (
            max(5, round(row['CURRENT_SPEED'] * 0.7)) if sim_accident else row['CURRENT_SPEED']
        )
        dot = "🟣" if is_accident else ("🟢" if spd >= 25 else "🟡" if spd >= 18 else "🔴")
        bold_start = "**" if is_selected else ""
        bold_end   = "**" if is_selected else ""
        st.markdown(f"{dot} {bold_start}{row['STREET']}{bold_end} — {spd} mph")