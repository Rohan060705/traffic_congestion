import pandas as pd
import folium

df = pd.read_csv('chicago_traffic.csv')
df.columns = df.columns.str.strip()
df = df[df['CURRENT_SPEED'] > 0]

# ── DOWNTOWN LOOP BOUNDING BOX ────────────────────────────────
lat_min, lat_max = 41.875, 41.900
lon_min, lon_max = -87.640, -87.620

df_loop = df[
    (df['START_LATITUDE'].between(lat_min, lat_max)) &
    (df['START_LONGITUDE'].between(lon_min, lon_max)) &
    (df['END_LATITUDE'].between(lat_min, lat_max)) &
    (df['END_LONGITUDE'].between(lon_min, lon_max))
].copy().reset_index(drop=True)

# Save for dashboard use
df_loop.to_csv('chicago_small.csv', index=False)
print(f"Saved {len(df_loop)} downtown segments")

# ── COLOR AND LABEL HELPERS ───────────────────────────────────
def speed_color(speed):
    if speed >= 25: return '#4ade80'
    elif speed >= 18: return '#facc15'
    else: return '#f87171'

def congestion_label(speed):
    if speed >= 25: return 'Free flow'
    elif speed >= 18: return 'Moderate'
    else: return 'Congested'

# ── BUILD MAP ─────────────────────────────────────────────────
m = folium.Map(
    location=[41.886, -87.628],
    zoom_start=15,
    tiles='CartoDB dark_matter',
    zoom_control=True
)

# Lock map to downtown only — no zooming out to full Chicago
m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

for _, row in df_loop.iterrows():
    color  = speed_color(row['CURRENT_SPEED'])
    label  = congestion_label(row['CURRENT_SPEED'])

    # Road segment line
    folium.PolyLine(
        locations=[
            [row['START_LATITUDE'], row['START_LONGITUDE']],
            [row['END_LATITUDE'],   row['END_LONGITUDE']]
        ],
        color=color,
        weight=7,
        opacity=0.95,
        tooltip=(
            f"<b style='font-size:13px'>{row['STREET']}</b><br>"
            f"{row['FROM_STREET']} → {row['TO_STREET']}<br>"
            f"Speed: <b>{row['CURRENT_SPEED']} mph</b><br>"
            f"Status: <b style='color:{color}'>{label}</b>"
        )
    ).add_to(m)

    # Node at start point
    folium.CircleMarker(
        location=[row['START_LATITUDE'], row['START_LONGITUDE']],
        radius=6,
        color='#111',
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=1,
        tooltip=f"<b>{row['STREET']}</b> @ {row['FROM_STREET']}<br>{row['CURRENT_SPEED']} mph — {label}"
    ).add_to(m)

    # Node at end point
    folium.CircleMarker(
        location=[row['END_LATITUDE'], row['END_LONGITUDE']],
        radius=6,
        color='#111',
        weight=1.5,
        fill=True,
        fill_color=color,
        fill_opacity=1,
        tooltip=f"<b>{row['STREET']}</b> @ {row['TO_STREET']}<br>{row['CURRENT_SPEED']} mph — {label}"
    ).add_to(m)

    # Street label at midpoint
    mid_lat = (row['START_LATITUDE'] + row['END_LATITUDE']) / 2
    mid_lon = (row['START_LONGITUDE'] + row['END_LONGITUDE']) / 2
    folium.Marker(
        location=[mid_lat, mid_lon],
        icon=folium.DivIcon(
            html=f"""<div style="
                font-size:9px;font-weight:700;color:#fff;
                white-space:nowrap;pointer-events:none;
                text-shadow:0 0 4px #000,0 0 4px #000,0 0 4px #000;">
                {row['STREET']}
            </div>""",
            icon_size=(90, 16),
            icon_anchor=(45, -6)
        )
    ).add_to(m)

# ── LEGEND ────────────────────────────────────────────────────
legend_html = """
<div style="position:fixed;bottom:24px;left:24px;z-index:1000;
     background:#1a1a18;padding:12px 16px;border-radius:10px;
     border:1px solid #444;font-family:sans-serif;min-width:190px">
  <div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:8px">
    Chicago Loop — live traffic
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#4ade80;font-size:18px">&#9632;</span>
    Free flow &nbsp;<span style="color:#555">25+ mph</span>
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#facc15;font-size:18px">&#9632;</span>
    Moderate &nbsp;<span style="color:#555">18–25 mph</span>
  </div>
  <div style="font-size:12px;color:#aaa;margin:5px 0">
    <span style="color:#f87171;font-size:18px">&#9632;</span>
    Congested &nbsp;<span style="color:#555">under 18 mph</span>
  </div>
  <div style="font-size:11px;color:#555;margin-top:10px;border-top:1px solid #333;padding-top:8px">
    19 real road segments · Chicago open data
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save('chicago_map.html')
print("chicago_map.html saved — open in browser to preview!")