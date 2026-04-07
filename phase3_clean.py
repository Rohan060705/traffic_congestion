import pandas as pd
import matplotlib.pyplot as plt

# ── 1. LOAD RAW DATA ──────────────────────────────────────────
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
print("Original shape:", df.shape)

# ── 2. FIX BAD DATA ───────────────────────────────────────────
# Remove rows where temperature is 0 Kelvin (impossible / sensor error)
df = df[df['temp'] > 0]

# Remove extreme rainfall outlier (9831mm is physically impossible)
df = df[df['rain_1h'] < 1000]

# Remove duplicate rows if any exist
df = df.drop_duplicates()

# Check for missing values
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("Shape after cleaning:", df.shape)

# ── 3. FEATURE ENGINEERING ────────────────────────────────────
# Extract time features from the datetime column
df['hour']       = df['date_time'].dt.hour
df['day']        = df['date_time'].dt.dayofweek   # 0=Mon, 6=Sun
df['month']      = df['date_time'].dt.month
df['is_weekend'] = (df['day'] >= 5).astype(int)   # 1 if Sat/Sun, else 0

# Rush hour flag: 1 if 7-9am or 4-6pm on a weekday
df['is_rush_hour'] = (
    ((df['hour'].between(7, 9)) | (df['hour'].between(16, 18))) &
    (df['is_weekend'] == 0)
).astype(int)

# Convert temperature from Kelvin to Celsius (more readable)
df['temp_celsius'] = df['temp'] - 273.15

print("\nNew features added:", ['hour','day','month','is_weekend','is_rush_hour','temp_celsius'])

# ── 4. ENCODE CATEGORICAL COLUMNS ─────────────────────────────
# 'holiday' column has text like "None" or "Christmas Day"
# Convert to binary: 0 = no holiday, 1 = holiday
df['is_holiday'] = (df['holiday'] != 'None').astype(int)

# 'weather_main' has categories like "Clear", "Rain", "Snow"
# Convert using label encoding (each category becomes a number)
df['weather_code'] = df['weather_main'].astype('category').cat.codes
print("\nWeather categories:\n", df['weather_main'].value_counts())

# ── 5. CREATE CONGESTION LABELS ───────────────────────────────
# Based on traffic volume, assign congestion level:
# Low    = below 2000 vehicles/hour
# Medium = 2000 to 4500 vehicles/hour
# High   = above 4500 vehicles/hour
def label_congestion(volume):
    if volume < 2000:
        return 'Low'
    elif volume < 4500:
        return 'Medium'
    else:
        return 'High'

df['congestion_level'] = df['traffic_volume'].apply(label_congestion)
print("\nCongestion level distribution:\n", df['congestion_level'].value_counts())

# Also create numeric version for ML models
df['congestion_code'] = df['congestion_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# ── 6. SELECT FINAL FEATURES ──────────────────────────────────
# Keep only the columns our ML model will use
final_cols = [
    'hour', 'day', 'month', 'is_weekend', 'is_rush_hour',
    'is_holiday', 'temp_celsius', 'rain_1h', 'snow_1h',
    'clouds_all', 'weather_code',
    'traffic_volume', 'congestion_level', 'congestion_code'
]
df_clean = df[final_cols]

print("\nFinal clean dataset shape:", df_clean.shape)
print("\nFirst 5 rows of clean data:\n", df_clean.head())

# ── 7. SAVE CLEAN DATASET ─────────────────────────────────────
df_clean.to_csv('clean_traffic.csv', index=False)
print("\nclean_traffic.csv saved successfully!")

# ── 8. PLOT CONGESTION DISTRIBUTION ───────────────────────────
colors = ['#4ade80', '#facc15', '#f87171']
df_clean['congestion_level'].value_counts().plot(
    kind='bar', color=colors, figsize=(7, 4)
)
plt.title('Congestion level distribution')
plt.xlabel('Congestion level')
plt.ylabel('Number of records')
plt.tight_layout()
plt.savefig('congestion_distribution.png')
plt.show()