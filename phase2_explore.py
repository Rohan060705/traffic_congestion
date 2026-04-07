import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. LOAD THE DATA ──────────────────────────────────────────
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

# ── 2. FIRST LOOK ─────────────────────────────────────────────
print("Shape:", df.shape)          # how many rows and columns?
print("\nColumn names:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# ── 3. BASIC STATISTICS ───────────────────────────────────────
print("\nSummary statistics:\n", df.describe())
# ── 4. CONVERT DATE COLUMN ────────────────────────────────────
df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
df['hour']  = df['date_time'].dt.hour
df['day']   = df['date_time'].dt.dayofweek
df['month'] = df['date_time'].dt.month

# ── 5. PLOT: TRAFFIC BY HOUR ──────────────────────────────────
plt.figure(figsize=(10, 4))
df.groupby('hour')['traffic_volume'].mean().plot(kind='bar', color='steelblue')
plt.title('Average traffic volume by hour of day')
plt.xlabel('Hour')
plt.ylabel('Avg traffic volume')
plt.tight_layout()
plt.savefig('traffic_by_hour.png')
plt.show()

# ── 6. PLOT: TRAFFIC BY DAY ───────────────────────────────────
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
plt.figure(figsize=(8, 4))
df.groupby('day')['traffic_volume'].mean().plot(kind='bar', color='coral')
plt.xticks(range(7), days)
plt.title('Average traffic volume by day of week')
plt.xlabel('Day')
plt.ylabel('Avg traffic volume')
plt.tight_layout()
plt.savefig('traffic_by_day.png')
plt.show()