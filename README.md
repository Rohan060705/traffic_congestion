# 🚦 Traffic Congestion / Bottleneck Predictor

A machine learning project that predicts traffic congestion levels (Low, Medium, High) based on time, weather, and road conditions — built with Python, XGBoost, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.14-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-90.58%25_accuracy-green) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)

---

## Live demo

Run locally using the steps in the [Getting Started](#getting-started) section below.

---

## What it does

- Predicts whether traffic at a given hour will be **Low**, **Medium**, or **High** congestion
- Takes inputs like hour of day, day of week, temperature, rainfall, and weather condition
- Automatically detects rush hours and weekends from your inputs
- Shows a live hourly traffic chart with your selected hour highlighted
- Powered by a trained XGBoost model with **90.58% accuracy**

---

## Screenshots

| Prediction dashboard | Feature importance |
|---|---|
| Input parameters on sidebar | Hour of day drives 72% of predictions |
| Live congestion prediction | Rush hour flag is second most important |

---

## Project structure

```
traffic_congestion/
│
├── dashboard.py                        # Streamlit web app
├── phase2_explore.py                   # Data exploration
├── phase3_clean.py                     # Data cleaning & preprocessing
├── phase4_model.py                     # Model training & evaluation
│
├── best_model.pkl                      # Trained XGBoost model
├── feature_cols.pkl                    # Saved feature list
├── clean_traffic.csv                   # Preprocessed dataset
├── Metro_Interstate_Traffic_Volume.csv # Raw dataset (download separately)
│
├── traffic_by_hour.png                 # EDA chart
├── traffic_by_day.png                  # EDA chart
├── confusion_matrix.png                # Model evaluation
├── feature_importance.png              # Feature importance chart
└── congestion_distribution.png        # Class distribution chart
```

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/Rohan060705/traffic_congestion.git
cd traffic_congestion
```

### 2. Install dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib
```

### 3. Download the dataset

Download the Metro Interstate Traffic Volume dataset from Kaggle and place it in the project folder:

👉 https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume

### 4. Run the pipeline

```bash
# Step 1 — Explore the data
python phase2_explore.py

# Step 2 — Clean and preprocess
python phase3_clean.py

# Step 3 — Train the models
python phase4_model.py

# Step 4 — Launch the dashboard
streamlit run dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

---

## Dataset

**Metro Interstate Traffic Volume** — hourly traffic data collected from a US interstate highway.

| Feature | Description |
|---|---|
| `date_time` | Timestamp of the record |
| `traffic_volume` | Number of vehicles per hour (target) |
| `temp` | Temperature in Kelvin |
| `rain_1h` | Rainfall in mm |
| `snow_1h` | Snowfall in mm |
| `clouds_all` | Cloud cover percentage |
| `weather_main` | Weather condition (Clear, Rain, Snow etc.) |
| `holiday` | Whether it was a public holiday |

Total records: **48,204** hourly readings

---

## ML pipeline

```
Raw CSV → Clean & preprocess → Feature engineering → Train models → Evaluate → Deploy dashboard
```

### Features used for prediction

| Feature | Importance |
|---|---|
| `hour` | 72% — most dominant predictor |
| `is_rush_hour` | 10% — engineered feature |
| `day` | 6% |
| `is_weekend` | 4% |
| `temp_celsius` | 4% |
| `clouds_all`, `weather_code`, `rain_1h` | < 2% each |

### Model comparison

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest | 88.81% | 0.8886 |
| **XGBoost** | **90.58%** | **0.9055** |

XGBoost was selected as the final model and saved as `best_model.pkl`.

### Per-class performance (XGBoost)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low | 0.96 | 0.95 | 0.95 |
| Medium | 0.88 | 0.84 | 0.86 |
| High | 0.88 | 0.93 | 0.91 |

---

## Key findings

- Hour of day alone explains **72% of congestion prediction** — rush hours at 7–9am and 4–6pm are the dominant pattern
- Weekend traffic is approximately **35% lower** than weekday traffic
- The dataset is well balanced — roughly 16,000 records per congestion class
- XGBoost outperforms Random Forest by **1.77%** accuracy on this dataset

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| Pandas | Data loading and manipulation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Train/test split, evaluation metrics |
| XGBoost | Final prediction model |
| Joblib | Model serialization |
| Streamlit | Interactive web dashboard |

---

## Author

**Rohan** — [@Rohan060705](https://github.com/Rohan060705)

---

## License

This project is open source and available under the [MIT License](LICENSE).
