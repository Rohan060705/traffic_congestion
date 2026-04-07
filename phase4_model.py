import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns

# ── 1. LOAD CLEAN DATA ────────────────────────────────────────
df = pd.read_csv('clean_traffic.csv')
print("Dataset shape:", df.shape)

# ── 2. DEFINE FEATURES AND TARGET ─────────────────────────────
# X = input features the model will learn from
# y = what we want to predict (congestion level)
feature_cols = [
    'hour', 'day', 'month', 'is_weekend', 'is_rush_hour',
    'is_holiday', 'temp_celsius', 'rain_1h', 'snow_1h',
    'clouds_all', 'weather_code'
]
X = df[feature_cols]
y = df['congestion_code']   # 0=Low, 1=Medium, 2=High

print("Features:", feature_cols)
print("Target distribution:\n", y.value_counts())

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────
# 80% of data goes to training, 20% reserved for testing
# random_state=42 ensures same split every time you run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 4. TRAIN RANDOM FOREST ────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    max_depth=10,        # each tree can be max 10 levels deep
    random_state=42
)
rf_model.fit(X_train, y_train)   # this is where learning happens!
rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_f1  = f1_score(y_test, rf_preds, average='weighted')
print(f"Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print(f"Random Forest F1 Score: {rf_f1:.4f}")

# ── 5. TRAIN XGBOOST ──────────────────────────────────────────
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,   # how fast the model learns
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1  = f1_score(y_test, xgb_preds, average='weighted')
print(f"XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"XGBoost F1 Score: {xgb_f1:.4f}")

# ── 6. COMPARE MODELS ─────────────────────────────────────────
print("\n--- Model Comparison ---")
print(f"{'Model':<20} {'Accuracy':>10} {'F1 Score':>10}")
print(f"{'Random Forest':<20} {rf_acc*100:>9.2f}% {rf_f1:>10.4f}")
print(f"{'XGBoost':<20} {xgb_acc*100:>9.2f}% {xgb_f1:>10.4f}")

# ── 7. DETAILED REPORT FOR BEST MODEL ─────────────────────────
best_preds = xgb_preds if xgb_acc > rf_acc else rf_preds
best_name  = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
print(f"\nDetailed report for best model ({best_name}):")
print(classification_report(y_test, best_preds,
      target_names=['Low', 'Medium', 'High']))

# ── 8. CONFUSION MATRIX PLOT ──────────────────────────────────
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low','Medium','High'],
            yticklabels=['Low','Medium','High'])
plt.title(f'Confusion matrix — {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# ── 9. FEATURE IMPORTANCE PLOT ────────────────────────────────
importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
importance.sort_values().plot(kind='barh', color='steelblue', figsize=(8, 5))
plt.title('Feature importance — Random Forest')
plt.xlabel('Importance score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# ── 10. SAVE THE BEST MODEL ───────────────────────────────────
best_model = xgb_model if xgb_acc > rf_acc else rf_model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
print(f"\nbest_model.pkl saved! ({best_name})")