import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Force correct working directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "aqi_5_years.csv")

print("📂 Loading dataset from:", DATA_PATH)

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded")
print(df.head())

# --- Features & target ---
X = df[["PM2.5", "PM10", "NO2", "CO"]]
y = df["AQI"]  # Continuous → Regression

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model (CPU SAFE) ---
model = RandomForestRegressor(
    n_estimators=150,   # reduced (prevents freeze)
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("🚀 Training model...")
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 MAE: {mae:.2f}")
print(f"📈 R2 Score: {r2:.2f}")

# --- Save model ---
joblib.dump(model, "aqi_model.pkl")
print("✅ Model saved as aqi_model.pkl")
