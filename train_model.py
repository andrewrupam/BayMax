# Step 1: Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # for saving model

# Step 2: Generate synthetic data
np.random.seed(42)
n_samples = 20000  # increased samples for better training

# Features
age = np.random.randint(18, 66, n_samples)
weight = np.random.randint(40, 150, n_samples)
height = np.random.randint(150, 201, n_samples)
gender = np.random.choice([0, 1], n_samples)  # 0: Male, 1: Female
exercise_hours = np.random.randint(0, 16, n_samples)
smoking = np.random.choice([0, 1], n_samples)
drinking = np.random.choice([0, 1], n_samples)
screen_time = np.random.randint(0, 13, n_samples)
sleep_hours = np.random.randint(4, 11, n_samples)

# Step 3: Calculate BMI
bmi = weight / ((height / 100) ** 2)

# Step 4: Calculate health risk score with nonlinear emphasis
# Normal BMI: 18.5–24.9 → minimal impact
# Underweight: <18.5 → risk increases
# Overweight/Obese: >24.9 → risk increases significantly
bmi_risk = np.where(
    bmi < 18.5, 
    (18.5 - bmi) * 2, 
    np.where(
        bmi <= 24.9, 
        0, 
        (bmi - 24.9) ** 1.5  # sharper increase for overweight
    )
)

# More emphasis on smoking and drinking
risk_score = (
    0.15*age + 
    0.5*bmi_risk + 
    10*smoking + 
    7*drinking - 
    2*exercise_hours + 
    0.5*screen_time - 
    1.5*sleep_hours
)

# Step 5: Rescale to 1–100
min_score, max_score = risk_score.min(), risk_score.max()
risk_score_scaled = 1 + 99 * (risk_score - min_score) / (max_score - min_score)

# Step 6: Create DataFrame
data = pd.DataFrame({
    "age": age,
    "weight": weight,
    "height": height,
    "gender": gender,
    "exercise_hours": exercise_hours,
    "smoking": smoking,
    "drinking": drinking,
    "screen_time": screen_time,
    "sleep_hours": sleep_hours,
    "health_risk_score": risk_score_scaled
})

# Optional: save dataset
data.to_csv("synthetic_health_data.csv", index=False)

# Step 7: Split into train/test
X = data.drop("health_risk_score", axis=1)
y = data["health_risk_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)  # deeper + more trees
model.fit(X_train, y_train)

# Step 9: Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")

# Step 10: Save the trained model
joblib.dump(model, "health_risk_model.pkl")
print("Model saved as health_risk_model.pkl")
