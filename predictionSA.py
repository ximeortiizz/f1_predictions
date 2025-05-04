import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

fastf1.Cache.enable_cache("f1_cache")

session_2023= fastf1.get_session(2023, "Saudi Arabia", "R")
session_2023.load()
#print(session_2024)

laps_2023=session_2023.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2023.dropna(inplace=True)

for col in ["LapTime","Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2023[f"{col} (s)"] = laps_2023[col].dt.total_seconds()

sector_times_2023= laps_2023.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
qualifying_2024 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Charles Leclerc", "Sergio Perez", "Fernando Alonso", "Oscar Piastri",
               "Lando Norris", "George Russell", "Lewis Hamilton", "Yuki Tsunoda", "Lance Stroll",
               "Oliver Bearman", "Alexander Albon", "Kevin Magnussen", "Daniel Riccardo", "Nico Hulkenberg",
               "Valtteri Bottas", "Esteban Ocon", "Pierre Gasly", "Logan Sargeant", "Zhou Guanyu"],
    "QualifyingTime (s)": [87.472, 87.791, 87.807, 87.846,88.089,
                        88.132, 88.316,88.460,88.547,88.572,
                        88.642,88.980,89.020,89.025, 89.055,
                        89.179,89.475,89.479,89.526,0]
})

driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Sergio Perez": "PER",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA",  "Kevin Magnussen":"MAG", "Daniel Riccardo": "RIC","Valtteri Bottas":"BOT","Logan Sargeant": "SAR",
    "Zhou Guanyu":"ZHO"
}

qualifying_2024["DriverCode"] = qualifying_2024["Driver"].map(driver_mapping)

merged_data = qualifying_2024.merge(sector_times_2023, left_on="DriverCode", right_on="Driver", how="left")

# Define feature set (Qualifying + Sector Times)
X = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)
qualifying_2024["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2024 = qualifying_2024.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("Predicted 2025 Chinese GP Winner with New Drivers and Sector Times 🏁\n")
print(qualifying_2024[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")


