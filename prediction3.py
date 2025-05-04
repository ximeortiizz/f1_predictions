#modelo de machine learning grading boosting (analisis de regresion)
import os
import fastf1 #cargar datos oficiales de f1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split #para entrenar un modelo de predicción y evaluarlo.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, "Japan", "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy() #Se extraen los tiempos de vuelta (LapTime) por piloto.
laps_2024.dropna(inplace=True) #Se eliminan los valores vacíos.

for col in ["LapTime", "Sector1Time","Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"]= laps_2024[col].dt.total_seconds()

#Group bu driver to get average sector times per driver
sector_times_2024=laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

#Qualifying Data- Chinese GrandPrix Data

#Se crea un DataFrame con nombres de pilotos y sus tiempos de clasificación 2025 en segundos.   
qualifying_2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen", "Lewis Hamilton",
               "Charles Leclerc", "Isack Hadjar", "Andrea Kimi Antonelli", "Yuki Tsunoda", "Alexander Albon",
               "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll", "Carlos Sainz Jr.",
               "Pierre Gasly", "Oliver Bearman", "Jack Doohan", "Gabriel Bortoleto", "Liam Lawson"],
    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927,
                           91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840,
                           91.992, 92.018, 92.092, 92.141, 92.174]
})

# Map full names to FastF1 3-letter codes
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Andrea Kimi Antonelli": "ANT",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(
    sector_times_2024, left_on="DriverCode", right_on="Driver", how="left"
).rename(columns={"Driver_x": "Driver", "Driver_y": "DriverCode"})

#DriverSpecific Wet Performance based on the Canadian GP 2022 and 2023
driver_wet_performance={
    "Max Verstappen": 0.975196,
    "Lewis Hamilton": 0.976464,
    "Charles Leclerc": 0.975862,
    "Lando Norris":0.978179, 
    "Fernando Alonso": 0.972655,
    "George Russell":0.968678,
    "Carlos Sainz Jr.":  0.978754,
    "Yuki Tsunoda": 0.996338,
    "Esteban Ocon": 0.981810,
    "Alexander Albon": 0.978120,
    "Pierre Gasly": 0.978832,
    "Lance Stroll": 0.979857
}

merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(driver_wet_performance)
merged_data["WetPerformanceFactor"] = merged_data["WetPerformanceFactor"].fillna(1.0)


weather_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Suzuka/2025-03-30?unitGroup=us&key=MEU6QNFEWXX73H7EZYFYWVWFQ&contentType=json&include=hours'
response = requests.get(weather_url)
weather_data = response.json()

#Extract the weather relevant data for the race (sunday at 2pm local time)
forecast_time= "2025-03-30 14:00:00"
forecast_data = {}
for hour_data in weather_data.get('days', [])[0].get('hours', []):
    if hour_data['datetime'] == "14:00:00":
        forecast_data = hour_data
        break
rain_probability = forecast_data.get("precipprob", 0)
temperature = forecast_data.get("temp", 20)

#print(forecast_data)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

merged_data["WetPerformanceFactor"] = merged_data["Driver"].map(driver_wet_performance).fillna(1.0)

# Preparar datos para modelo
X = merged_data[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor", "RainProbability", "Temperature"]]
y = merged_data["QualifyingTime (s)"]

X = X.dropna()
y = y[X.index]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times using 2025 qualifying and sector data
predicted_race_times = model.predict(X)

# Verificar qué pilotos faltan en sector_times_2024
missing_drivers = qualifying_2025[~qualifying_2025["DriverCode"].isin(sector_times_2024["Driver"])]
print(f"Missing pilots: {missing_drivers['Driver'].tolist()}")

# Solo asignar predicciones a los pilotos con datos completos
valid_drivers = merged_data.dropna(subset=["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"])
predicted_race_times = model.predict(valid_drivers[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor", "RainProbability", "Temperature"]])

merged_data.loc[valid_drivers.index, "PredictedRaceTime (s)"] = predicted_race_times

merged_data = merged_data.dropna(subset=["PredictedRaceTime (s)"])

merged_data = merged_data.sort_values(by="PredictedRaceTime (s)")


print("\n🏁 Predicted 2025 Chinese GP Winner with OLD Drivers and Sector Times 🏁\n")
print(merged_data[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
