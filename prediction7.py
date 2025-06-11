import os
import fastf1 #cargar datos oficiales de f1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split #para entrenar un modelo de predicción y evaluarlo.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

session_2024=fastf1.get_session(2024, "Emilia Romagna", "R")
session_2024.load()
#print(session_2024)

laps_2024=session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True) #Se eliminan los valores vacíos.

for col in ["LapTime", "Sector1Time","Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"]= laps_2024[col].dt.total_seconds()

#agregar tiempos de los corredores por sector
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

#pace en aire limpio
clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

#qualifying data de emilia romagna gp 2025
qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO",
            "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime (s)": [  
        74.704,  # VER
        74.962,  # NOR
        74.670,  # PIA
        74.807,  # RUS
        75.432,  # SAI
        75.473,  # ALB
        75.604,  # LEC
        76.613,  # OCO
        75.765,  # HAM
        75.581,  # STR
        75.787,  # GAS
        75.431,  # ALO
        76.518   # HUL
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

weather_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Emilia-Roma/2025-05-18?unitGroup=us&key=MEU6QNFEWXX73H7EZYFYWVWFQ&contentType=json&include=hours'
response = requests.get(weather_url)
weather_data = response.json()
#print (weather_data)

#Extract the weather relevant data for the race (sunday  4pm local time)
forecast_time= "2025-05-04 16:00:00"
forecast_data = {}
for hour_data in weather_data.get('days', [])[0].get('hours', []):
    if hour_data['datetime'] == "16:00:00":
        forecast_data = hour_data
        break
rain_probability = forecast_data.get("precipprob", 0)
temperature = forecast_data.get("temp", 20)

if rain_probability>=0.5:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] 


#constructors data
team_points = {
    "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37, "Ferrari": 94,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 8, "Alpine": 7
}

max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

# merge info
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]

X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=34)

# train gradient boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=34)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\nEmilia Romagna GP Winner GP 2025\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
