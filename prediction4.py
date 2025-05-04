import os
import fastf1 #cargar datos oficiales de f1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split #para entrenar un modelo de predicción y evaluarlo.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib as plt


# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Australian GP race session
session_2024 = fastf1.get_session(2024, "Saudi Arabia", "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy() #Se extraen los tiempos de vuelta (LapTime) por piloto.
laps_2024.dropna(inplace=True) #Se eliminan los valores vacíos.

for col in ["LapTime", "Sector1Time","Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"]= laps_2024[col].dt.total_seconds()

#Group bu driver to get average sector times per driver
sector_times_2024=laps_2024.groupby("Driver").agg({"Sector1Time (s)": "mean", "Sector2Time (s)": "mean", "Sector3Time (s)": "mean"}).reset_index()

sector_times_2024["TotalSectorTime (s)"]= (
    sector_times_2024["Sector1Time (s)"]+
    sector_times_2024["Sector2Time (s)"]+
    sector_times_2024["Sector3Time (s)"]
)

qualifying_2025 = pd.DataFrame({
    "Driver": ["VER","PIA", "LEC", "RUS", "SAI", "HAM", "TSU", "GAS","NOR", "ALB", "ALO", "STR",  
               "HUL", "OCO"],
    "QualifyingTime (s)": [87.291,87.294, 87.304, 87.670, 87.407, 88.164,
                           88.201,88.367,88.109,88.191,88.303, 88.418,
                           88.782,89.092]

})

average_2025={
    "VER":88.0, "PIA": 89.1, "LEC": 89.2, "RUS": 89.3, "HAM": 89.4,
    "GAS": 89.5, "ALO":89.6, "TSU": 89.2, "SAI":89.8, "HUL":89.9,
    "OCO": 90.0, "STR": 90.1, "NOR":90.2
}


#DriverSpecific Wet Performance based on the Canadian GP 2022 and 2023
driver_wet_performance={
    "VER": 0.975196,
    "HAM": 0.976464,
    "LEC": 0.975862,
    "NOR":0.978179, 
    "ALO": 0.972655,
    "RUS":0.968678,
    "SAI":  0.978754,
    "TSU": 0.996338,
    "OCO": 0.981810,
    "ALB": 0.978120,
    "GAS": 0.978832,
    "STR": 0.979857
}

qualifying_2025["WetPerformanceFactor"] = qualifying_2025["Driver"].map(driver_wet_performance)
qualifying_2025["WetPerformanceFactor"] = qualifying_2025["WetPerformanceFactor"].fillna(1.0)

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

if rain_probability>=0.75:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] 

team_points= {
    "McLaren":70, "Mercedes":53, "Red Bull":36, "Williams":17, "Ferrari":17,
    "Hass":14, "Aston Martin":10, "Kick Sauber":6, "Racing Bulls": 3, "Alpine":0
}
max_points= max(team_points.values())
team_performance_score={team:points/max_points for team,points in team_points.items()}

driver_to_team={
    "VER": "Red Bull", "HAM": "Mercedes","RUS": "Mercedes","LEC": "Ferrari",
    "SAI": "Ferrari", "NOR": "McLaren", "PIA": "McLaren", "ALO": "Aston Martin",
    "STR": "Aston Martin","OCO": "Alpine","GAS": "Alpine","ALB": "Williams",
    "HUL": "Kick Sauber", "TSU": "Racing Bulls"
}

qualifying_2025["Team"]=qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"]=qualifying_2025["Team"].map(team_performance_score)
qualifying_2025["Average2025Performance"]= qualifying_2025["Driver"].map(average_2025)

merged_data= qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")

#print(forecast_data)
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature

last_year_winner= "VER"
merged_data["LastYearWinner"]=(merged_data["Driver"]==last_year_winner).astype(int)

merged_data["QualifyingTime"]= merged_data["QualifyingTime"]**2

#features
X=merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "TotalSectorTime (s)", "Average2025Performance","LastYearWinner"
]].fillna(0)

y=laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

print("Contenido de merged_data:")
print(merged_data)
print(f"Longitud de merged_data: {len(merged_data)}")

clean_data=merged_data.copy()
clean_data["LapTime (s)"]=y.values
clean_data=clean_data.dropna(subset=["LapTime (s)"])
#print(f"Longitud de clean_data después de dropna: {len(clean_data)}")


X= clean_data[["QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "TotalSectorTime (s)", "Average2025Performance", "LastYearWinner"]].fillna(0)
y = clean_data["LapTime (s)"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

predicted_race_times = model.predict(X)

clean_data["PredictedRaceTime (s)"] = predicted_race_times

final_results = clean_data.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

print("\n🏁 Predicción de resultados para el GP de Saudi Arabia 2025 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
print(f"\n🔍 Error del modelo (MAE): {mean_absolute_error(y_test, y_pred):.2f} segundos")
