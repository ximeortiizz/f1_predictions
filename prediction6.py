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
session_2024 = fastf1.get_session(2024, "Miami", "R")
session_2024.load()

# Extract lap times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy() #Se extraen los tiempos de vuelta (LapTime) por piloto.
laps_2024.dropna(inplace=True) #Se eliminan los valores vacíos.

for col in ["LapTime", "Sector1Time","Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"]= laps_2024[col].dt.total_seconds()


"""#Group bu driver to get average sector times per driver
sector_times_2024=laps_2024.groupby("Driver").agg({"Sector1Time (s)": "mean", "Sector2Time (s)": "mean", "Sector3Time (s)": "mean"}).reset_index()

sector_times_2024["TotalSectorTime (s)"]= (
    sector_times_2024["Sector1Time (s)"]+
    sector_times_2024["Sector2Time (s)"]+
    sector_times_2024["Sector3Time (s)"]
)"""

#Se crea un DataFrame con nombres de pilotos y sus tiempos de clasificación 2025 en segundos.   
qualifying_2025 = pd.DataFrame({
    "Driver": ["Max Verstappen", "Lando Norris","Andrea Kimi Antonelli", "Oscar Piastri","George Russell",
                "Carlos Sainz Jr.",  "Alexander Albon", "Charles Leclerc",  "Esteban Ocon", "Yuki Tsunoda",
                "Isack Hadjar", "Lewis Hamilton", "Gabriel Bortoleto", "Jack Doohan", "Liam Lawson",
                "Nico Hülkenberg", "Fernando Alonso", "Pierre Gasly",  "Lance Stroll",  "Oliver Bearman"],
    "QualifyingTime (s)": [86.204, 86.269, 86.271, 86.375, 86.385,
                           86.569, 86.682, 86.754, 86.824, 86.943,
                           86.987, 87.006, 87.151, 87.186, 87.363,
                           87.473, 87.604, 87.710, 87.830, 87.999]
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

"""# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(
    sector_times_2024, left_on="DriverCode", right_on="Driver", how="left"
).rename(columns={"Driver_x": "Driver", "Driver_y": "DriverCode"})"""

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
forecast_time= "2025-05-04 16:00:00"
forecast_data = {}
for hour_data in weather_data.get('days', [])[0].get('hours', []):
    if hour_data['datetime'] == "16:00:00":
        forecast_data = hour_data
        break
rain_probability = forecast_data.get("precipprob", 0)
temperature = forecast_data.get("temp", 20)

if rain_probability>=0.75:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"]=qualifying_2025["QualifyingTime (s)"] 

team_points= {
    "McLaren":203, "Mercedes":118, "Red Bull":92,  "Ferrari":84,"Williams":25, 
    "Hass":20, "Aston Martin":14, "Racing Bulls": 8,"Alpine":7,"Kick Sauber":6,  
}

max_points= max(team_points.values())
team_performance_score={team:points/max_points for team,points in team_points.items()}

driver_to_team={
    "VER": "Red Bull", "HAM": "Ferrari","RUS": "Mercedes","LEC": "Ferrari","SAI": "Williams", 
    "NOR": "McLaren", "PIA": "McLaren", "ALO": "Aston Martin","STR": "Aston Martin","OCO": "Haas",
    "GAS": "Alpine","ALB": "Williams", "HUL": "Kick Sauber", "TSU": "Red Bull", "ANT": "Mercedes", 
    "BEA": "Haas", "HAD": "Racing Bulls", "LAW": "Racing Bulls", "DOO": "Alpine", "BOR": "Kick Sauber"
}

qualifying_2025["Team"]=qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"]=qualifying_2025["Team"].map(team_performance_score)

average_2025={
    "VER": 93.191067,
    "PIA": 93.232111,
    "LEC": 93.418667,
    "NOR": 93.428600,
    "RUS": 93.833378,
    "ANT": 93.909844,
    "HAM": 94.020622,
    "SAI": 94.497444,
    "ALB": 94.566600,
    "LAW": 94.630733,
    "HAD": 94.656444,
    "ALO": 94.784333,
    "BEA": 94.817889,
    "STR": 95.318250,
    "HUL": 95.345455,
    "DOO": 95.573659,
    "OCO": 95.682128,
    "BOR": 96.003043
}

qualifying_2025["Average2025Performance"] = qualifying_2025["DriverCode"].map(average_2025)

#print(forecast_data)

qualifying_2025["RainProbability"] = rain_probability
qualifying_2025["Temperature"] = temperature
qualifying_2025["LastYearWinner"] = (qualifying_2025["DriverCode"] == "VER").astype(int)
qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime"] ** 2


# Tiempo de carrera real promedio
lap_time_by_drivercode = laps_2024.groupby("Driver")["LapTime (s)"].mean()
qualifying_2025["LapTime (s)"] = qualifying_2025["DriverCode"].map(lap_time_by_drivercode)

#features
X=qualifying_2025[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", "Average2025Performance","LastYearWinner"
]].fillna(0)



clean_data = qualifying_2025.dropna(subset=["LapTime (s)"]).copy()
#print(f"Longitud de clean_data después de dropna: {len(clean_data)}")


X= clean_data[["QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore","Average2025Performance", "LastYearWinner"]].fillna(0)
y = clean_data["LapTime (s)"]
print("Número de muestras en X:", len(X))
print("Número de muestras en y:", len(y))
print("Primeras filas de y:\n", y.head())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

predicted_race_times = model.predict(X)

#Prediction
clean_data["PredictedRaceTime (s)"] = model.predict(X)
final_results = clean_data.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)

"""# Verificar qué pilotos faltan en sector_times_2024
missing_drivers = qualifying_2025[~qualifying_2025["DriverCode"].isin(sector_times_2024["Driver"])]
print(f"Missing pilots: {missing_drivers['Driver'].tolist()}")

# Solo asignar predicciones a los pilotos con datos completos
valid_drivers = final_results.dropna(subset=["TotalSectorTime (s)"])
predicted_race_times = model.predict(valid_drivers[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)", "WetPerformanceFactor", "RainProbability", "Temperature"]])

final_results.loc[valid_drivers.index, "PredictedRaceTime (s)"] = predicted_race_times

final_results = final_results.dropna(subset=["PredictedRaceTime (s)"])

final_results = final_results.sort_values(by="PredictedRaceTime (s)")"""


print("\n🏁 Predicción de resultados para el GP de Miami 2025 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
print(f"\n🔍 Error del modelo (MAE): {mean_absolute_error(y_test, y_pred):.2f} segundos")
