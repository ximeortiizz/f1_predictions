#modelo de machine learning grading boosting (analisis de regresion)

import fastf1 #cargar datos oficiales de f1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #para entrenar un modelo de predicción y evaluarlo.
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
#Crear un caché local donde FastF1 guarda los datos descargados, para no tener que bajarlos de nuevo cada vez.
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2024 Australian GP race session
#cargar los datos reales de carrera: 2024, numero de carrera (3(australia)), R (sesion de carrera)
session_2024 = fastf1.get_session(2024, 3, "R")
session_2024.load()

# Extract lap times
#Obtener tiempos de vuelta
laps_2024 = session_2024.laps[["Driver", "LapTime"]].copy() #Se extraen los tiempos de vuelta (LapTime) por piloto.
laps_2024.dropna(subset=["LapTime"], inplace=True) #Se eliminan los valores vacíos.
laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds() #Convierte los tiempos a segundos (para análisis numérico).

# 2025 Qualifying Data
#Se crea un DataFrame con nombres de pilotos y sus tiempos de clasificación 2025 en segundos.   
qualifying_2025 = pd.DataFrame({
    "Driver": ["Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
            "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz", "Fernando Alonso", "Lance Stroll"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670,
                        75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5] #se saca de la base de datos f1
})

# Map full names to FastF1 3-letter codes 
# Mapeo de nombres completos a códigos de piloto FastF1
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER", "George Russell": "RUS",
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM",
    "Pierre Gasly": "GAS", "Carlos Sainz": "SAI", "Lance Stroll": "STR", "Fernando Alonso": "ALO"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge 2025 Qualifying Data with 2024 Race Data 
#Combina (merge) los datos de clasificación 2025 con los tiempos reales de carrera 2024, usando los códigos de piloto como referencia.
merged_data = qualifying_2025.merge(laps_2024, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]] #X: las características (inputs) — aquí, solo el tiempo de clasificación
y = merged_data["LapTime (s)"] #y: la variable a predecir — tiempo de vuelta en carrera.

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model 
# Divide los datos en entrenamiento (80%) y prueba (20%).
#Usa un modelo llamado GradientBoostingRegressor (bueno para predicción de valores continuos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 qualifying times
predicted_lap_times = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions 
# Ordena de más rápido a más lento para simular el resultado de carrera.
print("\n🏁 Predicted 2025 Chinese GP Winner 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
#Calcula el error medio absoluto (MAE) para ver qué tan bien está funcionando el modelo.
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")
