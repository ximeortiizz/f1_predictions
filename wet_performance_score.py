import fastf1
import pandas as pd

fastf1.Cache.enable_cache("f1_cache")

#Cargar la sesion del GP de canada (wet race)
session_2023=fastf1.get_session(2023, "Canada", "R")
session_2023.load()
#print(session_2023)

#Cargar la sesion del GP de canada del 2022 (dry race)
session_2022=fastf1.get_session(2022, "Canada", "R")
session_2022.load()

#Extraer los lap times de ambas carreras
laps_2023 = session_2023.laps[["Driver", "LapTime"]].copy()
laps_2022 = session_2022.laps[["Driver", "LapTime"]].copy()

#Mostrar valores nulos en caso de que haya missing laps
laps_2023.dropna(inplace=True)
laps_2022.dropna(inplace=True)

#Convertir los laptimes a segundos
laps_2023["LapTime (s)"]=laps_2023["LapTime"].dt.total_seconds()
laps_2022["LapTime (s)"]=laps_2022["LapTime"].dt.total_seconds()

#calcular el average lap time para cada uno de los pilotos para ambas carreras
avg_lap_2023=laps_2023.groupby("Driver")["LapTime (s)"].mean().reset_index()
avg_lap_2022=laps_2022.groupby("Driver")["LapTime (s)"].mean().reset_index()

#merge los datos de ambas carreras en la columna de driver
merged_data = pd.merge(avg_lap_2023, avg_lap_2022, on="Driver", suffixes=('_2023','_2022'))

#calcular el performance change en cada time lap entre 2022 y 2023
merged_data["LapTimeDiff (s)"] = merged_data["LapTime (s)_2023"] - merged_data["LapTime (s)_2022"]

#calcular el porcentaje de cambio en el laptime  entre wet y dry conditions
merged_data["Performance change (%)"] = (merged_data["LapTimeDiff (s)"] / merged_data["LapTime (s)_2022"]) * 100

#Wet performance score
merged_data["WetPerformanceScore"] = 1 + (merged_data["Performance change (%)"] / 100)

print("Driver Wet Performance Scores (2023 vs 2022):")
print (merged_data[["Driver", "WetPerformanceScore"]])
