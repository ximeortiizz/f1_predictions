import requests
import json

# URL de la API de Visual Crossing
url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Suzuka?unitGroup=us&key=MEU6QNFEWXX73H7EZYFYWVWFQ&contentType=json"

# Realizar la solicitud GET
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    weather_data = response.json()

    # Mostrar el JSON de forma bonita para revisar la estructura
    print("Estructura de weather_data:\n")
    print(json.dumps(weather_data, indent=2))

    # Supongamos que la clave relevante es 'days' que contiene la información de pronóstico por hora.
    # Ahora puedes acceder a la lista de pronósticos por cada día.
    if 'days' in weather_data:
        # Acceder a los datos del primer día
        for day in weather_data['days']:
            print(f"Fecha: {day['datetime']}")
            print(f"Temperatura: {day['temp']}°C")
            print(f"Condiciones: {day['conditions']}")
            print(f"Probabilidad de precipitación: {day['precipprob']}%")
            print("-" * 40)
    else:
        print("No se encontraron datos de pronóstico en la respuesta.")
else:
    print("Error al obtener los datos del clima. Código de estado:", response.status_code)
