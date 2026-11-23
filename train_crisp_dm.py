import os
import json
import requests
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Evita bloqueos gráficos en consola
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("data", exist_ok=True)

print("==========================================================")
print("FASE 1: ENTENDIMIENTO DEL NEGOCIO")
print("==========================================================")
print("Objetivo: Predecir la temperatura máxima diaria (°C) a partir de")
print("variables meteorológicas básicas (temp. mínima, viento, radiación y lluvia).\n")

print("==========================================================")
print("FASE 2: ENTENDIMIENTO DE LOS DATOS (CONSUMO DE API REST)")
print("==========================================================")
url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=-15.49&longitude=-70.13"
    "&daily=temperature_2m_max,temperature_2m_min,wind_speed_10m_max,"
    "shortwave_radiation_sum,precipitation_sum"
    "&past_days=60&forecast_days=1&timezone=auto"
)

response = requests.get(url, timeout=15)
response.raise_for_status()
raw_json = response.json()

# Guardar respuesta cruda de la API para trazabilidad
with open("data/raw_open_meteo.json", "w", encoding="utf-8") as f:
    json.dump(raw_json, f, indent=2)

df = pd.DataFrame(raw_json["daily"])
print(f"Estado HTTP API Open-Meteo: {response.status_code} OK")
print(f"Registros descargados: {len(df)} días")
print(df.head(5), "\n")

print("==========================================================")
print("FASE 3: PREPARACIÓN DE LOS DATOS")
print("==========================================================")
df_clean = df.dropna().copy()
features = [
    "temperature_2m_min",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "precipitation_sum",
]
target = "temperature_2m_max"

df_clean.to_csv("data/dataset_clima_v1.csv", index=False)
print(f"Dataset limpio guardado en 'data/dataset_clima_v1.csv' ({len(df_clean)} filas válidas)")

X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"División de datos -> Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}\n")

print("==========================================================")
print("FASE 4: MODELADO (REGRESIÓN LINEAL)")
print("==========================================================")
model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "modelo_clima.joblib")
print("Modelo entrenado y exportado en 'modelo_clima.joblib'")
for col, coef in zip(features, model.coef_):
    print(f"  - Peso {col}: {coef:.4f}")
print()

print("==========================================================")
print("FASE 5: EVALUACIÓN DEL MODELO")
print("==========================================================")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Absoluto Medio (MAE): {mae:.2f} °C")
print(f"Coeficiente de Determinación (R²): {r2:.4f}\n")

# Generar gráfica para la presentación
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color="#2563EB", alpha=0.8, label="Predicciones")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Predicción Ideal")
plt.title(f"Evaluación CRISP-DM: Temperatura Real vs. Predicha\n(MAE = {mae:.2f} °C | R² = {r2:.2f})")
plt.xlabel("Temperatura Máxima Real (°C)")
plt.ylabel("Temperatura Máxima Predicha (°C)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("evaluacion_modelo.png", dpi=150)
print("Gráfica guardada en 'evaluacion_modelo.png'")
print("Listo para FASE 6 (Despliegue con FastAPI).")