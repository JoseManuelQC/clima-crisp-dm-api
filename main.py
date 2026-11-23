import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

if not os.path.exists("modelo_clima.joblib"):
    raise RuntimeError("Ejecuta primero 'python train_crisp_dm.py' para generar el modelo.")

modelo = joblib.load("modelo_clima.joblib")

app = FastAPI(
    title="API de Predicción Climática (CRISP-DM)",
    version="1.0.0",
    description="Despliegue de modelo de Regresión Lineal entrenado con datos de Open-Meteo API.",
)

class ClimaInput(BaseModel):
    temperature_2m_min: float = Field(..., json_schema_extra={"example": 4.5}, description="Temperatura mínima del día (°C)")
    wind_speed_10m_max: float = Field(..., ge=0, json_schema_extra={"example": 18.2}, description="Velocidad máxima del viento (km/h)")
    shortwave_radiation_sum: float = Field(..., ge=0, json_schema_extra={"example": 22.4}, description="Radiación solar acumulada (MJ/m²)")
    precipitation_sum: float = Field(..., ge=0, json_schema_extra={"example": 0.0}, description="Precipitación diaria (mm)")

class ClimaPrediction(BaseModel):
    temperatura_max_predicha_c: float
    unidad: str
    algoritmo: str
    metodologia: str

@app.post("/predict", response_model=ClimaPrediction, tags=["Fase 6: Despliegue"])
def predecir_temperatura(datos: ClimaInput):
    df_entrada = pd.DataFrame([datos.model_dump()])
    prediccion = float(modelo.predict(df_entrada)[0])

    return ClimaPrediction(
        temperatura_max_predicha_c=round(prediccion, 2),
        unidad="°C",
        algoritmo="LinearRegression (Scikit-Learn)",
        metodologia="CRISP-DM"
    )