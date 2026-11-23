# Predicción de Temperatura Máxima con Open-Meteo API y CRISP-DM

Proyecto de ciencia de datos y despliegue de microservicio desarrollado bajo la metodología **CRISP-DM**, consumiendo datos meteorológicos reales desde la API pública **Open-Meteo**.

## Fases CRISP-DM Implementadas
1. **Entendimiento del Negocio:** Estimación de temperatura máxima diaria (°C) a partir de variables atmosféricas.
2. **Entendimiento de los Datos:** Consumo directo de API REST (`Open-Meteo`) en formato JSON (`data/raw_open_meteo.json`).
3. **Preparación de los Datos:** Limpieza de registros y generación del dataset versionado (`data/dataset_clima_v1.csv`).
4. **Modelado:** Entrenamiento de Regresión Lineal con `scikit-learn` (`modelo_clima.joblib`).
5. **Evaluación:** Validación mediante MAE y $R^2$ con gráfica comparativa (`evaluacion_modelo.png`).
6. **Despliegue:** Exposición del modelo mediante **FastAPI** y documentación interactiva **Swagger UI**.

## Replicabilidad e Instalación

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el pipeline CRISP-DM (Descarga API, entrenamiento y evaluación):**
   ```bash
   python train_crisp_dm.pys
   ```

3. **Levantar la API de predicción:**
   ```bash
   uvicorn main:app --reload
   ```
   Acceder a Swagger UI en: `http://127.0.0.1:8000/docs`