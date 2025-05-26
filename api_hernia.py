from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Carga del modelo previamente guardado
model = joblib.load("mejor_modelo_hernia.pkl") 

# Define el esquema de entrada
class EntradaModelo(BaseModel):
    pelvic_incidence: float
    pelvic_tilt: float
    lumbar_lordosis_angle: float
    sacral_slope: float
    pelvic_radius: float
    degree_spondylolisthesis: float

@app.get("/")
def leer_root():
    return {"mensaje": "API activa y corriendo"}

@app.post("/predict")
def predict(data: EntradaModelo):
    # Validación de rangos fisiológicos
    if any(v < 0 for v in data.dict().values()):
        return {"error": "Valores fuera de rango fisiológico"}

    input_data = np.array([[  
        data.pelvic_incidence,
        data.pelvic_tilt,
        data.lumbar_lordosis_angle,
        data.sacral_slope,
        data.pelvic_radius,
        data.degree_spondylolisthesis
    ]])

    # Predicción y cálculo de probabilidad
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probabilidad de hernia

    # Registro de entrada y salida en CSV
    df = pd.DataFrame([dict(data)], columns=data.dict().keys())
    df["prediccion"] = prediction[0]
    df["probabilidad"] = probability
    df.to_csv("registro_predicciones.csv", mode="a", header=False, index=False)

    return {"prediccion": int(prediction[0]), "probabilidad": probability}
