# Importamos las bibliotecas necesarias
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Definir un modelo Pydantic para recibir los datos
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Cargar el modelo una vez al inicio para evitar cargarlo en cada solicitud
loaded_model = joblib.load('iris_model.joblib')

# Definir un endpoint para manejar la predicción
@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # Convertir los datos de entrada a un array de numpy
        input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])

        # Realizar la predicción
        prediction = loaded_model.predict(input_data)

        # Convertir el resultado de la predicción a un tipo nativo de Python
        prediction = prediction.item()

        # Retornar la predicción
        return {"prediction": prediction}
    except Exception as e:
        # Retornar un mensaje de error si ocurre alguna excepción
        return {"error": f"Ocurrió un error: {str(e)}"}

# Ejecutar la aplicación FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
