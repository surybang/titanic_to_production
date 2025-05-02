"""A simple API to expose our trained RandomForest model for Titanic survival."""
from fastapi import FastAPI
from joblib import load

import pandas as pd

model = load('docs/models/titanic_model.joblib')

app = FastAPI(
    title="Prédiction de survie sur le Titanic",
    description="Application de prédiction de survie sur le Titanic 🚢 <br>Une version par API" +
    "faciliter la réutilisation du modèle 🚀")


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction de survie sur le Titanic",
        "Model_name": 'Titanic ML',
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict(
    sex: str = "female",
    age: float = 29.0,
    fare: float = 16.5,
    embarked: str = "S"
) -> str:
    """
    """

    df = pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )

    prediction = "Survived 🎉" if int(model.predict(df)) == 1 else "Dead ⚰️"

    return prediction
