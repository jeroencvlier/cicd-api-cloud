from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.ml.training import infer_ml, load_model


app = FastAPI(
    title="Income Prediction API",
    description="This API uses a trained model to predict the income of a person. Returns 1 if the income is greater than 50K, and 0 otherwise.",
    version="0.1",
)


class IncomePrediction(BaseModel):
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str
    age: int
    fnlgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int


def catagorical_check():
    pipe, _, _ = load_model()

    preprocessor = pipe.named_steps["preprocessor"]
    transformers = preprocessor.transformers_

    for transformer_name, _, columns in transformers:
        if transformer_name == "cat":
            cat_columns = columns
            cat_pipeline = preprocessor.named_transformers_["cat"]
            onehotencoder = cat_pipeline.named_steps["onehotencoder"]
            categories = onehotencoder.categories_

    return cat_columns, categories


@app.get("/")
async def root():
    return {"message": "Welcome to the Model API!"}


@app.post("/predict")
async def exercise_function(data: IncomePrediction):
    cat_columns, categories = catagorical_check()
    for col, catagory in zip(cat_columns, categories):
        if getattr(data, col) not in catagory:
            raise HTTPException(
                status_code=400, detail=f"Invalid input for {col}")

    if (
        getattr(data, "age") < 0
        or getattr(data, "fnlgt") < 0
        or getattr(data, "education_num") < 0
        or getattr(data, "capital_gain") < 0
        or getattr(data, "capital_loss") < 0
        or getattr(data, "hours_per_week") < 0
    ):
        raise HTTPException(
            status_code=400, detail="Invalid input for numerical columns"
        )

    if getattr(data, "age") > 110 or getattr(data, "hours_per_week") > 100:
        raise HTTPException(
            status_code=400, detail="Invalid input for numerical columns"
        )

    data = pd.DataFrame([data.dict()])
    pred, pred_label = infer_ml(data)
    return {"prediction": str(pred[0]), "label": str(pred_label[0])}
