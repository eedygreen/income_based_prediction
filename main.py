"""
main.py
FastAPI application serving the trained census income model.
"""
import os
import joblib
import pandas as pd
from utils import logger
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from ml.data import CAT_FEATURES as cat_features
from ml.data import process_data
from ml.model import inference


MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

app = FastAPI(title="Census Income Inference API")

model = None
encoder = None
lb = None

@app.on_event("startup")
def load_artifact():
    """
    Load the trained model, encoder, and label binarizer once at
    startup rather than per-request, to avoid repeated disk I/O on
    every prediction call.
    """
    global model, encoder, lb
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))
    lb = joblib.load(os.path.join(MODEL_DIR, "lb.pkl"))

class CensusInput(BaseModel):
    """
    Pydantic Schema for a single inference request.

    Field names in the source dataset use hyphens (e.g. 'education-num'),
    which are not valid Python Idenitifies.
    Each such field is declared with a valid Python attribute name and
    mapped to the original column name via `Field(alias=...)`, with
    `populate_by_name` enabled so callers can supply either the
    Python name or the original hyphenated name.
    """
    model_config = ConfigDict(
        populated_by_name=True,
        json_shecma_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 178356,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(
        ..., alias="education-num", example=13
    )
    marital_status: str = Field(
        ..., alias="marital-status", example="Married-civ-spouse"
    )
    occupation: str = Field(..., example="Exec-managerial")
    relationship: str = Field(..., example="Husband")
    race: str = Field(...,example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(
        ..., alias="hours-per-week", example=40
    )
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )


@app.get("/")
def welcome() -> dict:
    """
    Root endpoint -- simple welcome/health message.
    """
    return {"message": "Welcome to the Census Income Inference API."}

@app.post("/predict")
def predict(payload: CensusInput) -> dict:
    """
    Run model inference on a single input record

    Inputs
    -------
    payload: CensusInput
        A single census record, supplied with either Python-style
        or original hyphenated field names

    Returns
    -------
    dict {"prediction": "<=50k"} or {"prediction": ">50k"}
    """
    try:
        logger.info("laoding payload...")
        record = payload.model_dump(by_alias=True)
        df = pd.DataFrame([record])

        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        logger.info("predicting outcome...")
        pred = inference(model, X)
        label = lb.inverse_transform(pred[0])

        logger.info(f"prediction: {label}")
        return {"prediction": label}

    except ValueError as err:
        logger.error(f"inference: error {err}")
    except Exception as e:
        logger.error(f"inference: error {e}")
