"""
main.py
FastAPI application serving the trained census income model.
"""
import os
import joblib
import pandas as pd
from utils import logger
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
from ml.data import CAT_FEATURES as cat_features
from ml.data import process_data
from ml.model import inference

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the trained model, encoder, and label binarizer once at
    startup rather than per-request, to avoid repeated disk I/O on
    every prediction call. Stored on app.state rather than module
    globals so artifact lifetime is tied explicitly to the app
    instance - this matters for tests, where multiple TestClient
    instances/app instances should not share or stomp on each
    other's loaded state.
    """
    app.state.model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    app.state.encoder = joblib.load(
        os.path.join(MODEL_DIR, "encoder.pkl")
    )
    app.state.lb = joblib.load(os.path.join(MODEL_DIR, "lb.pkl"))
    yield
    # No explicit cleanup needed -- these are in-memory objects with
    # no open file handles/connections to release. Listed here for
    # symmetry with the lifespan pattern, not because it does
    # anything: app.state is torn down with the app itself.
    app.state.model = None
    app.state.encoder = None
    app.state.lb = None


app = FastAPI(
    title="Census Income Inference API",
    lifespan=lifespan
)


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
        populate_by_name=True,
        json_schema_extra={
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
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(
        ..., alias="education-num"
    )
    marital_status: str = Field(
        ..., alias="marital-status"
    )
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(
        ..., alias="hours-per-week"
    )
    native_country: str = Field(
        ..., alias="native-country"
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

        X, _, _, lb = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=app.state.encoder,
            lb=app.state.lb,
        )

        logger.info("predicting outcome...")
        pred = inference(app.state.model, X)
        label = lb.inverse_transform(pred)[0]

        logger.info(f"prediction: {label}")
        return {"prediction": label}

    except ValueError as err:
        logger.error(f"inference: error {err}")
    except Exception as e:
        logger.error(f"inference: error {e}")
