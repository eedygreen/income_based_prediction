"""
tests/test_model.py
Unit tests for ml/model.py and ml/data.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def synthetic_df():
    """
    Small synthetic dataframe mimicking the census schema,
    used to tests don't depend on the full census.csv datatset
    """

    return pd.DataFrame(
        {
            "age": [25, 40, 35, 50, 28, 60, 33, 45],
            "education-num": [10, 13, 9, 14, 12, 16, 9, 13],
            "hours-per-week": [40, 45, 38, 50, 40, 60, 35, 42],
            "workclass": [
                "Private", "Private", "Self-emp", "Private",
                "Private", "Self-emp", "Private", "Private",
            ],
            "sex": ["Male", "Female", "Male", "Female",
                    "Male", "Female", "Male", "Female"],
            "salary": [
                "<=50K", ">50K", "<=50K", ">50K",
                "<=50K", ">50K", "<=50K", ">50K",
            ],
        }
    )


cat_feautres = ["workclass", "sex"]


def test_train_model_returns_fitted_estimator(synthetic_df):
    """
    train_model should return an object with a working predict
    method after being given valid training data.
    It actually fit, rather than returning an untrained model instance.
    """
    X, y, encoder, lb = process_data(
        synthetic_df, categorical_features=cat_feautres,
        label="salary", training=True,
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    # sklearn marks fitted estimators with trailing-underscore attrs
    assert hasattr(model, "estimators_")
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_inference_output_shape_and_values(synthetic_df):
    """
    inference() should return a prediction array
    matching the number of input rows,
    with values restricted to the binarized label set {0, 1}.
    """
    X, y, encoder, lb = process_data(
        synthetic_df, categorical_features=cat_feautres,
        label="salary", training=True,
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_bounds_and_known_case():
    """
    compute_model_metrics should return precision/recall/fbeta
    each within [0, 1], and should compute the textbook-correct
    values on known, hand-checkable input.
    """
    # 2 true positive, 1 false positive, 1 false negative, 0 true negative
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    for metric in (precision, recall, fbeta):
        assert 0.0 <= metric <= 1.0

    # precision = TP / (TP + FP) = 2 / 3
    assert precision == pytest.approx(2 / 3, abs=1e-6)
    # recall = TP / (TP + FN) = 2 / 3
    assert recall == pytest.approx(2 / 3, abs=1e-6)


def test_process_data_train_inference_dimension_consistency(synthetic_df):
    """
    The number of output feature columns must be identical between
    training mode (fit_transform) and inference mode (transform with
    a previously-fitted encoder) -- a mismatch here would silently
    break any downstream model.predict() call.
    """
    X_train, _, encoder, lb = process_data(
        synthetic_df,
        categorical_features=cat_feautres,
        label="salary",
        training=True,
    )
    X_infer, _, _, _ = process_data(
        synthetic_df.iloc[:3],
        categorical_features=cat_feautres,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert X_train.shape[1] == X_infer.shape[1]
