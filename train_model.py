"""
train_model.py
Top-level script: loads cleaned census data,
splits train/test, train the mdoel,
evaluate it overall and on categorical slices,
and save the model to disk.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import logger
from ml.data import CAT_FEATURES as cat_features
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    compute_all_slice_metrics,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "census_clean.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SLIC_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "slice_output.txt"
)


def main():
    try:
        logger.info("checking for model directory..")
        os.makedirs(MODEL_DIR, exist_ok=True)

        logger.info("Loading clean data...")
        data = pd.read_csv(DATA_PATH)

        logger.info("Splitting data...")
        train, test = train_test_split(
            data, test_size=0.20,
            random_state=42,
            stratify=data["salary"]
        )

        logger.info("processing training data...")
        X_train, y_train, encoder, lb = process_data(
            train,
            categorical_features=cat_features,
            label="salary", training=True,
        )
        logger.info("processing inference data...")
        X_test, y_test, _, _ = process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        logger.info("training model...")
        model = train_model(X_train, y_train)

        logger.info("computing inference...")
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(
            y_test,
            preds
        )
        model_performance = {
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "fbeta": f"{fbeta:.4f}"
        }
        logger.info(
            f"Overall test performance --\n {model_performance}"
        )

        compute_all_slice_metrics(
            df=test.reset_index(drop=True),
            categorical_features=cat_features,
            y=y_test,
            preds=preds,
            output_path=SLIC_OUTPUT_PATH,
        )
        logger.info(f"Slice performance written to {SLIC_OUTPUT_PATH}")

        logger.info("saving model to disk...")
        joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
        joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
        joblib.dump(lb, os.path.join(MODEL_DIR, "lb.pkl"))
        logger.info(f"Model, encoder amd label binarizer saved to {MODEL_DIR}/")

    except ValueError as err:
        logger.error(f"train model: error {err}")
    except Exception as e:
        logger.error(f"tarin model: error {e}", exc_info=True)

if __name__ == "__main__":
    main()
