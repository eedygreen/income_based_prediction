"""
ml/artifacts.py module handles the saving and loading artifacts
"""
import os
import joblib


def save_artifacts(
    model_dir: str, 
    model=None, 
    encoder=None, 
    lb=None
):
    """
    save_artifacts
    saves Model, Encoder, Label Binarizer to specified directory

    Inputs
    ------
        model_dir: str
            directory to save the model, encoder & labelbinarizer
        model: pkl
            the actual classifier fitted objects
        encoder: pkl
            the encoder used for training
        lb: pkl
            the LabelBinarizer use for training

    Returns Nothing
    ---------------
        writes files (model, encoder, label-binarizer) to disk
    """
    os.makedirs(model_dir, exist_ok=True)

    artifact = {
        "model.pkl": model,
        "encoder.pkl": encoder,
        "lb.pkl": lb
    }
    
    for filename, value in artifact.items():
        if value is not None:
            joblib.dump(value, os.path.join(model_dir, filename))


def load_artifacts(
    model_dir: str, 
    model:str="model.pkl", 
    encoder:str="encoder.pkl", 
    lb:str="lb.pkl"
) -> tuple:
    """
    load artifact
        loads the stored artifacts from specified directory

    Inputs
    ------
        model_dir: str
            directory to load (model, encoder & labelbinarizer) from
        model: str "model.pkl"
            the actual classifier fitted objects
        encoder: str "encoder.pkl"
            the encoder used for training the model
        lb: str "lb.pkl"
            the LabelBinarizer used for training the model

    Returns
    -------
        Tuple (model, encoder, lb)
            model: the actual fitted classifier
            encoder: encoder used to train the model
            lb: label Binarizer used to train the model
    """
    ld_model, ld_encoder, ld_lb = [joblib.load(
        os.path.join(model_dir, v)) for v in [model, encoder, lb]]
    
    return ld_model, ld_encoder, ld_lb
