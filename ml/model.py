"""
ml/model.py

Model training, inference, and evaluation.
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(df, feature, y, preds, categorical_features=None):
    """
    Compute precision/recall/fbeta for each unique value of one
    categorical feature, holding all rows matching that value as a
    'slice'.

    Inputs
    ----------
    df : pd.DataFrame
        The dataframe used to generate X/y (pre one-hot-encoding),
        must contain `feature` as a column, with the same row order
        and length as y/preds.
    feature : str
        Name of the categorical column to slice on.
    y : np.ndarray
        True labels, same row order as df.
    preds : np.ndarray
        Predicted labels, same row order as df.
    categorical_features : list[str], optional
        Unused, kept for signature symmetry / future extension.

    Returns
    -------
    list[str]:
        Lines of formatted slice performance output, suitable for
        writing to slice_output.txt.
    """
    lines = []
    for value in sorted(df[feature].unique()):
        mask = (df[feature] == value).values
        n = int(mask.sum())
        if n == 0:
            continue
        y_slice = y[mask]
        preds_slice = preds[mask]
        precision, recall, fbeta = compute_model_metrics(y_slice, preds_slice)
        lines.append(
            f"{feature}: {value}, n={n}, "
            f"precision={precision:.4f}, recall={recall:.4f}, "
            f"fbeta={fbeta:.4f}"
        )
    return lines


def compute_all_slice_metrics(df, categorical_features, y, preds, output_path):
    """
    Compute and write slice performance for every categorical
    feature to a single output file.

    Inputs
    ----------
    df : pd.DataFrame
        Dataframe with categorical_features as columns, row-aligned
        with y and preds.
    categorical_features : list[str]
        Categorical column names to slice on.
    y : np.ndarray
        True labels.
    preds : np.ndarray
        Predicted labels.
    output_path : str
        File path to write the slice report to.
    """
    all_lines = []
    for feature in categorical_features:
        all_lines.append(f"--- Slice performance for feature: {feature} ---")
        all_lines.extend(compute_slice_metrics(df, feature, y, preds))
        all_lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(all_lines))
