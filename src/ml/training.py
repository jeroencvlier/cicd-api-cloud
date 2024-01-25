import os
import logging
import pandas as pd
import itertools
import pickle
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from src.utils import get_project_root
from src.ml.model import (
    compute_model_metrics,
    load_model,
    compute_metrics_on_cat_slices,
    load_config,
    load_variables,
    load_data,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_pipeline():
    """Creates a pipeline to process the data.

    Returns:
        pipeline: sklearn.pipeline.Pipeline object.
    """

    configs = load_config()
    categorical_features = configs["model"]["cat_features"]
    numerical_features = configs["model"]["num_features"]

    logging.info("Creating pipeline")
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )

    catagorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", catagorical_transformer, categorical_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
               ("classifier", RandomForestClassifier())]
    )

    used_columns = list(
        itertools.chain.from_iterable(
            [x[2] for x in preprocessor.transformers])
    )

    return pipeline, used_columns


def train_ml():
    """Model pipeline for training."""

    configs, model_path, label_path, used_columns_path = load_variables()

    logging.info("Fitting")
    label = configs["model"]["target"]

    pipe, used_columns = create_pipeline()
    X = load_data()
    y = X.pop(label)

    # Binary encode the target
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y).ravel()

    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # fit the pipeline
    pipe.fit(X_train[used_columns], y_train)
    # validate
    y_pred = pipe.predict(X_test[used_columns])
    # evaluate results
    precision, recall, fbeta, accuracy, f1 = compute_model_metrics(
        y_test, y_pred)

    logging.info("Precision: %s", precision)
    logging.info("Recall: %s", recall)
    logging.info("FBeta: %s", fbeta)
    logging.info("Accuracy: %s", accuracy)
    logging.info("F1: %s", f1)

    # write scores to txt file
    scores_path = os.path.join(get_project_root(), "model", "scores.txt")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"FBeta: {fbeta}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1: {f1}\n")

    # serialise the model
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)

    with open(label_path, "wb") as f:
        pickle.dump(lb, f)

    with open(used_columns_path, "wb") as f:
        pickle.dump(used_columns, f)

    compute_metrics_on_cat_slices()


def infer_ml(X: pd.DataFrame):
    """
    Model pipeline for inference.

    Args:
        X: Pandas dataframe containing the data.

    Returns:
        y_pred: Predicted labels.
        y_labels: Predicted labels in string format."""

    pipe, lb, used_columns = load_model()
    missing_columns = set(X.columns) - set(used_columns)
    if missing_columns:
        logging.warning(
            "Columns %s are missing from the config file", missing_columns)

    y_pred = pipe.predict(X[used_columns])
    y_labels = lb.inverse_transform(y_pred)

    return y_pred, y_labels


if __name__ == "__main__":
    train_ml()
