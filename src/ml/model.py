import os
import json
import logging
import pickle
import pandas as pd

from sklearn.metrics import (
    fbeta_score, precision_score, recall_score, accuracy_score, f1_score)
from src.utils import get_project_root

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_data() -> pd.DataFrame:
    """Loads the data from the specified path.

    Returns:
        df: Pandas dataframe containing the data.
    """
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'data', 'processed', 'census.csv')
    logging.info(f'Loading data from %s', csv_path)
    df = pd.read_csv(csv_path)
    return df


def load_variables():
    logging.info("Loading variables")
    configs = load_config()
    label = configs['model']['target']
    model_path = os.path.join(get_project_root(), 'model', 'pipeline.pkl')
    label_path = os.path.join(get_project_root(), 'model', 'label.pkl')
    used_columns_path = os.path.join(
        get_project_root(), 'model', 'used_columns.pkl')
    return configs, model_path, label_path, used_columns_path


def load_config() -> dict:
    """Loads the configuration file from the specified path.

    Returns:
        config: Dictionary containing the configuration parameters.
    """
    project_root = get_project_root()
    json_path = os.path.join(project_root, 'src', 'config.json')
    logging.info(f'Loading config from {json_path}')
    with open(json_path, encoding='utf-8') as f:
        config = json.load(f)
    return config


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = round(fbeta_score(y, preds, beta=1, zero_division=1), 4)
    precision = round(precision_score(y, preds, zero_division=1), 4)
    recall = round(recall_score(y, preds, zero_division=1), 4)
    accuracy = round(accuracy_score(y, preds), 4)
    f1 = round(f1_score(y, preds, zero_division=1), 4)

    return precision, recall, fbeta, accuracy, f1


def load_model():
    """Loads the trained model from the specified path.

    Returns:
        model: Trained model.
    """
    _, model_path, label_path, used_columns_path = load_variables()

    assert os.path.exists(model_path), 'Model must be trained first'
    assert os.path.exists(
        label_path), 'LabelBinarizer must be trained first'
    assert os.path.exists(
        used_columns_path), 'Used columns must be trained first'

    logging.info('Loading model from %s', model_path)

    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)

    with open(label_path, 'rb') as f:
        lb = pickle.load(f)

    with open(used_columns_path, 'rb') as f:
        used_columns = pickle.load(f)

    return pipe, lb, used_columns


def compute_metrics_on_cat_slices():

    logging.info('Computing metrics on categorical slices')
    configs, model_path, label_path, used_columns_path = load_variables()
    pipe, lb, used_columns = load_model()
    df = load_data()

    cat_features = configs['model']['cat_features']
    label = configs['model']['target']
    slice_path = os.path.join(get_project_root(), 'model', 'slice_output.txt')
    with open(slice_path, 'w', encoding='utf-8') as file:
        file.write('Computing metrics on categorical slices:\n')
        for cat in cat_features:
            file.write(f'\n\tColumn: {cat}\n')
            for cat_value in df[cat].unique():
                df_slice = df[df[cat] == cat_value]
                X = df_slice.drop(label, axis=1)
                y = lb.transform(df_slice[label]).ravel()
                y_pred = pipe.predict(X[used_columns])
                precision, recall, fbeta, accuracy, f1 = compute_model_metrics(
                    y, y_pred)
                file.write(f'\t\t{cat_value}\n')
                file.write(f'\t\t\tPrecision: {precision}\n')
                file.write(f'\t\t\tRecall: {recall}\n')
                file.write(f'\t\t\tFBeta: {fbeta}\n')
                file.write(f'\t\t\tAccuracy: {accuracy}\n')
                file.write(f'\t\t\tF1: {f1}\n')
