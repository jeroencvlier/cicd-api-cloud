import os
import logging
import pandas as pd
from src.ml.model import load_data, load_variables, compute_model_metrics


def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100


def test_load_variables():
    configs, model_path, label_path, used_columns_path = load_variables()

    assert isinstance(configs, dict)
    assert isinstance(model_path, str)
    assert isinstance(label_path, str)
    assert isinstance(used_columns_path, str)
    assert model_path.endswith('.pkl')
    assert label_path.endswith('.pkl')
    assert used_columns_path.endswith('.pkl')


def test_compute_model_metrics():
    y = [0, 1, 0, 1, 0, 1, 0, 0, 0,  1, 0, 1, ]
    preds = [0, 1, 0, 1, 1, 1, 0, 0, 0,  1, 0, 1, ]
    precision, recall, fbeta, accuracy, f1 = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert isinstance(accuracy, float)
    assert isinstance(f1, float)
    assert precision > 0.8
    assert recall > 0.8
    assert fbeta > 0.8
    assert accuracy > 0.8
    assert f1 > 0.8
