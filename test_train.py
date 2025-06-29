# test_train.py
import os


import shutil
import pandas as pd
import pytest
# from train import load_data, preprocess_data, train_model, save_artifacts
from src.train import *
@pytest.fixture
def dummy_df():
    # Create a small dummy dataframe
    return pd.DataFrame({
        'sepal_length': [5.1, 4.9, 4.7, 6.0],
        'sepal_width': [3.5, 3.0, 3.2, 3.4],
        'petal_length': [1.4, 1.4, 1.3, 4.5],
        'petal_width': [0.2, 0.2, 0.2, 1.5],
        'species': ['setosa', 'setosa', 'setosa', 'versicolor']  # include at least 2 classes
    })

def test_load_data(tmp_path):
    # Create a dummy CSV file
    df = pd.DataFrame({
        'sepal_length': [5.1],
        'sepal_width': [3.5],
        'petal_length': [1.4],
        'petal_width': [0.2],
        'species': ['setosa']
    })
    test_file = tmp_path / "iris.csv"
    df.to_csv(test_file, index=False)

    loaded_df = load_data(str(test_file))
    assert not loaded_df.empty
    assert 'species' in loaded_df.columns

def test_preprocess_data(dummy_df):
    X, y, scaler = preprocess_data(dummy_df)
    assert X.shape[0] == dummy_df.shape[0]
    assert len(y) == dummy_df.shape[0]
    assert hasattr(scaler, "transform")

def test_train_model(dummy_df):
    X, y, _ = preprocess_data(dummy_df)
    model = train_model(X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)

from pathlib import Path

def test_save_artifacts(dummy_df):
    X, y, scaler = preprocess_data(dummy_df)
    model = train_model(X, y)

    save_artifacts(model, scaler)

    model_path = Path("artifacts/model.joblib")

    assert model_path.exists()
