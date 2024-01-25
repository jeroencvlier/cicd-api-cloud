# pytest in data ingestion

import pytest
import pandas as pd
from src.data.ingest_data import hyphen_remover, white_space_stripper, data_ingestion


def test_hyphen_remover():
    df = pd.DataFrame({'col-1': [1, 2, 3], 'col-2': ['a-b', 'c-d', 'e-f']})
    df_cleaned = hyphen_remover(df)
    assert 'col_1' in df_cleaned.columns
    assert 'col_2' in df_cleaned.columns
    assert 'a_b' in df_cleaned['col_2'].values
    assert 'c_d' in df_cleaned['col_2'].values
    assert 'e_f' in df_cleaned['col_2'].values


def test_data_ingestion():
    df = data_ingestion()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 100


def test_white_space_stripper():
    df = pd.DataFrame({'col 1': [1, 2, 3], 'col 2': [' a', 'b ', 'c c']})
    df_cleaned = white_space_stripper(df)
    assert 'col1' not in df_cleaned.columns
    assert 'col2' not in df_cleaned.columns
    assert 'col 1' in df_cleaned.columns
    assert 'col 2' in df_cleaned.columns
    assert 'a' in df_cleaned['col 2'].values
    assert 'b' in df_cleaned['col 2'].values
    assert 'c c' in df_cleaned['col 2'].values
