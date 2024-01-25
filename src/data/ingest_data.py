import os
import pandas as pd
from src.utils import get_project_root
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def data_ingestion():
    """Reads the data from the csv file and returns a pandas dataframe."""
    filepath = os.path.join(get_project_root(), 'data', 'raw', 'census.csv')
    logging.info(f'Reading data from {filepath}')
    df = pd.read_csv(filepath)
    return df


def white_space_stripper(df):
    """Removes whitespaces from column names and values in the dataframe.

    Args:
        df: Pandas dataframe.

    Returns:
        df: Pandas dataframe.
    """
    logging.info('Removing whitespaces from column names and values')
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df


def hyphen_remover(df):
    """Removes hyphens from column names and values in the dataframe.

    Args:
        df: Pandas dataframe.

    Returns:
        df: Pandas dataframe.
    """
    logging.info('Removing hyphens from column names and values')
    df.columns = df.columns.str.replace('-', '_')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('-', '_')
    return df


def save_data(df):
    """Saves the dataframe to a csv file.

    Args:
        df: Pandas dataframe.
    """
    filepath = os.path.join(get_project_root(), 'data',
                            'processed', 'census.csv')
    logging.info(f'Saving data to {filepath}')
    df.to_csv(filepath, index=False)


def clean_main():
    df = data_ingestion()
    df = white_space_stripper(df)
    df = hyphen_remover(df)
    save_data(df)


if __name__ == '__main__':
    clean_main()
