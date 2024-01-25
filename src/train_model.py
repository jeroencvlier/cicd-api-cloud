from data.ingest_data import clean_main
from ml.training import train_ml


if __name__ == '__main__':
    clean_main()
    train_ml()
