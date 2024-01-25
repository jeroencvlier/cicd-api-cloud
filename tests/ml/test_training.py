from src.ml.training import create_pipeline
from sklearn.pipeline import Pipeline


def test_create_pipeline():
    """Tests the create_pipeline function."""
    pipeline, used_columns = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert isinstance(used_columns, list)
    assert len(used_columns) > 0
