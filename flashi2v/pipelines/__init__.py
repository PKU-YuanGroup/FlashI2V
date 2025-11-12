from .t2v_pipeline import pipeline as t2v
from .flashi2v_pipeline import pipeline as flashi2v

pipelines = {}
pipelines.update(t2v)
pipelines.update(flashi2v)

__all__ = [
    'pipelines'
]