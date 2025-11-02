"""
Template adapter for a user-defined Transformer-style model.

If you have a custom class (for example in `my_models.py`):

class MyTransformerModel:
    def __init__(self, **kwargs):
        ...
    def fit(self, X: List[str], y: List[Any], **kwargs):
        # train or fine-tune
        ...
    def predict(self, X: List[str]) -> List[Any]:
        ...
    def predict_proba(self, X: List[str]) -> List[float]:
        ...
    def save(self, path: str):
        ...
    @classmethod
    def load(cls, path: str):
        ...

You can plug it into the training CLI by passing the import path:

python main.py train --model-spec my_models:MyTransformerModel

Or adapt this file to wrap Hugging Face transformers and implement the BaseModel interface.
"""

from typing import Any, List
from .base import BaseModel


class TemplateCustomModel(BaseModel):
    """A minimal wrapper that demonstrates the required methods for a custom model adapter.

    This file is intentionally a template/example and is not a runnable Transformer trainer.
    Copy/modify it to integrate your own model class.
    """

    def __init__(self, inner=None):
        # inner can be an instance of your custom model
        self.inner = inner

    def fit(self, X: List[str], y: List[Any], **kwargs) -> None:
        if self.inner is None:
            raise RuntimeError("Inner model not provided. Instantiate with your model instance or implement training here.")
        self.inner.fit(X, y, **kwargs)

    def predict(self, X: List[str]) -> List[Any]:
        if self.inner is None:
            raise RuntimeError("Inner model not provided. Instantiate with your model instance or implement prediction here.")
        return self.inner.predict(X)

    def predict_proba(self, X: List[str]):
        if self.inner is None:
            raise RuntimeError("Inner model not provided. Instantiate with your model instance or implement probability prediction here.")
        if hasattr(self.inner, "predict_proba"):
            return self.inner.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        if self.inner is None:
            raise RuntimeError("Inner model not provided. Instantiate with your model instance or implement saving logic here.")
        if hasattr(self.inner, "save"):
            self.inner.save(path)
        else:
            raise RuntimeError("Inner model has no save method")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("Implement loading logic for your custom model here")
