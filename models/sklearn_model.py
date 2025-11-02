from typing import Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

from .base import BaseModel


class SklearnModel(BaseModel):
    """Adapter that builds a sklearn Pipeline (Tfidf + LogisticRegression).

    This adapter keeps compatibility with previous joblib format (dict with pipeline and label_encoder).
    """

    def __init__(self, pipeline: Optional[Pipeline] = None, label_encoder: Optional[LabelEncoder] = None):
        self.pipeline = pipeline
        self.label_encoder = label_encoder

    def fit(self, X: List[str], y: List[Any], max_features: int = 50000, ngram_range=(1, 2), **kwargs) -> None:
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
        else:
            y_enc = self.label_encoder.transform(y)

        if self.pipeline is None:
            self.pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
                ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
            ])

        self.pipeline.fit(X, y_enc)

    def predict(self, X: List[str]) -> List[Any]:
        preds = self.pipeline.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(preds)
        return preds.tolist()

    def predict_proba(self, X: List[str]) -> Optional[List[float]]:
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            # return max prob per sample
            return probs.max(axis=1).tolist()
        return None

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "label_encoder": self.label_encoder}, p)

    @classmethod
    def load(cls, path: str) -> "SklearnModel":
        data = joblib.load(path)
        pipeline = data.get("pipeline")
        le = data.get("label_encoder")
        return SklearnModel(pipeline=pipeline, label_encoder=le)
