"""
Ensemble wrapper around multiple BERTTextClassifier models.

Trains several BERT models on stratified subsamples to create diversity, then
averages probabilities at inference time.
"""
from typing import Any, List, Optional, Dict
from pathlib import Path
import logging
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .base import BaseModel
from .bert_model import BERTTextClassifier

logger = logging.getLogger(__name__)


class BertEnsembleClassifier(BaseModel):
    """Train multiple BERTTextClassifier models and average their probabilities."""

    def __init__(
        self,
        n_models: int = 3,
        sample_frac: float = 0.85,
        seeds: Optional[List[int]] = None,
        combine_mode: str = "prob",
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_length_choices: Optional[List[int]] = None,
        # n_models=3,  不同截断视角
    ):
        '''
        __init__ 的 Docstring

        :param n_models: 模型数量
        :type n_models: int
        :param sample_frac:
        :type sample_frac: float
        :param seeds: 种子列表
        :type seeds: Optional[List[int]]
        :param combine_mode
        :type combine_mode: str
        :param model_kwargs: 模型参数
        :type model_kwargs: Optional[Dict[str, Any]]
        :param max_length_choices: [1024, 1536, 2048] # 例子，默认为None
        :type max_length_choices: Optional[List[int]]
        '''
        self.n_models = max(1, n_models)
        self.sample_frac = max(0.1, min(sample_frac, 1.0))
        self.seeds = seeds or list(range(42, 42 + self.n_models))
        if len(self.seeds) < self.n_models:
            # extend seed list if too short
            last = self.seeds[-1] if self.seeds else 42
            self.seeds.extend(range(last + 1, last + 1 + (self.n_models - len(self.seeds))))
        self.combine_mode = combine_mode
        self.model_kwargs = model_kwargs or {}
        self.max_length_choices = max_length_choices

        self.models: List[BERTTextClassifier] = []
        self.label_encoder = LabelEncoder()
        self.num_labels: Optional[int] = None

    def fit(self, X: List[str], y: List[Any], **kwargs) -> None:
        """Train multiple BERT models on stratified subsamples."""
        if not X or not y:
            raise ValueError("Empty training data")

        # Fit a shared label encoder to keep label mapping consistent.
        self.label_encoder.fit(y)
        self.num_labels = len(self.label_encoder.classes_)

        self.models = []
        for i in range(self.n_models):
            seed = self.seeds[i]
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.sample_frac < 1.0:
                X_sub, _, y_sub, _ = train_test_split(
                    X, y, train_size=self.sample_frac, stratify=y, random_state=seed
                )
            else:
                X_sub, y_sub = X, y

            logger.info(f"[Ensemble {i+1}/{self.n_models}] training with seed={seed}, samples={len(X_sub)}")
            model_cfg = copy.deepcopy(self.model_kwargs)
            if self.max_length_choices:
                chosen_max_len = self.max_length_choices[i % len(self.max_length_choices)]
                model_cfg["max_length"] = chosen_max_len
                logger.info(f"[Ensemble {i+1}] using max_length={chosen_max_len}")
            model = BERTTextClassifier(**model_cfg)
            # Use shared label encoder so all models share label-id mapping.
            model.label_encoder = self.label_encoder
            model.num_labels = self.num_labels
            model.fit(X_sub, y_sub, **kwargs)
            self.models.append(model)

    def _ensure_trained(self):
        if not self.models:
            raise RuntimeError("No trained models in ensemble. Call fit() first or load a saved ensemble.")

    def predict_proba(self, X: List[str], **kwargs) -> List[List[float]]:
        self._ensure_trained()
        all_probs = []
        for idx, model in enumerate(self.models):
            probs = model.predict_proba(X, **kwargs)
            if probs is None:
                raise RuntimeError("Underlying model does not support predict_proba")
            all_probs.append(np.array(probs))
            logger.debug(f"Model {idx+1} prob shape: {np.array(probs).shape}")
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs.tolist()

    def predict(self, X: List[str]) -> List[Any]:
        avg_probs = np.array(self.predict_proba(X))
        pred_ids = avg_probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(pred_ids).tolist()

    def save(self, path: str) -> None:
        self._ensure_trained()
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_entries = []
        for model in self.models:
            state_dict = (
                model.model.module.state_dict() if hasattr(model.model, "module") else model.model.state_dict()
            )
            entry = {
                "model_state_dict": state_dict,
                "tokenizer_vocab": model.tokenizer.token_to_idx,
                "tokenizer_max_length": model.tokenizer.max_length,
                "label_encoder_classes": model.label_encoder.classes_,
                "num_labels": model.num_labels,
                "config": {
                    "vocab_size": model.vocab_size,
                    "d_model": model.d_model,
                    "num_layers": model.num_layers,
                    "num_heads": model.num_heads,
                    "d_ff": model.d_ff,
                    "max_length": model.max_length,
                    "dropout": model.dropout,
                    "warmup_ratio": model.warmup_ratio,
                    "label_smoothing": model.label_smoothing,
                    "gradient_accumulation_steps": model.gradient_accumulation_steps,
                    "use_focal_loss": model.use_focal_loss,
                    "focal_gamma": model.focal_gamma,
                    "batch_size": model.batch_size,
                    "learning_rate": model.learning_rate,
                    "epochs": model.epochs,
                    "use_multi_gpu": False,  # save CPU-friendly
                },
            }
            model_entries.append(entry)

        checkpoint = {
            "ensemble_meta": {
                "n_models": self.n_models,
                "sample_frac": self.sample_frac,
                "seeds": self.seeds,
                "combine_mode": self.combine_mode,
                "model_kwargs": self.model_kwargs,
                "max_length_choices": self.max_length_choices,
            },
            "label_encoder_classes": self.label_encoder.classes_,
            "models": model_entries,
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Ensemble saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> "BertEnsembleClassifier":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("ensemble_meta", {})
        instance = cls(
            n_models=meta.get("n_models", len(checkpoint.get("models", []))),
            sample_frac=meta.get("sample_frac", 1.0),
            seeds=meta.get("seeds"),
            combine_mode=meta.get("combine_mode", "prob"),
            model_kwargs=meta.get("model_kwargs", {}),
            max_length_choices=meta.get("max_length_choices"),
        )
        instance.label_encoder.classes_ = checkpoint["label_encoder_classes"]
        instance.num_labels = len(instance.label_encoder.classes_)

        models_data = checkpoint.get("models", [])
        instance.models = []
        for idx, entry in enumerate(models_data):
            cfg = entry["config"]
            model = BERTTextClassifier(
                vocab_size=cfg.get("vocab_size", 7000),
                d_model=cfg.get("d_model", 768),
                num_layers=cfg.get("num_layers", 10),
                num_heads=cfg.get("num_heads", 12),
                d_ff=cfg.get("d_ff", 3072),
                max_length=cfg.get("max_length", 1024),
                dropout=cfg.get("dropout", 0.2),
                warmup_ratio=cfg.get("warmup_ratio", 0.1),
                label_smoothing=cfg.get("label_smoothing", 0.1),
                gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
                use_focal_loss=cfg.get("use_focal_loss", True),
                focal_gamma=cfg.get("focal_gamma", 2.0),
                batch_size=cfg.get("batch_size", 16),
                learning_rate=cfg.get("learning_rate", 1.5e-5),
                epochs=cfg.get("epochs", 20),
                use_multi_gpu=False,
            )
            model.tokenizer.token_to_idx = entry["tokenizer_vocab"]
            model.tokenizer.vocab_built = True
            model.tokenizer.max_length = entry.get("tokenizer_max_length", cfg.get("max_length", 1024))
            model.label_encoder.classes_ = entry["label_encoder_classes"]
            model.num_labels = entry["num_labels"]

            state_dict = entry["model_state_dict"]
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
            model.model.eval()
            instance.models.append(model)
            logger.info(f"Loaded ensemble member {idx+1}/{len(models_data)}")

        return instance
