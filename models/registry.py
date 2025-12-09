import importlib
from typing import Any


def import_from_string(path: str) -> Any:
    """Import a class from a string like 'module.sub:ClassName' or 'module.sub.ClassName'."""
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    elif path.count(".") >= 1:
        parts = path.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]
    else:
        raise ImportError(f"Invalid import path: {path}")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def load_model_class(spec: str):
    """Resolve a model class spec to a class object.

    Supported specs:
    - 'sklearn' -> models.sklearn_model.SklearnModel
    - 'transformer' -> models.transformer_model.TransformerModel
    - 'bert' -> models.bert_model.BERTTextClassifier
    - full import path 'models.sklearn_model.SklearnModel'
    - 'module:Class'
    """
    if spec in ("sklearn", "sklearn_model"):
        return import_from_string("models.sklearn_model.SklearnModel")

    if spec in ("transformer", "transformer_model"):
        return import_from_string("models.transformer_model.TransformerModel")

    if spec in ("bert", "bert_model"):
        return import_from_string("models.bert_model.BERTTextClassifier")

    if spec in ("textcnn", "textcnn_model"):
        return import_from_string("models.textcnn_model.TextCNNModel")

    if spec in ("bert_ensemble", "bert-ensemble", "bertens"):
        return import_from_string("models.bert_ensemble.BertEnsembleClassifier")

    if spec in ("bert_finetune", "bert-hf", "berthf"):
        return import_from_string("models.bert_finetune.BertHFClassifier")

    # otherwise assume it's an import path
    return import_from_string(spec)
