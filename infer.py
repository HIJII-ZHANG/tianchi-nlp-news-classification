import argparse
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd

from data_utils import load_data, save_predictions
from models.sklearn_model import SklearnModel
from models.transformer_model import TransformerModel
from models.bert_model import BERTTextClassifier
from models.base import BaseModel


def load_model(model_path: str, model_type: Optional[str] = None) -> BaseModel:
    """加载模型，支持指定模型类型"""
    path = Path(model_path)

    # 如果指定了模型类型，直接使用
    if model_type == "bert":
        return BERTTextClassifier.load(model_path)
    elif model_type == "transformer":
        return TransformerModel.load(model_path)
    elif model_type == "sklearn":
        return SklearnModel.load(model_path)

    # 否则根据文件扩展名判断模型类型
    if path.suffix == ".pt":
        # PyTorch模型 - 先尝试BERT，再尝试Transformer
        try:
            return BERTTextClassifier.load(model_path)
        except Exception:
            return TransformerModel.load(model_path)
    elif path.suffix == ".joblib":
        # Sklearn模型
        try:
            return SklearnModel.load(model_path)
        except Exception:
            # 向后兼容：尝试直接加载joblib
            data = joblib.load(model_path)
            pipe = data.get("pipeline")
            le = data.get("label_encoder")
            return SklearnModel(pipeline=pipe, label_encoder=le)
    else:
        # 尝试按顺序加载
        try:
            return BERTTextClassifier.load(model_path)
        except Exception:
            try:
                return TransformerModel.load(model_path)
            except Exception:
                try:
                    return SklearnModel.load(model_path)
                except Exception:
                    raise ValueError(f"Unable to load model from {model_path}")


def predict_text(model_path: str, text: str, model_type: Optional[str] = None):
    model = load_model(model_path, model_type)
    pred = model.predict([text])[0]
    prob = None
    probs = model.predict_proba([text])
    if probs is not None:
        prob = probs[0]
    return pred, prob


def predict_csv(model_path: str, input_csv: str, output_csv: str, model_type: Optional[str] = None):
    texts, df_labels = load_data(input_csv)
    model = load_model(model_path, model_type)

    preds = model.predict(texts)
    df_out = pd.DataFrame({"label": preds})
    #if probs is not None:
    #    df_out["score"] = probs


    save_predictions(df_out, output_csv)
    print(f"Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Inference for news classifier")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--model-type", choices=["sklearn", "transformer", "bert"], help="Model type (sklearn, transformer, or bert)")
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--input-csv", help="CSV file to classify")
    parser.add_argument("--output-csv", default="predictions.csv")
    args = parser.parse_args()

    if args.text:
        label, prob = predict_text(args.model, args.text, args.model_type)
        print(f"Prediction: {label} (score={prob})")
    elif args.input_csv:
        predict_csv(args.model, args.input_csv, args.output_csv, args.model_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
