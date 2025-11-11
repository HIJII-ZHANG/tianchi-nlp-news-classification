import argparse
from pathlib import Path
import joblib
import pandas as pd

from data_utils import load_data, save_predictions
from models.sklearn_model import SklearnModel
from models.transformer_model import TransformerModel
from models.base import BaseModel


def load_model(model_path: str) -> BaseModel:
    """智能加载模型，自动检测模型类型"""
    path = Path(model_path)

    # 根据文件扩展名判断模型类型
    if path.suffix == ".pt":
        # PyTorch模型 - 使用TransformerModel
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
            return TransformerModel.load(model_path)
        except Exception:
            try:
                return SklearnModel.load(model_path)
            except Exception:
                raise ValueError(f"Unable to load model from {model_path}")


def predict_text(model_path: str, text: str):
    model = load_model(model_path)
    pred = model.predict([text])[0]
    prob = None
    probs = model.predict_proba([text])
    if probs is not None:
        prob = probs[0]
    return pred, prob


def predict_csv(model_path: str, input_csv: str, output_csv: str):
    texts, df_labels = load_data(input_csv)
    model = load_model(model_path)

    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    df_out = pd.DataFrame({"prediction": preds})
    if probs is not None:
        df_out["score"] = probs

    save_predictions(df_out, output_csv)
    print(f"Saved predictions to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Inference for news classifier")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--input-csv", help="CSV file to classify")
    parser.add_argument("--output-csv", default="predictions.csv")
    args = parser.parse_args()

    if args.text:
        label, prob = predict_text(args.model, args.text)
        print(f"Prediction: {label} (score={prob})")
    elif args.input_csv:
        predict_csv(args.model, args.input_csv, args.output_csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
