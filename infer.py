import argparse
from pathlib import Path
import joblib
import pandas as pd

from data_utils import load_data, save_predictions
from models.registry import load_model_class, import_from_string
from models.sklearn_model import SklearnModel
from models.base import BaseModel


def predict_text(model_path: str, text: str):
    # try to detect which model adapter to use by loading the saved artifact
    # prefer using model adapter's load when possible
    try:
        # try loading with sklearn adapter first (backwards compatible)
        model = SklearnModel.load(model_path)
    except Exception:
        # fallback: joblib raw load and try to reconstruct
        data = joblib.load(model_path)
        pipe = data.get("pipeline")
        le = data.get("label_encoder")
        model = SklearnModel(pipeline=pipe, label_encoder=le)

    pred = model.predict([text])[0]
    prob = None
    probs = model.predict_proba([text])
    if probs is not None:
        prob = probs[0]
    return pred, prob


def predict_csv(model_path: str, input_csv: str, output_csv: str):
    texts, df_labels = load_data(input_csv)

    # attempt to load using sklearn adapter first
    try:
        model = SklearnModel.load(model_path)
    except Exception:
        data = joblib.load(model_path)
        model = SklearnModel(pipeline=data.get("pipeline"), label_encoder=data.get("label_encoder"))

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
