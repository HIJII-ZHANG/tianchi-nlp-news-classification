"""
Ensemble inference over multiple BERTTextClassifier checkpoints.

Usage:
    python infer_bert_ensemble.py \
        --models models/bert_a.pt models/bert_b.pt models/bert_c.pt \
        --input-csv data/test_a.csv \
        --output-csv predictions.csv \
        [--use-tta --tta-rounds 5]
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import sys

# Ensure project root (folder containing `models/`) is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "models").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import load_data, save_predictions  # noqa: E402
from models.bert_model import BERTTextClassifier  # noqa: E402


def load_models(paths: List[str]) -> List[BERTTextClassifier]:
    models = []
    for p in paths:
        model = BERTTextClassifier.load(p)
        models.append(model)
    # Sanity: ensure label spaces align
    base_labels = models[0].label_encoder.classes_
    for m in models[1:]:
        if list(m.label_encoder.classes_) != list(base_labels):
            raise ValueError(f"Label classes mismatch between models: {base_labels} vs {m.label_encoder.classes_}")
    return models


def ensemble_predict_proba(models: List[BERTTextClassifier], texts: List[str], use_tta: bool, tta_rounds: int):
    probs_accum = []
    for idx, m in enumerate(models):
        probs = m.predict_proba(texts, use_tta=use_tta, tta_rounds=tta_rounds)
        if probs is None:
            raise RuntimeError(f"Model {idx} does not support predict_proba")
        probs_accum.append(np.array(probs))
    avg_probs = np.mean(probs_accum, axis=0)
    return avg_probs


def main():
    parser = argparse.ArgumentParser(description="Ensemble inference for BERT models (probability averaging)")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to trained BERT .pt checkpoints")
    parser.add_argument("--input-csv", required=True, help="CSV/TSV with text column")
    parser.add_argument("--output-csv", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--use-tta", action="store_true", help="Enable test-time augmentation (dropout averaging)")
    parser.add_argument("--tta-rounds", type=int, default=3, help="Number of TTA rounds when --use-tta")
    args = parser.parse_args()

    # Load data
    texts, _ = load_data(args.input_csv)
    if not texts:
        raise ValueError(f"No texts loaded from {args.input_csv}")

    # Load models
    model_paths = [str(Path(p)) for p in args.models]
    models = load_models(model_paths)

    # Ensemble probability
    avg_probs = ensemble_predict_proba(models, texts, use_tta=args.use_tta, tta_rounds=args.tta_rounds)
    label_encoder = models[0].label_encoder
    pred_ids = np.argmax(avg_probs, axis=1)
    preds = label_encoder.inverse_transform(pred_ids)

    # Save
    df_out = pd.DataFrame({"label": preds})
    save_predictions(df_out, args.output_csv)
    print(f"Saved ensemble predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
