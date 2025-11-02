import argparse
import logging
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_utils import load_data
from models.registry import load_model_class

logger = logging.getLogger(__name__)




def train(train_csv: str, model_out: str, test_size: float = 0.1, random_state: int = 42, max_features: int = 50000, nrows: Optional[int] = None, model_spec: str = "sklearn"):
    texts, labels = load_data(train_csv, nrows=nrows)
    if not texts:
        logger.error("No texts found in the training CSV: %s", train_csv)
        return

    if not labels:
        logger.error("No labels found in the training CSV: %s", train_csv)
        return

    # Use a pluggable model class
    ModelClass = load_model_class(model_spec)
    model = ModelClass()

    print("Preparing training/validation split...")
    # label encoding if needed will be handled by model implementation
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=test_size, random_state=random_state)

    print(f"Training model ({model_spec})...")
    # pass common kwargs like max_features to the adapter
    model.fit(X_train, y_train, max_features=max_features)

    print("Evaluating on validation set...")
    preds = model.predict(X_val)
    # Print classification report (let sklearn infer labels/names)
    print(classification_report(y_val, preds))

    model.save(model_out)
    print(f"Saved model to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train news classification model")
    parser.add_argument("--train-csv", default="data/train_set.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--nrows", type=int, default=None, help="Only read this many rows from the CSV (for quick tests)")
    parser.add_argument("--model-spec", type=str, default="sklearn", help="Model class spec (e.g. 'sklearn' or 'models.my:MyModel')")
    args = parser.parse_args()
    train(args.train_csv, args.model_out, test_size=args.test_size, max_features=args.max_features, nrows=args.nrows, model_spec=args.model_spec)


if __name__ == "__main__":
    main()
