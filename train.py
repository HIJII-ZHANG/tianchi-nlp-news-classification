import argparse
import logging
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_utils import load_data
from models.registry import load_model_class

logger = logging.getLogger(__name__)




def train(train_csv: str, model_out: str, test_size: float = 0.1, random_state: int = 42, max_features: int = 50000, nrows: Optional[int] = None, model_spec: str = "sklearn", epochs: int = 10, batch_size: int = 16, learning_rate: float = 1e-4):
    """
    参数说明:
    - train_csv: 训练数据的CSV文件路径
    - model_out: 训练完成后模型保存的路径
    - test_size: 验证集占比
    - random_state: 随机种子，确保结果可复现
    - max_features: 特征最大数量（仅对sklearn模型有效）
    - nrows: 仅读取CSV的前n行（用于快速测试）
    - model_spec: 模型类规范（如'sklearn'或'transformer'或'models.my:MyModel'）
    - epochs: 训练轮数（针对transformer模型）
    - batch_size: 批量大小（针对transformer模型）
    - learning_rate: 学习率（针对transformer模型）
    该函数从CSV文件加载数据，进行训练/验证集划分，训练指定模型，并在验证集上评估性能，最后保存训练好的模型。
    """
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
    # pass kwargs - transformer models need epochs, batch_size, learning_rate; sklearn ignores them
    model.fit(X_train, y_train, max_features=max_features, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, val_split=0.0)

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
    parser.add_argument("--model-spec", type=str, default="sklearn", help="Model class spec (e.g. 'sklearn' or 'transformer' or 'models.my:MyModel')")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (for transformer models)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (for transformer models)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (for transformer models)")
    args = parser.parse_args()
    train(args.train_csv, args.model_out, test_size=args.test_size, max_features=args.max_features, nrows=args.nrows, model_spec=args.model_spec, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
