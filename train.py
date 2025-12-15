import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_utils import load_data
from models.registry import load_model_class

# 配置日志
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_FILE_LOG_TARGETS = set()


def enable_file_logging(log_file: str) -> None:
    """Attach a file handler for logging if not already active."""
    if not log_file:
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path = str(log_path.resolve())
    if abs_path in _FILE_LOG_TARGETS:
        return

    file_handler = logging.FileHandler(abs_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logging.getLogger().addHandler(file_handler)
    _FILE_LOG_TARGETS.add(abs_path)
    logger.info("File logging enabled: %s", abs_path)




def train(train_csv: str, model_out: str, test_size: float = 0.1, random_state: int = 42, max_features: int = 50000, nrows: Optional[int] = None, model_spec: str = "sklearn", epochs: int = 10, batch_size: int = 16, learning_rate: float = 1e-4, pretrained: Optional[str] = None, dataloader_num_workers: int = 0, log_file: Optional[str] = None):
    """
    参数说明:
    - train_csv: 训练数据的CSV文件路径
    - model_out: 训练完成后模型保存的路径
    - test_size: 验证集占比
    - random_state: 随机种子，确保结果可复现
    - max_features: 特征最大数量（仅对sklearn模型有效）
    - nrows: 仅读取CSV的前n行（用于快速测试）
    - model_spec: 模型类规范（如'sklearn'或'transformer'或'models.my:MyModel'）
    - epochs: 训练轮数
    - batch_size: 批量大小
    - learning_rate: 学习率
    - log_file: 若提供，则将训练日志追加写入该文件
    该函数从CSV文件加载数据，进行训练/验证集划分，训练指定模型，并在验证集上评估性能，最后保存训练好的模型。
    """
    if log_file:
        enable_file_logging(log_file)

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
    # forward pretrained (local path or HF id) to model.fit; if pretrained is provided and
    # points to a local directory containing weights, the model loader will use it and
    # avoid downloading from the Hub.
    model.fit(
        X_train,
        y_train,
        max_features=max_features,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_split=0.0,
        pretrained=pretrained,
        dataloader_num_workers=dataloader_num_workers,
    )

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
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="Number of PyTorch DataLoader workers (transformer models)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to local pretrained model directory (prefered) or HF model id")
    parser.add_argument("--log-file", type=str, default=None, help="Optional path to append training logs")
    args = parser.parse_args()
    train(
        args.train_csv,
        args.model_out,
        test_size=args.test_size,
        max_features=args.max_features,
        nrows=args.nrows,
        model_spec=args.model_spec,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pretrained=args.pretrained,
        dataloader_num_workers=args.dataloader_num_workers,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
