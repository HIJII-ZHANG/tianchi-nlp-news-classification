"""
基于TextCNN的PyTorch文本分类模型，适配BaseModel接口。

- 使用数字化分词（空格分隔）和可调max_length/vocab。
- 自动根据训练集长度/词表规模微调卷积核、embedding维度。
- 类别不平衡时自动启用加权损失（Focal或Weighted CE）。
"""
from typing import Any, List, Optional, Sequence
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """简单的数字token分词器，用于处理空格分隔的匿名化数字序列"""

    def __init__(self, vocab_size: int = 20000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_built = False

    def build_vocab(self, texts: List[str]) -> int:
        """从文本构建词汇表，返回实际词表大小"""
        token_freq = {}
        for text in texts:
            for token in text.split():
                if token.isdigit():
                    token_freq[token] = token_freq.get(token, 0) + 1

        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        limit = max(0, self.vocab_size - 2)  # 留给PAD/UNK
        for idx, (token, _) in enumerate(sorted_tokens[:limit], start=2):
            self.token_to_idx[token] = idx

        self.vocab_built = True
        logger.info(f"Built vocabulary size: {len(self.token_to_idx)} (cap={self.vocab_size})")
        return len(self.token_to_idx)

    def encode(self, texts: List[str]) -> torch.Tensor:
        """将文本编码为定长token id序列"""
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab first.")

        encoded = []
        for text in texts:
            tokens = [self.token_to_idx.get(tok, 1) for tok in text.split() if tok.strip()]
            tokens = tokens[: self.max_length]
            if len(tokens) < self.max_length:
                tokens.extend([0] * (self.max_length - len(tokens)))
            encoded.append(tokens)

        return torch.tensor(encoded, dtype=torch.long)


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(self, token_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        item = {"input_ids": self.token_ids[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class TextCNNClassifier(nn.Module):
    """标准TextCNN分类器"""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 256,
        kernel_sizes: Sequence[int] = (2, 3, 4, 5),
        num_filters: int = 256,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len] -> [batch, embed_dim, seq_len]
        x = self.embedding(input_ids).transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, reduction="none", weight=self.weight, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def _effective_num_weights(class_counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """基于Effective Number计算类别权重，适合长尾分布"""
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * len(class_counts)
    return weights


class TextCNNModel(BaseModel):
    """TextCNN文本分类模型适配器"""

    def __init__(
        self,
        vocab_size: int = 20000,
        max_length: int = 512,
        embedding_dim: int = 256,
        kernel_sizes: Sequence[int] = (2, 3, 4, 5),
        num_filters: int = 256,
        dropout: float = 0.3,
        batch_size: int = 128,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-2,
        epochs: int = 10,
        warmup_ratio: float = 0.1,
        label_smoothing: float = 0.05,
        focal_gamma: float = 1.5,
        focal_imbalance_threshold: float = 3.0,
        gradient_accumulation_steps: int = 1,
        auto_tune: bool = True,
        min_seq_len: int = 64,
        use_multi_gpu: bool = True,
        device: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.kernel_sizes = tuple(kernel_sizes)
        self.num_filters = num_filters
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.focal_imbalance_threshold = focal_imbalance_threshold
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.auto_tune = auto_tune
        self.min_seq_len = min_seq_len
        self.use_multi_gpu = use_multi_gpu

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
        self.model: Optional[nn.Module] = None
        self.label_encoder = LabelEncoder()
        self.num_labels: Optional[int] = None
        self.multi_gpu_enabled = False

    def _select_hyperparams(self, lengths: List[int], vocab_built: int):
        """根据训练集动态调整max_length/embedding/kernel_sizes"""
        if not lengths:
            return

        if self.auto_tune:
            p95 = int(np.percentile(lengths, 95))
            tuned_max_len = min(1024, max(self.min_seq_len, p95))
            if tuned_max_len != self.max_length:
                logger.info(f"Auto-tuning max_length: {self.max_length} -> {tuned_max_len}")
                self.max_length = tuned_max_len
                self.tokenizer.max_length = tuned_max_len

            if vocab_built < 8000:
                tuned_emb = 128
            elif vocab_built < 20000:
                tuned_emb = 256
            else:
                tuned_emb = 320
            if tuned_emb != self.embedding_dim:
                logger.info(f"Auto-tuning embedding_dim: {self.embedding_dim} -> {tuned_emb}")
                self.embedding_dim = tuned_emb

            median_len = int(np.median(lengths))
            if median_len > 200:
                tuned_kernel = (3, 4, 5, 7)
            elif median_len > 80:
                tuned_kernel = (2, 3, 4, 5, 6)
            else:
                tuned_kernel = (2, 3, 4, 5)
            if tuned_kernel != self.kernel_sizes:
                logger.info(f"Auto-tuning kernel_sizes: {self.kernel_sizes} -> {tuned_kernel}")
                self.kernel_sizes = tuned_kernel

    def _init_model(self):
        base_model = TextCNNClassifier(
            vocab_size=len(self.tokenizer.token_to_idx),
            num_labels=self.num_labels,
            embedding_dim=self.embedding_dim,
            kernel_sizes=self.kernel_sizes,
            num_filters=self.num_filters,
            dropout=self.dropout,
        ).to(self.device)

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and self.use_multi_gpu:
            logger.info(f"Using DataParallel on {num_gpus} GPUs")
            self.model = nn.DataParallel(base_model)
            self.multi_gpu_enabled = True
        else:
            self.model = base_model
            self.multi_gpu_enabled = False

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,} | embedding_dim={self.embedding_dim}, kernels={self.kernel_sizes}")

    def fit(self, X: List[str], y: List[Any], val_split: float = 0.1, dataloader_num_workers: int = 0, **kwargs) -> None:
        """训练模型"""
        logger.info("Starting TextCNN training")
        logger.info(f"Training samples: {len(X)}")

        # 覆盖可选超参
        self.epochs = kwargs.get("epochs", self.epochs)
        self.batch_size = kwargs.get("batch_size", self.batch_size)
        self.learning_rate = kwargs.get("learning_rate", self.learning_rate)

        # 长度/词表统计
        lengths = [len(t.split()) for t in X]
        logger.info(
            f"Seq length stats -> mean: {np.mean(lengths):.1f}, median: {np.median(lengths):.1f}, p95: {np.percentile(lengths, 95):.1f}"
        )

        # 构建词汇表
        vocab_built = self.tokenizer.build_vocab(X)
        # 动态调参
        self._select_hyperparams(lengths, vocab_built)

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_labels = len(self.label_encoder.classes_)
        logger.info(f"Number of classes: {self.num_labels}")

        # 类别分布
        class_counts = np.bincount(y_encoded, minlength=self.num_labels)
        imbalance_ratio = class_counts.max() / max(1, class_counts.min())
        logger.info(f"Class counts: {class_counts.tolist()}, imbalance ratio: {imbalance_ratio:.2f}")

        # 数据划分
        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=val_split, random_state=42, stratify=y_encoded
            )
            logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val = None, None

        # 编码文本
        train_tokens = self.tokenizer.encode(X_train)
        train_labels = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TextDataset(train_tokens, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=dataloader_num_workers,
            pin_memory=self.device.type == "cuda",
        )

        val_loader = None
        if X_val is not None:
            val_tokens = self.tokenizer.encode(X_val)
            val_labels = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TextDataset(val_tokens, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=dataloader_num_workers,
                pin_memory=self.device.type == "cuda",
            )

        # 初始化模型
        self._init_model()

        # 优化器/调度器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_steps = max(1, len(train_loader) * self.epochs // self.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 损失函数：根据不平衡程度选择
        class_weights = _effective_num_weights(class_counts)
        class_weights_t = torch.tensor(class_weights, dtype=torch.float, device=self.device)
        if imbalance_ratio >= self.focal_imbalance_threshold:
            criterion = FocalLoss(gamma=self.focal_gamma, weight=class_weights_t, label_smoothing=self.label_smoothing)
            loss_name = f"FocalLoss(gamma={self.focal_gamma})"
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=self.label_smoothing)
            loss_name = "Weighted CrossEntropy"
        logger.info(f"Using loss: {loss_name}")

        # 训练循环
        global_step = 0
        best_val_acc = 0.0
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            optimizer.zero_grad()
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                loss = criterion(logits, labels) / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % (20 * self.gradient_accumulation_steps) == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} | Step {batch_idx+1}/{len(train_loader)} | "
                        f"Loss {loss.item() * self.gradient_accumulation_steps:.4f} | "
                        f"Acc {100.0 * correct / max(1, total):.2f}% | LR {scheduler.get_last_lr()[0]:.2e}"
                    )

            train_acc = 100.0 * correct / max(1, total)
            avg_train_loss = epoch_loss / len(train_loader)

            if val_loader is not None:
                val_acc, val_loss = self._evaluate(val_loader, criterion)
                logger.info(
                    f"[Epoch {epoch+1}/{self.epochs}] Train Loss {avg_train_loss:.4f} | "
                    f"Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} | Val Acc {val_acc:.2f}%"
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    logger.info(f"New best val acc: {best_val_acc:.2f}%")
            else:
                logger.info(f"[Epoch {epoch+1}/{self.epochs}] Train Loss {avg_train_loss:.4f} | Train Acc {train_acc:.2f}%")

        logger.info("Training completed.")

    def _evaluate(self, loader: DataLoader, criterion):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        self.model.train()
        acc = 100.0 * correct / max(1, total)
        avg_loss = total_loss / len(loader)
        return acc, avg_loss

    def predict(self, X: List[str]) -> List[Any]:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self.model.eval()
        token_ids = self.tokenizer.encode(X)
        dataset = TextDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        preds = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                logits = self.model(input_ids)
                pred = logits.argmax(dim=-1).cpu().numpy()
                preds.extend(pred)

        return self.label_encoder.inverse_transform(preds).tolist()

    def predict_proba(self, X: List[str]) -> Optional[List[List[float]]]:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self.model.eval()
        token_ids = self.tokenizer.encode(X)
        dataset = TextDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        all_probs: List[List[float]] = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                logits = self.model(input_ids)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.extend(probs.tolist())
        return all_probs

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        checkpoint = {
            "model_state_dict": state_dict,
            "tokenizer_token_to_idx": self.tokenizer.token_to_idx,
            "tokenizer_max_length": self.tokenizer.max_length,
            "label_encoder_classes": self.label_encoder.classes_,
            "num_labels": self.num_labels,
            "config": {
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "embedding_dim": self.embedding_dim,
                "kernel_sizes": list(self.kernel_sizes),
                "num_filters": self.num_filters,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "warmup_ratio": self.warmup_ratio,
                "label_smoothing": self.label_smoothing,
                "focal_gamma": self.focal_gamma,
                "focal_imbalance_threshold": self.focal_imbalance_threshold,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "auto_tune": self.auto_tune,
                "min_seq_len": self.min_seq_len,
            },
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> "TextCNNModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        instance = cls(
            vocab_size=config.get("vocab_size", 20000),
            max_length=config.get("max_length", 512),
            embedding_dim=config.get("embedding_dim", 256),
            kernel_sizes=config.get("kernel_sizes", (2, 3, 4, 5)),
            num_filters=config.get("num_filters", 256),
            dropout=config.get("dropout", 0.3),
            batch_size=config.get("batch_size", 128),
            learning_rate=config.get("learning_rate", 2e-4),
            weight_decay=config.get("weight_decay", 1e-2),
            epochs=config.get("epochs", 10),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            label_smoothing=config.get("label_smoothing", 0.05),
            focal_gamma=config.get("focal_gamma", 1.5),
            focal_imbalance_threshold=config.get("focal_imbalance_threshold", 3.0),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            auto_tune=config.get("auto_tune", True),
            min_seq_len=config.get("min_seq_len", 64),
            use_multi_gpu=False,  # 加载时不启用多GPU包装
        )

        # 恢复tokenizer
        instance.tokenizer.token_to_idx = checkpoint["tokenizer_token_to_idx"]
        instance.tokenizer.vocab_built = True
        instance.tokenizer.max_length = checkpoint.get("tokenizer_max_length", instance.max_length)

        # 恢复标签编码器
        instance.label_encoder.classes_ = checkpoint["label_encoder_classes"]
        instance.num_labels = checkpoint["num_labels"]

        # 初始化并加载模型
        instance._init_model()
        state_dict = checkpoint["model_state_dict"]
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        logger.info(f"Model loaded from {path}")
        return instance
