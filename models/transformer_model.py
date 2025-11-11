"""
纯PyTorch实现的Transformer文本分类模型
数据格式：空格分隔的数字序列（已匿名化的token）
"""
from typing import Any, List, Optional
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

from .base import BaseModel

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """简单的数字token分词器，用于处理空格分隔的匿名化数字序列"""
    def __init__(self, vocab_size=8000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_built = False

    def build_vocab(self, texts: List[str]):
        """从文本构建词汇表（从数字序列中提取所有唯一数字）"""
        token_freq = {}
        for text in texts:
            # 分割空格分隔的数字
            tokens = text.split()
            for token in tokens:
                if token.isdigit():  # 确保是数字
                    token_freq[token] = token_freq.get(token, 0) + 1

        # 按频率排序，取top vocab_size-2（留给PAD和UNK）
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size - 2], start=2):
            self.token_to_idx[token] = idx

        self.vocab_built = True
        logger.info(f"Built vocabulary with {len(self.token_to_idx)} tokens")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """将文本编码为token id序列"""
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab first.")

        encoded = []
        for text in texts:
            tokens = text.split()[:self.max_length]
            ids = [self.token_to_idx.get(token, 1) for token in tokens if token.strip()]
            # Padding
            if len(ids) < self.max_length:
                ids.extend([0] * (self.max_length - len(ids)))
            encoded.append(ids[:self.max_length])

        return torch.tensor(encoded, dtype=torch.long)


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx],
            "labels": self.labels[idx]
        }


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """基于TransformerEncoder的文本分类器"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 14,
        max_length: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_length)

        # TransformerEncoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids) * (self.d_model ** 0.5)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)

        # 创建padding mask
        if attention_mask is None:
            attention_mask = (input_ids != 0)  # PAD token id = 0

        # TransformerEncoder需要mask，True表示被mask的位置
        src_key_padding_mask = ~attention_mask

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 平均池化（只对非padding位置）
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (x * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        x = sum_embeddings / sum_mask

        # 分类
        logits = self.classifier(x)
        return logits


class TransformerModel(BaseModel):
    """纯PyTorch实现的Transformer文本分类模型适配器"""

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: int = 14,
        max_length: int = 512,
        device: Optional[str] = None
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=max_length)
        self.model = None
        self.label_encoder = None

    def _init_model(self):
        """初始化模型"""
        if self.model is None:
            self.model = TransformerClassifier(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                num_classes=self.num_classes,
                max_length=self.max_length
            )
            self.model.to(self.device)
            logger.info(f"Initialized TransformerClassifier on {self.device}")

    def fit(self, X: List[str], y: List[Any], **kwargs) -> None:
        """训练模型

        Args:
            X: 文本列表（空格分隔的数字序列）
            y: 标签列表
            **kwargs: epochs, batch_size, learning_rate, val_split等
        """
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 16)
        learning_rate = kwargs.get("learning_rate", 1e-4)
        val_split = kwargs.get("val_split", 0.1)

        logger.info("="*60)
        logger.info("Starting Transformer Model Training")
        logger.info("="*60)
        logger.info(f"Training samples: {len(X)}")
        logger.info(f"Hyperparameters:")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Validation split: {val_split}")
        logger.info(f"  - Device: {self.device}")

        # 构建词汇表
        if not self.tokenizer.vocab_built:
            logger.info("-"*60)
            logger.info("Building vocabulary from training data...")
            self.tokenizer.build_vocab(X)

        # 编码标签
        if self.label_encoder is None:
            logger.info("-"*60)
            logger.info("Encoding labels...")
            self.label_encoder = LabelEncoder()
            y_enc = self.label_encoder.fit_transform(y)
            logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            logger.info(f"Classes: {self.label_encoder.classes_}")
        else:
            y_enc = self.label_encoder.transform(y)

        # 更新类别数
        actual_num_classes = len(self.label_encoder.classes_)
        if actual_num_classes != self.num_classes:
            logger.warning(f"Adjusting num_classes from {self.num_classes} to {actual_num_classes}")
            self.num_classes = actual_num_classes
            self.model = None  # 重新初始化

        # 初始化模型
        logger.info("-"*60)
        logger.info("Initializing model...")
        self._init_model()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model architecture:")
        logger.info(f"  - d_model: {self.d_model}")
        logger.info(f"  - num_layers: {self.num_layers}")
        logger.info(f"  - nhead: {self.nhead}")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")

        # 分割训练/验证集
        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_enc, test_size=val_split, random_state=42
            )
        else:
            X_train, y_train = X, y_enc
            X_val, y_val = None, None

        # 编码文本
        logger.info(f"Encoding {len(X_train)} training samples...")
        train_encodings = self.tokenizer.encode(X_train)
        train_labels = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TextDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            logger.info(f"Encoding {len(X_val)} validation samples...")
            val_encodings = self.tokenizer.encode(X_val)
            val_labels = torch.tensor(y_val, dtype=torch.long)
            val_dataset = TextDataset(val_encodings, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 训练
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        logger.info(f"Starting training for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits = self.model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if (batch_idx + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)

            # 验证
            if val_loader is not None:
                val_acc, val_loss = self._evaluate(val_loader, criterion)
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

        logger.info("Training completed.")

    def _evaluate(self, data_loader, criterion):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.model.train()
        return correct / total, total_loss / len(data_loader)

    def predict(self, X: List[str]) -> List[Any]:
        """预测"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first or load a saved model.")

        self.model.eval()
        encodings = self.tokenizer.encode(X)

        all_preds = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(encodings), batch_size):
                batch_encodings = encodings[i:i+batch_size].to(self.device)
                logits = self.model(batch_encodings)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)

        # 解码标签
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(all_preds).tolist()
        return all_preds

    def predict_proba(self, X: List[str]) -> Optional[List[float]]:
        """预测概率"""
        if self.model is None:
            raise RuntimeError("Model not initialized. Call fit() first or load a saved model.")

        self.model.eval()
        encodings = self.tokenizer.encode(X)

        all_probs = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(encodings), batch_size):
                batch_encodings = encodings[i:i+batch_size].to(self.device)
                logits = self.model(batch_encodings)
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values.cpu().numpy()
                all_probs.extend(max_probs)

        return all_probs

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型权重和配置
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "tokenizer_token_to_idx": self.tokenizer.token_to_idx,
            "label_encoder": self.label_encoder,
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
                "num_classes": self.num_classes,
                "max_length": self.max_length,
            }
        }

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(cls, path: str) -> "TransformerModel":
        """加载模型"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        # 创建实例
        instance = cls(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
            max_length=config["max_length"]
        )

        # 恢复tokenizer
        instance.tokenizer.token_to_idx = checkpoint["tokenizer_token_to_idx"]
        instance.tokenizer.vocab_built = True

        # 恢复label encoder
        instance.label_encoder = checkpoint["label_encoder"]

        # 初始化并加载模型
        instance._init_model()
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model.eval()

        logger.info(f"Model loaded from {path}")
        return instance
