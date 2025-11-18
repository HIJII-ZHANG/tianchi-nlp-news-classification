"""
纯PyTorch实现的BERT文本分类模型
数据格式：空格分隔的数字序列（已匿名化的token）
"""
from typing import Any, List, Optional
import logging
import math
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
    def __init__(self, vocab_size=5000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        self.vocab_built = False

    def build_vocab(self, texts: List[str]):
        """从文本构建词汇表（从数字序列中提取所有唯一数字）"""
        token_freq = {}
        for text in texts:
            tokens = text.split()
            for token in tokens:
                if token.isdigit():
                    token_freq[token] = token_freq.get(token, 0) + 1

        # 按频率排序，取top vocab_size-4（留给特殊token）
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size - 4], start=4):
            self.token_to_idx[token] = idx

        self.vocab_built = True
        logger.info(f"Built vocabulary with {len(self.token_to_idx)} tokens")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """将文本编码为token id序列，添加CLS和SEP标记"""
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab first.")

        encoded = []
        for text in texts:
            tokens = text.split()[:self.max_length - 2]  # 留空间给CLS和SEP
            # 添加CLS token
            token_ids = [self.token_to_idx["<CLS>"]]
            # 编码文本
            for token in tokens:
                if token.isdigit() and token in self.token_to_idx:
                    token_ids.append(self.token_to_idx[token])
                else:
                    token_ids.append(self.token_to_idx["<UNK>"])
            # 添加SEP token
            token_ids.append(self.token_to_idx["<SEP>"])

            # Padding
            padding_length = self.max_length - len(token_ids)
            token_ids.extend([self.token_to_idx["<PAD>"]] * padding_length)

            encoded.append(token_ids[:self.max_length])

        return torch.tensor(encoded, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = x.size(0)

        # 线性变换并分割成多头
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.W_o(attn_output)
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class BERTEncoderLayer(nn.Module):
    """BERT编码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 多头自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class BERTClassifier(nn.Module):
    """BERT分类器模型"""
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # BERT编码器层
        self.encoder_layers = nn.ModuleList([
            BERTEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 分类头
        self.pooler = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 词嵌入 + 位置编码
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # 创建注意力掩码（用于padding）
        if attention_mask is not None:
            # 扩展维度以匹配多头注意力
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        else:
            mask = None

        # 通过所有编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)

        # 使用[CLS] token的输出（第一个位置）进行分类
        cls_output = x[:, 0, :]  # [batch_size, d_model]
        pooled_output = torch.tanh(self.pooler(cls_output))
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, token_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.token_ids = token_ids
        self.labels = labels

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        item = {'input_ids': self.token_ids[idx]}

        # 创建attention mask（非padding的位置为1）
        attention_mask = (self.token_ids[idx] != 0).long()
        item['attention_mask'] = attention_mask

        if self.labels is not None:
            item['labels'] = self.labels[idx]

        return item


class BERTTextClassifier(BaseModel):
    """BERT文本分类模型包装器"""

    def __init__(
        self,
        vocab_size: int = 5000,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_length: int = 512,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        """
        初始化BERT文本分类器

        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            d_ff: 前馈网络维度
            max_length: 最大序列长度
            batch_size: 批次大小
            learning_rate: 学习率
            epochs: 训练轮数
            dropout: Dropout概率
            device: 设备 (cuda/cpu)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.tokenizer = SimpleTokenizer(vocab_size, max_length)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = None

        logger.info(f"Initialized BERT classifier with device: {self.device}")

    def fit(self, X: List[str], y: List[Any], val_size: float = 0.1, **kwargs) -> None:
        """训练模型"""
        logger.info(f"Starting BERT training with {len(X)} samples")

        # 构建词汇表
        self.tokenizer.build_vocab(X)

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_labels = len(self.label_encoder.classes_)
        logger.info(f"Number of classes: {self.num_labels}")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=val_size, random_state=42, stratify=y_encoded
        )
        logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

        # 编码文本
        logger.info("Encoding training texts...")
        train_token_ids = self.tokenizer.encode(X_train)
        logger.info("Encoding validation texts...")
        val_token_ids = self.tokenizer.encode(X_val)

        # 创建数据集
        train_dataset = TextDataset(train_token_ids, torch.tensor(y_train, dtype=torch.long))
        val_dataset = TextDataset(val_token_ids, torch.tensor(y_val, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # 初始化模型
        self.model = BERTClassifier(
            vocab_size=len(self.tokenizer.token_to_idx),
            num_labels=self.num_labels,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_length=self.max_length,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # 训练循环
        best_val_acc = 0.0
        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                        f"Loss: {loss.item():.4f} - Acc: {100 * train_correct / train_total:.2f}%"
                    )

            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")

        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    def predict(self, X: List[str]) -> List[Any]:
        """预测类别"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        logger.info(f"Predicting {len(X)} samples...")
        self.model.eval()

        # 编码文本
        token_ids = self.tokenizer.encode(X)
        dataset = TextDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())

        # 解码标签
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions.tolist()

    def predict_proba(self, X: List[str]) -> List[List[float]]:
        """预测每个类别的概率"""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        logger.info(f"Predicting probabilities for {len(X)} samples...")
        self.model.eval()

        # 编码文本
        token_ids = self.tokenizer.encode(X)
        dataset = TextDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        probabilities = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        return probabilities

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("No model to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'tokenizer_vocab': self.tokenizer.token_to_idx,
            'label_encoder_classes': self.label_encoder.classes_,
            'num_labels': self.num_labels,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'max_length': self.max_length,
            'dropout': self.dropout,
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BERTTextClassifier":
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')

        # 创建模型实例
        instance = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            num_layers=checkpoint['num_layers'],
            num_heads=checkpoint['num_heads'],
            d_ff=checkpoint['d_ff'],
            max_length=checkpoint['max_length'],
            dropout=checkpoint['dropout'],
        )

        # 恢复tokenizer
        instance.tokenizer.token_to_idx = checkpoint['tokenizer_vocab']
        instance.tokenizer.vocab_built = True

        # 恢复标签编码器
        instance.label_encoder.classes_ = checkpoint['label_encoder_classes']
        instance.num_labels = checkpoint['num_labels']

        # 重建模型
        instance.model = BERTClassifier(
            vocab_size=len(instance.tokenizer.token_to_idx),
            num_labels=instance.num_labels,
            d_model=instance.d_model,
            num_layers=instance.num_layers,
            num_heads=instance.num_heads,
            d_ff=instance.d_ff,
            max_seq_length=instance.max_length,
            dropout=instance.dropout
        ).to(instance.device)

        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.model.eval()

        logger.info(f"Model loaded from {path}")
        return instance
