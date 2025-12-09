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

# 新的损失函数，适应类别不平衡
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        # 计算概率
        pt = torch.exp(-ce_loss)
        # Focal loss
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


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
        vocab_size: int = 7000, # 6000->7000
        d_model: int = 768,
        num_layers: int = 10, # 8->10
        num_heads: int = 12,
        d_ff: int = 3072, # 2048->3072
        max_length: int = 2560, # 1024->2560
        batch_size: int = 16,
        learning_rate: float = 1.5e-5,
        epochs: int = 20,
        dropout: float = 0.2,
        warmup_ratio: float = 0.15,
        label_smoothing: float = 0.1,
        gradient_accumulation_steps: int = 2,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        use_multi_gpu: bool = True,
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
        self.warmup_ratio = warmup_ratio
        self.label_smoothing = label_smoothing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_multi_gpu = use_multi_gpu

        # 设置设备和多GPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 检测可用GPU数量
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1 and self.use_multi_gpu:
            logger.info(f"Detected {self.num_gpus} GPUs, will use DataParallel for training")
            self.multi_gpu_enabled = True
        else:
            self.multi_gpu_enabled = False
            if self.num_gpus <= 1 and self.use_multi_gpu:
                logger.warning(f"use_multi_gpu=True but only {self.num_gpus} GPU available")

        self.tokenizer = SimpleTokenizer(vocab_size, max_length)
        self.model = None
        self.label_encoder = LabelEncoder()
        self.num_labels = None

        logger.info(f"Initialized BERT classifier with device: {self.device}, Multi-GPU: {self.multi_gpu_enabled}")

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
        model = BERTClassifier(
            vocab_size=len(self.tokenizer.token_to_idx),
            num_labels=self.num_labels,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_length=self.max_length,
            dropout=self.dropout
        ).to(self.device)

        # 多GPU包装
        if self.multi_gpu_enabled:
            logger.info(f"Wrapping model with DataParallel for {self.num_gpus} GPUs")
            self.model = nn.DataParallel(model, device_ids=list(range(self.num_gpus)))
            # 调整batch size以充分利用多GPU
            effective_batch_size = self.batch_size * self.num_gpus
            logger.info(f"Effective batch size with {self.num_gpus} GPUs: {effective_batch_size}")
        else:
            self.model = model

        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 计算总训练步数
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)

        # 学习率调度器：warmup + cosine decay
        def get_lr_multiplier(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)

        # 损失函数：Focal Loss或标签平滑交叉熵
        if self.use_focal_loss:
            criterion = FocalLoss(gamma=self.focal_gamma, label_smoothing=self.label_smoothing)
            logger.info(f"Using Focal Loss with gamma={self.focal_gamma}")
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        logger.info(f"Training config: warmup_steps={warmup_steps}, total_steps={total_steps}, gradient_accumulation={self.gradient_accumulation_steps}")

        # 训练循环
        best_val_acc = 0.0
        global_step = 0
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

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                train_loss += loss.item() * self.gradient_accumulation_steps
                _, predicted = torch.max(logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                if (batch_idx + 1) % (10 * self.gradient_accumulation_steps) == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                        f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f} - Acc: {100 * train_correct / train_total:.2f}% - LR: {current_lr:.2e}"
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

    def predict_proba(self, X: List[str], use_tta: bool = True, tta_rounds: int = 5) -> List[List[float]]:
        """预测每个类别的概率

        Args:
            X: 文本列表
            use_tta: 是否使用Test-Time Augmentation
            tta_rounds: TTA轮数（如果use_tta=True）
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        if use_tta:
            logger.info(f"Predicting probabilities for {len(X)} samples with TTA (rounds={tta_rounds})...")
            return self._predict_proba_with_tta(X, tta_rounds)

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

    def _predict_proba_with_tta(self, X: List[str], tta_rounds: int = 3) -> List[List[float]]:
        """使用Test-Time Augmentation进行预测

        通过多次dropout推理并平均，提升预测稳定性
        """
        self.model.train()  # 启用dropout

        # 编码文本
        token_ids = self.tokenizer.encode(X)
        dataset = TextDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # 多轮预测
        all_probs = []
        for round_idx in range(tta_rounds):
            round_probs = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    logits = self.model(input_ids, attention_mask)
                    probs = F.softmax(logits, dim=1)
                    round_probs.extend(probs.cpu().numpy())
            all_probs.append(round_probs)

        # 平均所有轮次的预测
        import numpy as np
        avg_probs = np.mean(all_probs, axis=0)

        self.model.eval()  # 恢复eval模式
        return avg_probs.tolist()

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("No model to save")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果是多GPU模型，获取原始模型
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,
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
            'warmup_ratio': self.warmup_ratio,
            'label_smoothing': self.label_smoothing,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'use_focal_loss': self.use_focal_loss,
            'focal_gamma': self.focal_gamma,
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BERTTextClassifier":
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # 创建模型实例
        instance = cls(
            vocab_size=checkpoint.get('vocab_size', 7000),
            d_model=checkpoint.get('d_model', 768),
            num_layers=checkpoint.get('num_layers', 10),
            num_heads=checkpoint.get('num_heads', 12),
            d_ff=checkpoint.get('d_ff', 3072),
            max_length=checkpoint.get('max_length', 1024),
            dropout=checkpoint.get('dropout', 0.15),
            warmup_ratio=checkpoint.get('warmup_ratio', 0.1),
            label_smoothing=checkpoint.get('label_smoothing', 0.1),
            gradient_accumulation_steps=checkpoint.get('gradient_accumulation_steps', 2),
            use_focal_loss=checkpoint.get('use_focal_loss', True),
            focal_gamma=checkpoint.get('focal_gamma', 2.0),
            use_multi_gpu=False  # 加载时不启用多GPU
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

        # 加载state_dict，处理可能的DataParallel包装
        state_dict = checkpoint['model_state_dict']
        # 如果state_dict中有"module."前缀，移除它
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        logger.info(f"Model loaded from {path}")
        return instance
