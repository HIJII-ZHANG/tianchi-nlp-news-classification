
### 1. 更深网络
- **层数**: 8层 → **10层**
- **参数**: 40M → **50M**
- 更强的表达能力

### 2. Focal Loss
```python
# 自动处理类别不平衡
# 聚焦难分类样本
FocalLoss(gamma=2.0)
```
- 解决14类不平衡问题
- 对困难样本给予更多关注

### 3. 优化超参数
- **词汇**: 6000 → **7000**
- **学习率**: 2e-5 → **1.5e-5** (更稳定)
- **Dropout**: 0.15 → **0.2** (更强正则)
- **Warmup**: 10% → **15%** (更平滑)
- **Epochs**: 15 → **20** (更充分)

### 5. Test-Time Augmentation (TTA) (+0.3-0.5%)
```python
# 推理时启用
model.predict_proba(X, use_tta=True, tta_rounds=5)
```
- 多次dropout推理并平均
- 提升预测稳定性

## 🚀 快速开始

### 方式1: 使用脚本（推荐）

```bash
chmod +x train_97_percent.sh
./train_97_percent.sh
```

### 方式2: 直接命令

**冲刺版** (6-8小时，150K样本):
```bash
python main.py train \
    --model-spec bert \
    --nrows 150000 \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1.5e-5 \
    --model-out models/model_bert_97.pt
```

**终极版** (12-16小时，全部200K):
```bash
python main.py train \
    --model-spec bert \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1.5e-5 \
    --model-out models/model_bert_97_ultimate.pt
```

## 🔧 技术细节

### Focal Loss原理
```python
# 标准交叉熵: 所有样本权重相同
loss = -log(p_t)

# Focal Loss: 难样本权重更大
loss = -(1 - p_t)^γ * log(p_t)

# 当p_t高(易分类): (1-p_t)小，loss被降权
# 当p_t低(难分类): (1-p_t)大，loss被加权
```

### TTA (Test-Time Augmentation)
```python
# 启用dropout进行多次推理
model.train()  # 启用dropout
predictions = []
for _ in range(5):
    pred = model(x)
    predictions.append(pred)

# 平均预测
final_pred = mean(predictions)
```

## 💡 使用TTA提升推理

修改 `infer.py` 或直接调用：

```python
from models.bert_model import BERTTextClassifier

# 加载模型
model = BERTTextClassifier.load("models/model_bert_97.pt")

# 标准预测
preds = model.predict(test_texts)

# 使用TTA预测 (更准确但慢3-5倍)
probs = model.predict_proba(test_texts, use_tta=True, tta_rounds=5)
preds = label_encoder.inverse_transform(probs.argmax(axis=1))
```

## 推理命令

```bash
# 标准推理
python main.py infer \
    --model models/model_bert_97.pt \
    --model-type bert \
    --input-csv data/test_a.csv \
    --output-csv predictions.csv
```

## ⚙️ 如果还想更高

### 1. 集成学习 (+0.5-1%)
```bash
# 训练3个模型
for i in 1 2 3; do
    python main.py train --model-spec bert --model-out models/bert_$i.pt
done

# 投票或平均（需要自己实现）
```

### 3. 更大模型 (+0.3-0.5%)
```python
# 在bert_model.py中修改
d_model = 1024      # BERT-large级别
num_layers = 12     # 更深
num_heads = 16      # 更多头
```
⚠️ 但会慢很多，需要更强GPU

### 4. 预训练+微调 (+1-2%)
- 使用相关领域的预训练BERT
- 然后在本任务上微调


```bash
# 冲刺版 (6-8小时)
python main.py train --model-spec bert --nrows 150000 --epochs 20 --batch-size 16 --learning-rate 1.5e-5 --model-out models/model_bert_97.pt
```


```python
# 在代码中启用TTA
predict_proba(texts, use_tta=True, tta_rounds=5)
```
