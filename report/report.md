# tianchi-nlp-news-classification report

## 问题介绍

## 数据处理

## 模型 && 训练

本次实验中采取若干种模型结构进行训练，并在部分模型上进一步调优。在选择模型调优之前，需要先对比几种模型训练结果的基准表现，选取较为优秀、有进步空间的模型进行调整。

### 训练textCNN模型

#### textcnn方案

#### 训练参数

```bash
python main.py train --model-spec textcnn --epochs 15 --batch-size 128 --learning-rate 1.5e-4 --dataloader-num-workers 4 --model-out models/textcnn.pt
```

#### 训练过程

![alt text](fig/cnn-train.png)

![alt text](fig/cnn-traincomplete.png)

#### 结果

得分92.98。

### 训练transformer模型

transformer 模型广泛的运用在各种模型中，可以处理自然语言等顺序输入数据，适用于机器翻译、文本摘要等任务。

#### transformer方案

##### 模型

* **Embeddings**：输入的单词（或 token）转换成数字向量（比如 "猫" → [0.2, -0.5, 0.7…]）。
* **encoder**

  * **Multi-Headed Self-Attention（多头自注意力）**：让模型同时关注输入中的所有单词，并计算它们之间的关系。
  * **Norm（层归一化）**：稳定训练过程，防止数值过大或过小（类似"调音量"到合适范围）。
  * **Feed-Forward Network（前馈神经网络）**：对每个单词的表示进行进一步加工（比如提取更复杂的特征）。
* **decoder(X)**
* **masked mean pooling**

  * 把 `[B, L, D]` 的 token 表示，用 mask 把 PAD 位置的向量清零，按 mask 对非 PAD 位置做平均，
    得到一个 `[B, D]` 的句向量，作为这条样本的整体表示。
* **classifier MLP**

  * 最终输出分类结果(softmax后的argmax)

下图为完整的transformer，此处的transformer去除了decoder。

![alt text](fig/Transformer,_full_architecture.png)

#### 结果

得分91.58。

### 训练bert模型

#### bert方案

选择bert模型的原因是其结构广泛的用于分类任务中，可以支撑大数据量、长输入的训练。

##### 模型

bert 模型具有以下架构：

* 嵌入层 **Embedding** 后加正弦/余弦 **PositionalEncoding**，形成 **[batch, seq_len, d_model]** 输入。
* 堆叠 **num_layers** 个 Transformer 编码层：每层是多头自注意力 (**MultiHeadAttention**) + 前馈两层全连接 (**FeedForward**)，各自带残差和 **LayerNorm**。
* 注意力mask来自padding（**attention_mask**），避免对pad位置计算权重。
* 分类头取 **[CLS]** 位置向量，经 **tanh(pooler)** + dropout + 全连接输出 **num_labels** 维 logits。
* 最后通过取 softmax 得到输出概率列表。

![alt text](fig/BERT_masked_language_modelling_task.png)

##### 动态学习率

```python
# Warmup (前15%步数线性增长)
if step < warmup_steps:
    lr = base_lr * (step / warmup_steps)

# Cosine Decay (后续余弦衰减)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = base_lr * 0.5 * (1 + cos(π * progress))
```

##### Focal Loss

```python
# 标准交叉熵: 所有样本权重相同
loss = -log(p_t)

# Focal Loss: 难样本权重更大
loss = -(1 - p_t)^γ * log(p_t)

# 当p_t高(易分类): (1-p_t)小，loss被降权
# 当p_t低(难分类): (1-p_t)大，loss被加权
```

实现如下：

```python
# 计算交叉熵
ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
# 计算概率
pt = torch.exp(-ce_loss)
# Focal loss
focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
```

##### 多轮推理(TTA)

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

#### 训练参数与调参

```python
vocab_size = 7000              # 更大词汇表覆盖 (6000->7000->7000)
d_model = 768                  # BERT-base级别 (512->768->768)
num_layers = 10                # 更深 (6->8->10)
num_heads = 12                 # 更多注意力 (8->10->12)
d_ff = 3072                    # 4倍模型维度 (2048->3072->3072)
max_length = 2560              # 覆盖95%样本 (512->1024->2560)
dropout = 0.2                  # 优化过拟合 (0->0.1->0.2)
batch_size = 16                # 配合梯度累积
learning_rate = 1.2e-5         # 更稳定 (1e-4->2e-5->1.2e-5)
epochs = 20                    # 充分训练 (10->15->20)
warmup_ratio = 0.15            # warmup步数 (0->0.1->0.15)
label_smoothing = 0.1          # 标签平滑 (0->0.1->0.1)
gradient_accumulation = 2      # 有效batch=32 (0->2->2)
```

* 参数调整

注释中标记了3次主要训练中的参数调节过程，总体变化趋势是参数量逐渐增大，实际上进行了不止3次训练，但报告中选取能造成表现较显著进步的训练参数进行调参报告。

* 第一次训练参数较为保守，同时没有对 loss, label, infer, lr 进行优化改造，所以不少参数还是0。
* 第一次训练测试结果为0.91。
* 第二次训练在采取优化后进行训练，同时最大的变化是增加模型深度和注意力头数，并降低学习率，启用动态学习率，推测这3项变化对测试结果的提升最为明显。
* 第二次训练测试结果为0.95。

#### 训练过程(第三次训练)

![alt text](fig/bert-train-start.png)

### 从bertbase模型继续训练

#### 方案

典型的bert模型需要经过预训练和下游微调两个阶段。也就是在 HF 中加载预训练的 bert，再进行微调分类头。本次实验中同样测试了这种方案。

这种方法实际执行下效果非常差，准确率只有0.3。可以猜测原因如下：

##### BERT 对数字密文其实毫无先验，微调收益非常有限

* 预训练模型：`bert-base-chinese`
* 输入： **经过加密的数字 token** ，和自然语言完全没关系

从 BERT 视角看：预训练的 embedding 和 encoder 层里语义知识的几乎派不上用场；且在大量位置上还需要用随机初始化的新 token embedding；这本质上是在用一个带着一堆“无关先验”的网络去学习一堆随机编号序列上的分类问题。

所以才会导致收敛更难，初始阶段模型对这些 token 的表示是纯随机的，需要大量数据 + 训练步数才能把 embedding 和高层适配好。同时内容加密其实也把 BERT 能利用的语义信息全抹掉了。
