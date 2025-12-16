# tianchi-nlp-news-classification report

## 1 问题介绍

通过不同的机器学习模型对已经匿名化的新闻文本进行分类，并根据训练结果预测文本的新闻所属类型，从而实现新闻的自动分类

数据共分为三部分：20 万条训练集样本，以及各 5 万条样本的测试集 A、测试集 B

## 2 数据处理

* **train.py**:从指定 CSV 文件加载文本分类数据集，自动拆分训练 / 验证集；
* **data_utils.py**:
  * 自动加载 TSV（制表符分隔）格式的数据集，智能检测文本列、标签列；
  * 提供固定的类别名称与数字 ID 映射（如 “科技”→0、“股票”→1 等 14 个类别）；
  * 支持标签 ID 与名称的双向转换，以及预测结果的 CSV 保存。
* **机器学习模型**:通过 SimpleTokenizer（数字分词器）实现将空格分隔的数字序列文本转换成固定长度的 token ID。

操作后数据便转换为模型所需的Token了。

### 2.1 数据分析

| 标签ID | 样本数 |   占比 |
| ------ | -----: | -----: |
| 0      |  38918 | 19.46% |
| 1      |  36945 | 18.47% |
| 2      |  31425 | 15.71% |
| 3      |  22133 | 11.07% |
| 4      |  15016 |  7.51% |
| 5      |  12232 |  6.12% |
| 6      |   9985 |  4.99% |
| 7      |   8841 |  4.42% |
| 8      |   7847 |  3.92% |
| 9      |   5878 |  2.94% |
| 10     |   4920 |  2.46% |
| 11     |   3131 |  1.57% |
| 12     |   1821 |  0.91% |
| 13     |    908 |  0.45% |

| 指标    | Token长度 |
| ------- | --------: |
| 最小值  |         2 |
| 均值    |       907 |
| 中位数  |       676 |
| 75 分位 |      1131 |
| 90 分位 |      1796 |
| 95 分位 |      2457 |
| 99 分位 |      4228 |
| 最大值  |     57921 |

1. **词表与分词**：文本是匿名数字，可能需要用单独设计分词器统计高频 token 并固定 ID，才能让 TextCNN、BERT 与微调流程共享同一输入空间。
2. **长序列策略**：95% 样本超过 2400 token，1% 甚至超 4000，因此模型可能需要采用 2560 的 `max_length` 并辅以滑窗（stride）或 head-tail 拼接避免信息丢失。
3. **类别不平衡**：标签 0/1/2 合计近 53%，而 12/13 不足 1%。训练时可能需要使用 FocalLoss、类别权重、TTA+集成来维持小类召回率，并在 TextCNN/Transformer 阶段选取合适的损失函数。
4. **训练 / 验证划分策略**：由于比赛只提供单一训练集，我们在 `train.py` 中统一做 9:1 的分层拆分，保证每个标签在验证集中保持原始占比。当然还有一种想法是每一个类别都抽取相同样本数的数据做拆分，两种方式都可以尝试。

### 2.2 SimpleTokenizer

一个样本从“字符串”到“张量”大致经历以下步骤：

1. **切分得到 token 序列**：SimpleTokenizer 直接按空格分隔，将 `"3750 648 900 ..."` 变成 `['3750','648','900',…]`。
2. **映射到词表 ID**：查词表把每个 token 变成整数 ID；若未在词表中出现，就映射到 `<UNK>`。BERT 系列会在首尾追加 `[CLS]`、`[SEP]` 的 ID。
3. **长度对齐**：依据模型设定的 `max_length` 进行截断或补 `<PAD>`。同时生成 `attention_mask`（1 表示真实 token，0 表示 PAD），以及 `token_type_ids`（单句任务固定为 0）。
4. **打包成张量**：批量样本被堆叠成 `input_ids: [B, L]`、`attention_mask: [B, L]`、`token_type_ids: [B, L]`，并与标签张量一起交给模型/Trainer。
5. **标签同步编码**：所有模型共享 `label_to_id` 映射，把原始标签（如 "科技" 或数字字符串）编码成连续整数，预测阶段再映射回去，保证不同模型输出一致。

此后，匿名化的数字串被稳定地转成定长、带掩码的输入张量，能喂给 TextCNN/Transformer/BERT，也能无缝对接 HuggingFace 的 `BertForSequenceClassification`。

## 3 模型 && 训练

本次实验中采取若干种模型结构进行训练，并在部分模型上进一步调优。在选择模型调优之前，需要先对比几种模型训练结果的基准表现，选取较为优秀、有进步空间的模型进行调整。

### 3.1 训练textCNN模型

#### 3.1.1 textcnn方案

##### 模型

* **SimpleTokenizer（数字分词器）**：将空格分隔的数字序列文本，先统计高频数字 token 构建词表（给每个数字分配唯一 ID，预留 PAD/UNK），再把文本转换成固定长度的 token ID 序列（超长截断、超短补 PAD）。
* **TextDataset（文本数据集）**：封装 token ID 序列和标签，按索引返回单条样本（包含 input_ids 和可选 labels），适配 PyTorch 数据加载逻辑。
* **textcnn_model**:

  * **嵌入层（Embedding）**：把 token ID 转换为稠密向量（embedding_dim 维度），忽略 `<PAD>`的向量更新；
  * **卷积层（Conv1d）**：多尺寸卷积核（如 2/3/4/5）提取不同长度的文本特征（类似 N-gram）；
  * **池化层（MaxPool1d）**：对卷积结果取最大值，保留关键特征；
  * **分类层（Linear）**：拼接所有卷积核的池化结果，映射到类别数维度。

下图为TextCNN模型的示意图，从左至右依次为嵌入层、卷积层、池化层与分类层

![textcnn.png](fig/textcnn.png)

* **FocalLoss（聚焦损失）**：基于交叉熵损失，通过 (1-pt)^gamma 调制因子，降低易分类样本权重、聚焦难分类样本（解决类别不平衡）。
* **TextCNNModel（模型适配器）**：
  * **_select_hyperparams（动态调参）**：根据训练集文本长度（95% 分位数）调整 max_length，按词表规模调整 embedding_dim，按文本中位数长度调整卷积核尺寸。
  * **fit（训练流程）**：统计文本长度 / 构建词表→编码标签→划分训练 / 验证集→初始化模型 / 优化器 / 调度器→根据类别不平衡比选择损失函数→训练循环（前向计算损失、反向传播更新参数、梯度累积、验证集评估）。
  * **_evaluate（评估函数）**：无梯度计算模式下，遍历验证集计算损失和分类精度。predict（预测）：将文本编码为 token ID→模型输出 logits→取 argmax 得到类别 ID→解码为原始类别名。
  * **predict_proba（概率预测）**：模型输出 logits 后用 softmax 归一化，得到每个类别的概率值。
  * **save/load（模型保存 / 加载）**：保存模型参数、分词器词表、标签编码器、超参数配置；加载时恢复所有状态，适配多 GPU / 单 GPU 环境。

#### 3.1.2 训练参数

```bash
python main.py train --model-spec textcnn --epochs 15 --batch-size 128 --learning-rate 1.5e-4 --dataloader-num-workers 4 --model-out models/textcnn.pt
```

#### 3.1.3 训练过程

![alt text](fig/cnn-train.png)

![alt text](fig/cnn-traincomplete.png)

#### 3.1.4 结果

得分92.98。

### 3.2 训练transformer模型

transformer 模型广泛的运用在各种模型中，可以处理自然语言等顺序输入数据，适用于机器翻译、文本摘要等任务。

#### 3.2.1 transformer方案

##### 模型

* **Embeddings**：输入的单词（或 token）转换成数字向量（比如 "猫" → [0.2, -0.5, 0.7…]）。
* **encoder**

  * **Multi-Headed Self-Attention（多头自注意力）**：让模型同时关注输入中的所有单词，并计算它们之间的关系。
  * **Norm（层归一化）**：稳定训练过程，防止数值过大或过小（类似"调音量"到合适范围）。
  * **Feed-Forward Network（前馈神经网络）**：对每个单词的表示进行进一步加工（比如提取更复杂的特征）。
* **decoder(X) 删除**
* **masked mean pooling**

  * 把 `[B, L, D]` 的 token 表示，用 mask 把 PAD 位置的向量清零，按 mask 对非 PAD 位置做平均，
    得到一个 `[B, D]` 的句向量，作为这条样本的整体表示。
* **classifier MLP**

  * 最终输出分类结果(softmax后的argmax)

下图为完整的transformer，此处的transformer去除了decoder，因为在分类任务中如果保留decoder和自回归性质进行prefill的话反而容易引起不必要的麻烦，对于分类来说是额外复杂度和算力开销，而难以提升准确率。

**（这个改造过的模型实际上已经和 BERT 模型很接近了，所以这里考虑使用更成熟的 BERT 模型为基准，不再对transformer进行改造）**

![alt text](fig/Transformer,_full_architecture.png)

#### 3.2.2 训练参数

此处的训练参数是测试性质的，没有发挥其全部威力。

```python
vocab_size: int = 8000
d_model: int = 256,
nhead: int = 8,
num_layers: int = 4,
dim_feedforward: int = 1024,
dropout: float = 0.1,
max_length: int = 512
```

#### 3.2.3 结果

得分91.58。

### 3.3 训练 BERT 模型

#### 3.3.1 BERT 方案

选择 BERT 模型的原因是其结构广泛的用于分类任务中，可以支撑大数据量、长输入的训练。

##### 模型

BERT 模型具有以下架构：

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

Warmup 整体趋势如下图所示。橙色区域表示 15% 的线性 warmup，确保训练前期稳定；绿色区域表示余弦衰减段，确保后期细致收敛。

![动态学习率调度](fig/dynamic_lr_schedule.png)

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

难样本调制因子对损失曲线的影响如下图所示，横轴是模型对真实类别的置信度 $p_t$，纵轴是损失值。相比标准交叉熵（蓝色实线），Focal Loss 会在 $p_t$ 较大时迅速压低损失，$\gamma$ 越大这一效果越明显，从而把梯度主要留给低置信度、难分类的样本。

![Focal Loss 曲线](fig/focal_loss_curve.png)

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

##### 梯度累积

```python
# 实际batch=16，累积2步
# 等效batch=32，减少内存占用
for step in range(0, len(data), 16):
    loss = model(batch) / 2
    loss.backward()
    if (step + 1) % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

##### 标签平滑

原始: [0, 0, 1, 0, 0]

平滑: [0.007, 0.007, 0.964, 0.007, 0.007]

效果: 减少过拟合，提升泛化

##### multi gpu 支持

由于实验中有一些模型训练配置参数量很大，训练中配置了数据并行和模型并行处理，可以为大型的模型训练避免OOM。

#### 3.3.2 训练参数与调参

| 参数                     | 最终值 | 调整轨迹 / 理由                                                     |
| ------------------------ | -----: | ------------------------------------------------------------------- |
| 词表大小 `vocab_size`  |   7000 | 逐步扩大（6000→7000）以覆盖更多高频数字 token，减少 `<UNK>` 比例 |
| 隐层维度 `d_model`     |    768 | 从 512 提升到 BERT-base 规格，匹配更深网络容量                      |
| 编码层数 `num_layers`  |     10 | 6→8→10，增加表达能力以适应长序列模式                              |
| 注意力头数 `num_heads` |     12 | 8→10→12，提升不同子空间的关注能力                                 |
| 前馈维度 `d_ff`        |   3072 | 保持 4×d_model 的经典配置，提供更强非线性投影                      |
| 最大长度 `max_length`  |   2560 | 512→1024→2560，覆盖 95%+ 样本长度，配合滑窗减少截断损失           |
| Dropout                  |    0.2 | 从 0→0.1→0.2，缓解深层模型过拟合                                  |
| Batch size               |     16 | 配合梯度累积步数 2，实现等效 32 的大批次稳定性                      |
| 学习率                   | 1.2e-5 | 1e-4→2e-5→1.2e-5，结合 warmup + cosine，训练更平滑                |
| Epochs                   |     20 | 10→15→20，保留更充分的收敛时间                                    |
| Warmup ratio             |   0.15 | 0→0.1→0.15，避免初期震荡                                          |
| Label smoothing          |    0.1 | 0→0.1，稳定 logits、提高泛化                                       |
| Gradient accumulation    |      2 | 0→2，减少显存占用且保持高有效 batch                                |

逐轮调参的核心经验是“先扩容表达能力，再精细化训练策略”。保守的首轮配置在未做损失/推理优化的情况下仅 0.91；第二轮通过加深网络、增加注意力头并启用动态学习率，准确率跃升至 0.94；第三轮重点拉高 `max_length` 与训练轮次，准确率进一步提升到 0.95。所有实验都在 2×RTX 4090（约 10 小时/轮）上完成。

#### 3.3.3 训练过程

![alt text](fig/bert-train-start.png)
![alt text](fig/bert-train-finish.png)

#### 3.3.4 训练结果

最优为95.05。

### 3.4 从 BERT-base 模型继续训练

#### 3.4.1 方案

典型的 BERT 模型需要经过预训练和下游微调两个阶段。也就是在 HF 中加载预训练的 BERT，再进行微调分类头。本次实验中同样测试了这种方案。

这种方法实际执行下效果非常差，若干轮训练后准确率只有0.3，远远不如直接训练的收敛速度。可以猜测原因如下：

##### BERT 对数字密文其实毫无先验，微调收益非常有限

* 预训练模型：`bert-base-chinese`
* 输入： **经过加密的数字 token** ，和自然语言完全没关系

从 BERT 视角看：预训练的 embedding 和 encoder 层里语义知识的几乎派不上用场；且在大量位置上还需要用随机初始化的新 token embedding；这本质上是在用一个带着一堆“无关先验”的网络去学习一堆随机编号序列上的分类问题。

所以才会导致收敛更难，初始阶段模型对这些 token 的表示是纯随机的，需要大量数据 + 训练步数才能把 embedding 和高层适配好。同时内容加密其实也把 BERT 能利用的语义信息全抹掉了。

### 3.5 从 BERT 模型集成学习

#### 3.5.1 方案

有两种方法得到集成学习的结果，方法一是 Boosting，串行训练若干个模型并加权组合，可以指定模型类型为集成学习的模型(bert_ensemble)：

```bash
python main.py train --model-spec bert_ensemble --epochs 20 --batch-size 8 --learning-rate 1.5e-5 --model-out models/bert_ens.pt
```

```mermaid
graph LR
  D[训练数据集 D] --> W0[初始化权重/残差]
  W0 --> M1[弱模型1]
  M1 --> E1[计算误差]
  E1 --> W1[更新权重/残差]
  W1 --> M2[弱模型2]
  M2 --> E2[计算误差]
  E2 --> W2[更新权重/残差]
  W2 --> MK[重复迭代到第K个弱模型]

  M1 --> C[加权组合]
  M2 --> C
  MK --> C
  C --> Y[最终预测]

```

方法二是 Bagging，进行并行训练 + 平均/投票：

```bash
python infer_bert_ensemble.py \
  --models models/bert_a.pt models/bert_b.pt models/bert_c.pt \
  --input-csv data/test_a.csv \
  --output-csv predictions.csv \
  --use-tta --tta-rounds 5   # 可选
```

```mermaid
graph LR
  D[训练数据集 D] --> B1[Bootstrap重采样/随机子特征 1]
  D --> B2[Bootstrap重采样/随机子特征 2]
  D --> B3[Bootstrap重采样/随机子特征 3]
  D --> BK[Bootstrap重采样/随机子特征 K]

  B1 --> M1[模型1]
  B2 --> M2[模型2]
  B3 --> M3[模型3]
  BK --> MK[模型K]

  M1 --> A[聚合: 平均/投票]
  M2 --> A
  M3 --> A
  MK --> A

  A --> Y[最终预测]
```

在实际操作时，受限于显存的大小与训练成本，最终没有选择串行训练的方法，但代码展示了实现方法。而是直接通过 BERT 方案中训练的 3 个模型进行并行推理，同时需要说明的是，这里并没有采用典型的 Bagging 进行重采样选取特征等操作(又要重新训，太贵了😢)，也就是每个模型都是单独用一套训练集训练的，但仍然取得了一些提升。

事实上，这里也可以用不同的模型类型进行集成学习，不限于都适用 BERT 模型。

#### 3.5.2 结果

获得了最佳结果，准确率在 0.9557，相较于 3 个模型单独推理提升了 0.5%～1%，符合预期。（显示成绩结果13/0.96）

#### 3.5.3 结果分析

实际上，从 3 个模型单独的推理结果进行数据分析，发现两两之间不同结果数的占比能达到3%。

这意味着虽然这 3 个模型都是通过 BERT 模型架构、相同的数据训练出来的，实际上在推理中的结果却并不会收敛到相同的值。由于这 3 份推理结果的错误率均只在 5% 左右，此处不同的推理结果极有可能造成并行推理带来提升，目前的提升是在预期之内的，也有可能通过不同的概率组合达到更高的提升。

|                                         | 不同的行数 | 总行数 | 不同占比 |
| --------------------------------------- | ---------: | -----: | -------: |
| `predictions_1` vs `predictions_3` |       1620 |  50000 |    3.24% |
| `predictions_1` vs `predictions-2`  |       1691 |  50000 |   3.382% |
| `predictions_3` vs `predictions-2` |       1624 |  50000 |   3.248% |

### 3.6 loss 函数讨论

对于分类任务的 loss 函数主要选用交叉熵，部分模型针对不平衡样本分布在交叉熵的基础上采用了一些技术方法(3.3.1)，但仍然还是基于交叉熵的，下面对交叉熵函数的原理进行简单介绍。

#### 3.6.1 交叉熵

模型输出类别分布$q_θ(y∣x)$。
真实标签对应的分布是 one-hot 的$p(y∣x)$。

最小化交叉熵(one-hot 消掉了求和号)：

$$
H(p,q)=−y∑p(y∣x)\log q_θ(y∣x)=−\log⁡ q_θ(y_{true}∣x)
$$

就是让模型**把真实类别的概率变大**，因为让这个值最小的方法只能是把真实类别的预测概率往 1 推 → 自然得到分类能力。

回归任务与分类任务不同的是预测连续变量的概率分布，但最小化交叉熵也可以使得分布更靠近真实分布，和分类任务是相似的。

## 4 相关示例

### 4.1 推理与输出示例

本部分采用对单条文本进行推理，输出其预测结果和 score。

![alt text](fig/infer_example.png)

### 4.2 log 文件

`train_bert_log_example.log` 为训练所产生的 log 文件，但这个文件并不是第 3 节中产生的较优解相对应的 log 文件，为了在单 gpu 上完成训练所采取的参数都进行了减弱，所以最终准确率并没有那么高。

## 5 代码结构概览

```mermaid
graph TD
  main[main.py CLI] -->|train 子命令| train_entry[train.py]
  main -->|infer 子命令| infer_entry[infer.py]

  train_entry --> data_utils[data_utils.py\nload_data]
  train_entry --> registry[models/registry.py]
  registry --> base[models/base.py]

  base --> textcnn[models/textcnn_model.py]
  base --> transformer[models/transformer_model.py]
  base --> bert_local[models/bert_model.py]
  base --> bert_hf[models/bert_finetune.py]
  base --> ensemble_model[models/bert_ensemble.py]

  train_entry --> outputs[模型保存 / 日志]
  infer_entry --> data_utils
  infer_entry --> registry
  infer_entry --> ensemble_infer[infer_bert_ensemble.py]

  bert_hf --> hf_trainer[HuggingFace Trainer]
  bert_local --> pytorch_stack[PyTorch BERT Stack]
```

## 6 比赛排名

截止 12 月 12 日的排名如下图所示。

![alt text](fig/2025-12-11_19.56.22.png)

## 7 团队成员分工

该仓库在 github 上链接为 [https://github.com/HIJII-ZHANG/tianchi-nlp-news-classification](https://github.com/HIJII-ZHANG/tianchi-nlp-news-classification)，详细贡献可以查看commit。大致分工如下：

(HIJII-ZHANG) 负责主要代码编写、运行训练、提交结果和撰写报告。

(GuoZhikang2007) 修复了一些 bug，并尝试进行优化和撰写报告。
