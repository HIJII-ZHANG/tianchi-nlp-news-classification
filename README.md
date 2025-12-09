# 使用文档

## 准备

* 需要将比赛官方提供的训练和测试csv放入data文件夹下。
* uv 包管理器安装：[https://uv.doczh.com/getting-started/installation/](https://uv.doczh.com/getting-started/installation/)

  * 可以按照 `pyproject.toml` 中的依赖直接安装到环境以替代 uv 下载。

## 训练

在 `models/` 下提供 5 种模型的训练，从而探寻适应于该任务的更优架构。

### textcnn、transformer、bert

参数说明：

* `nrows`：仅读取CSV的前n行（用于快速测试）
* `model-spec`：模型名称，支持 `'sklearn', 'textcnn', 'transformer', 'bert', 'bert_finetune'`
* `model_out`: 训练完成后模型保存的路径
* `test_size`: 验证集占比
* `max_features`: 特征最大数量（仅对sklearn模型有效）
* `epochs`: 训练轮数
* `batch_size`: 批量大小
* `learning_rate`: 学习率

```bash
# 使用默认数据路径 data/train_set.csv，训练sklearn所配置的模型
uv run main.py train --train-csv data/train_set.csv --model-out models/model.joblib
# 训练torch下的transformer/bert模型
uv run main.py train --model-spec transformer --nrows 1000 --model-out models/model_transformer.pt --epochs 3 --batch-size 16
```

### 基于bert_base进行训练

```bash
# 先准备huggingface
huggingface-cli login
# 按提示粘贴你在 HF 网站上生成的 Access Token（scope 至少包含 read）
git clone https://huggingface.co/bert-base-chinese models/pretrained/bert-base-chinese
cd models/pretrained/bert-base-chinese
# 拉取 LFS 大文件
git lfs pull


#也可以使用python：
from huggingface_hub import snapshot_download
snapshot_download("bert-base-chinese", local_dir="models/pretrained/bert-base-chinese", local_dir_use_symlinks=False)

export HUGGINGFACE_HUB_TOKEN="hf_XXXXXXXXXXXXXXXXXXXX"
uv run <your_file_name>.py
```

下一步训练需要提供：

```bash
# 从bertbase微调训练
uv run main.py train \
  --train-csv data/train_set.csv \
  --model-out models/bert_finetuned \
  --model-spec bert_finetune \
  --nrows 2000 \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 2e-5
  --pretrained models/pretrained/bert-base-chinese
```

参数说明：

* `pretrained`：设置该参数意味着不从huggingface远程拉去代码，需要将值设置为模型所在目录

## 推理（对 CSV 批量预测）

```bash
python main.py infer --model models/model_bert.pt --model-type bert --input-csv data/test_a.csv --output-csv predictions.csv
```

说明：

`model`：参数文件路径
`model-type`：bert/transformer/sklearn 可以不指定，会进行自主尝试加载。
