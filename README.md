这是2025天池NLP新闻分类学习赛仓库

准备：

需要将比赛官方提供的训练和测试csv放入data文件夹下。

uv 包管理器安装：[https://uv.doczh.com/getting-started/installation/](https://uv.doczh.com/getting-started/installation/)

训练：

```bash
# 使用默认数据路径 data/train_set.csv，训练sklearn所配置的模型
uv run main.py train --train-csv data/train_set.csv --model-out models/model.joblib
# 训练torch下的transformer/bert模型
uv run main.py train --model-spec transformer --nrows 1000 --model-out models/model_transformer.pt --epochs 3 --batch-size 16
```

说明：

`nrows`：会选择训练数据文件中的一部分训练

推理（对 CSV 批量预测）：

```bash
python main.py infer --model models/model_bert.pt --model-type bert --input-csv data/test_a.csv --output-csv predictions.csv
```

说明：

`model`：参数文件路径
`model-type`：bert/transformer/sklearn 可以不指定，会进行自主尝试加载。
