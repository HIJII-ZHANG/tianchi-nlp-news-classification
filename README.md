这是2025天池NLP新闻分类学习赛仓库

准备：

需要将比赛官方提供的训练和测试csv放入data文件夹下。

uv 包管理器安装：[https://uv.doczh.com/getting-started/installation/](https://uv.doczh.com/getting-started/installation/)

训练：

```bash
# 使用默认数据路径 data/train_set.csv
uv run main.py train --train-csv data/train_set.csv --model-out models/model.joblib
```

推理（单条文本）：

```bash
uv run main.py infer --text "这是一条待分类的新闻文本"
```

推理（对 CSV 批量预测）：

```bash
uv run main.py infer --input-csv data/test_a.csv --output-csv predictions.csv
```

可插拔模型说明：

- 默认使用 sklearn 后端（TF-IDF + LogisticRegression）。要使用不同的 model 实现，可通过 `--model-spec` 指定一个 class 路径，例如：

```bash
# 使用自定义类 my_models.py 中的 MyTransformerModel
uv run main.py train --model-spec my_models:MyTransformerModel --train-csv data/train_set.csv --model-out models/model_transformer.joblib
```

- `--model-spec` 支持：
	- 内置别名 `sklearn`（默认）
	- 全路径导入，如 `models.sklearn_model.SklearnModel` 或 `my_package.module:ClassName`

- 如果你要接入自定义的 Transformer 类，请实现一个适配器类满足下面的接口（示例见 `models/template_custom_model.py`）：
	- 方法： `fit(X: List[str], y: List[Any], **kwargs)`, `predict(X: List[str]) -> List[Any]`, `predict_proba(X) -> List[float] | None`, `save(path)`, `@classmethod load(path)`。
	- 训练/保存/加载逻辑由你的类负责，主程序只会调用这些通用方法。

示例（伪代码）:

```python
class MyTransformerAdapter:
		def fit(self, X, y, **kwargs):
				# 调用 transformers 的 Trainer 或自定义训练逻辑
				...
		def predict(self, X):
				...
		def save(self, path):
				...
		@classmethod
		def load(cls, path):
				...
```

然后在 CLI 使用 `--model-spec my_module:MyTransformerAdapter` 即可。
