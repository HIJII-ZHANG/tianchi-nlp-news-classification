from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseModel(ABC):
    """
    自定义模型的基类接口。
    需要求实现 fit/predict/predict_proba/save/load 方法。
    """

    @abstractmethod
    def fit(self, X: List[str], y: List[Any], **kwargs) -> None:
        """
        训练模型。
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: List[str]) -> List[Any]:
        """
        使用模型进行预测。
        """
        raise NotImplementedError()

    def predict_proba(self, X: List[str]) -> Optional[List[float]]:
        """
        获取每个类的预测概率。如果模型不支持概率预测，则返回None。
        """
        return None

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型到指定路径。
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        从指定路径加载模型。
        """
        raise NotImplementedError()
