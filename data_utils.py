from typing import List, Tuple, Optional
import logging
import pandas as pd


COMMON_TEXT_COLS = ["text", "content", "article", "sentence", "title", "words"]
COMMON_LABEL_COLS = ["label", "class", "category", "cat"]


logger = logging.getLogger(__name__)

# Provided label mapping for the competition: name -> id
NAME_TO_ID = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load a TSV (tab-separated) file into a DataFrame.

    Competition data is tab-separated. On error we log and return an empty
    DataFrame so callers can decide how to proceed.
    """
    try:
        return pd.read_csv(path, nrows=nrows, sep='\t')
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", path, e)
        return pd.DataFrame()


def load_data(path: str, nrows: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    将数据集加载到文本和标签列表中，并自动检测文本和标签列。

    输入参数:
    - path: 数据集文件路径
    - nrows: 读取的行数（用于快速测试）

    输出参数:
    - texts: 文本内容列表
    - labels: 标签列表(str)
    """
    df = load_csv(path, nrows=nrows)
    if df.empty:
        logger.error("Empty or unreadable DataFrame from %s", path)
        return [], []

    text_col = _find_column(list(df.columns), COMMON_TEXT_COLS)
    label_col = _find_column(list(df.columns), COMMON_LABEL_COLS)

    if text_col is None:
        logger.error("No text-like column found in %s", path)
        return [], []

    texts: List[str] = list(map(str, df[text_col].tolist()))

    if label_col is None or label_col not in df.columns:
        logger.error("Label column not found in %s; returning empty labels", path)
        labels: List[str] = []
    else:
        labels = list(map(str, df[label_col].tolist()))

    return texts, labels


def save_predictions(df: pd.DataFrame, path: str) -> None:
    """
    保存预测结果到CSV文件。
    """
    df.to_csv(path, index=False)


def id_to_name(label_id: int) -> Optional[str]:
    """
    将数字标签ID转换为类别名称。如果未知则返回None。
    """
    return ID_TO_NAME.get(label_id)


def name_to_id(name: str) -> Optional[int]:
    """
    将类别名称转换为数字ID。如果未知则返回None。
    """
    return NAME_TO_ID.get(name)
