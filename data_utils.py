from typing import List, Tuple, Optional, Union
from collections import Counter
import logging
import pandas as pd


COMMON_TEXT_COLS = ["text", "content", "article", "sentence", "title", "words"]
COMMON_LABEL_COLS = ["label", "class", "category", "cat"]


logger = logging.getLogger(__name__)

# Provided label mapping for the competition: name -> id
NAME_TO_ID = {'科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6, '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
ALL_LABEL_NAMES = list(NAME_TO_ID.keys())
ALL_LABEL_IDS = list(NAME_TO_ID.values())


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    从列名列表中查找目标列（不区分大小写）
    :param columns: 数据框列名列表
    :param candidates: 候选列名列表
    :return: 找到的列名（原始大小写），未找到返回None
    """
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def normalize_label(label: Union[str, int]) -> Tuple[Optional[int], Optional[str]]:
    """
    标准化标签：统一转换为 (label_id, label_name) 格式
    支持输入：标签名（中文）、标签ID（整数/字符串）
    :param label: 原始标签
    :return: (标准化ID, 标准化名称)，无法识别返回 (None, None)
    """
    if label is None or pd.isna(label):
        return None, None
    
    # 处理数字类型标签（包括字符串格式的数字）
    if isinstance(label, (int, float)) or (isinstance(label, str) and label.strip().isdigit()):
        label_id = int(label)
        if label_id in ID_TO_NAME:
            return label_id, ID_TO_NAME[label_id]
        else:
            logger.warning(f"Unknown label ID: {label_id}")
            return None, None
    
    # 处理字符串类型标签（中文名称）
    label_name = str(label).strip()
    if label_name in NAME_TO_ID:
        return NAME_TO_ID[label_name], label_name
    else:
        logger.warning(f"Unknown label name: {label_name}")
        return None, None



def load_csv(path: str, nrows: Optional[int] = None, sep: str = '\t', encoding: str = 'utf-8') -> pd.DataFrame:
    """Load a TSV (tab-separated) file into a DataFrame.

    Competition data is tab-separated. On error we log and return an empty
    DataFrame so callers can decide how to proceed.
    """
    encodings = [encoding, 'gbk', 'gb2312', 'utf-8-sig'] if encoding == 'utf-8' else [encoding]
    
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                nrows=nrows,
                sep=sep,
                encoding=enc,
                na_filter=True,  # 识别NA值
                keep_default_dates=False
            )
            logger.info(f"Successfully loaded file: {path} (encoding: {enc}, rows: {len(df)})")
            return df
        except Exception as e:
            logger.debug(f"Failed to read with encoding {enc}: {e}")
            continue
    
    logger.error(f"Failed to read file {path} with all tried encodings")
    return pd.DataFrame()


def load_data( path: str, nrows: Optional[int] = None, sep: str = '\t', normalize_label_flag: bool = True, drop_invalid: bool = True) -> Tuple[List[str], List[Optional[int]], List[Optional[str]]]:
    """
    将数据集加载到文本和标签列表中，并自动检测文本和标签列。

    输入参数:
    - path: 数据集文件路径
    - nrows: 读取的行数（用于快速测试）

    输出参数:
    - texts: 文本内容列表
    - labels: 标签列表(str)
    """
    df = load_csv(path, nrows=nrows, sep=sep)
    if df.empty:
        logger.error("Empty or unreadable DataFrame")
        return [], [], []
    
    # 自动检测文本列和标签列
    text_col = _find_column(list(df.columns), COMMON_TEXT_COLS)
    label_col = _find_column(list(df.columns), COMMON_LABEL_COLS)
    
    if text_col is None:
        logger.error("No text-like column found in columns: %s", list(df.columns))
        return [], [], []
    
    # 提取并处理文本
    texts = df[text_col].fillna("").tolist()
    
    # 提取并处理标签
    label_ids: List[int] = []
    label_names: List[str] = []
    
    if label_col is None:
        logger.warning("No label column found, returning empty labels")
        return texts, [], []
    else:
        raw_labels = df[label_col].tolist()
        for raw_label in raw_labels:
            if not normalize_label_flag:
                # 不标准化，直接转换为字符串
                label_id = None
                label_name = str(raw_label) if not pd.isna(raw_label) else ""
            else:
                label_id, label_name = normalize_label(raw_label)
            
            label_ids.append(label_id)
            label_names.append(label_name)
    
    # 过滤无效数据（无文本或无有效标签）
    if drop_invalid:
        valid_mask = []
        for text, label_id in zip(texts, label_ids):
            is_valid = len(text.strip()) > 0 and label_id is not None
            valid_mask.append(is_valid)
        
        texts = [text for text, valid in zip(texts, valid_mask) if valid]
        label_ids = [lid for lid, valid in zip(label_ids, valid_mask) if valid]
        label_names = [ln for ln, valid in zip(label_names, valid_mask) if valid]
        
        logger.info(f"Filtered invalid data: remaining {len(texts)} valid samples")
    
    # 打印数据统计信息
    log_data_stats(texts, label_ids, label_names)
    
    return texts, label_ids, label_names

def log_data_stats(texts: List[str], label_ids: List[int], label_names: List[str]) -> None:
    """
    打印数据集统计信息
    """
    if not texts:
        logger.info("No valid data to show stats")
        return
    
    # 基本统计
    logger.info(f"Total samples: {len(texts)}")
    logger.info(f"Average text length: {sum(len(t) for t in texts) / len(texts):.1f} chars")
    
    # 标签分布
    if label_ids and any(lid is not None for lid in label_ids):
        label_counter = Counter(label_names)
        logger.info("Label distribution:")
        for label_name, count in sorted(label_counter.items(), key=lambda x: x[1], reverse=True):
            label_id = NAME_TO_ID.get(label_name, -1)
            percentage = (count / len(label_names)) * 100
            logger.info(f"  {label_name} (ID: {label_id}): {count} samples ({percentage:.1f}%)")

            
def save_predictions(predictions: Union[List[int], List[str]], output_path: str, text_col: Optional[List[str]] = None, prob_cols: Optional[List[List[float]]] = None) -> None:
    """
    保存预测结果到CSV文件。
    """
    # 构建结果DataFrame
    df_data = {}
    
    # 添加文本列（如果提供）
    if text_col is not None:
        df_data["text"] = text_col
    
    # 添加预测标签列
    if all(isinstance(pred, int) for pred in predictions):
        df_data["label_id"] = predictions
        df_data["label_name"] = [ID_TO_NAME.get(p, "unknown") for p in predictions]
    else:
        df_data["label_name"] = predictions
        df_data["label_id"] = [NAME_TO_ID.get(p, -1) for p in predictions]
    
    # 添加类别概率列（如果提供）
    if prob_cols is not None:
        prob_df = pd.DataFrame(
            prob_cols,
            columns=[f"prob_{name}" for name in ALL_LABEL_NAMES]
        )
        df_data = {**df_data, **prob_df.to_dict(orient="list")}
    
    # 保存文件
    df_out = pd.DataFrame(df_data)
    try:
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Predictions saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")


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
