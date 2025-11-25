"""
Fine-tune a pretrained BERT (HuggingFace) on the numeric-token CSV dataset.

This module provides a Trainer-based flow that:
- reads `data/train_set.csv` (tab-separated: label\ttext)
- extracts top-K numeric tokens and adds them to the tokenizer
- tokenizes with `bert-base-chinese` tokenizer and resizes embeddings
- trains with HF Trainer (supports multi-GPU + fp16 automatically)

Usage: use `run_finetune.py` to configure and run training.
"""
from pathlib import Path
import logging
import argparse
from collections import Counter
from typing import List, Tuple, Optional

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from inspect import signature
from .base import BaseModel
from huggingface_hub import snapshot_download
import os
from pathlib import Path
from safetensors import safe_open as _safe_open  # optional, used to validate safetensors files
import math
import os

# Avoid tokenizer parallelism causing deadlocks in some environments
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from data_utils import name_to_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read_csv(path: str, nrows: int = None) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        header = f.readline()
        for i, line in enumerate(f):
            if nrows and i >= nrows:
                break
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) != 2:
                continue
            label, text = parts
            texts.append(text.strip())
            labels.append(int(label))
    return texts, labels


def get_top_k_tokens(texts: List[str], top_k: int) -> List[str]:
    cnt = Counter()
    for t in texts:
        cnt.update(t.split())
    most = [tok for tok, _ in cnt.most_common(top_k)]
    return most


class NumericDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_datasets(
    tokenizer: BertTokenizer,
    texts: List[str],
    labels: List[int],
    max_length: int = 512,
    stride: Optional[int] = None,
):
    # If stride is provided and >0, produce sliding-window chunks for long sequences.
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = [] if hasattr(tokenizer, 'model_input_names') and 'token_type_ids' in tokenizer.model_input_names else None
    all_labels = []

    for text, lab in zip(texts, labels):
        if stride and stride > 0:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_overflowing_tokens=True,
                stride=stride,
                padding='max_length',
                return_attention_mask=True,
            )
            # when returning overflowing tokens, values are lists of lists
            input_ids_list = enc.get('input_ids')
            attn_list = enc.get('attention_mask')
            ttype_list = enc.get('token_type_ids') if 'token_type_ids' in enc else None
            for i in range(len(input_ids_list)):
                all_input_ids.append(input_ids_list[i])
                all_attention_mask.append(attn_list[i])
                if all_token_type_ids is not None and ttype_list is not None:
                    all_token_type_ids.append(ttype_list[i])
                all_labels.append(lab)
        else:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_attention_mask=True,
            )
            all_input_ids.append(enc['input_ids'])
            all_attention_mask.append(enc['attention_mask'])
            if all_token_type_ids is not None:
                all_token_type_ids.append(enc.get('token_type_ids'))
            all_labels.append(lab)

    encodings = {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
    }
    if all_token_type_ids is not None:
        encodings['token_type_ids'] = all_token_type_ids

    return NumericDataset(encodings, all_labels)


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = (preds == labels).astype(np.float32).mean().item()
    return {"accuracy": acc}


def build_training_args(ta_kwargs: dict):
    """Build TrainingArguments from ta_kwargs with compatibility fallbacks.

    - Filters unsupported kwargs based on TrainingArguments.__init__ signature.
    - If ValueError about mismatched eval/save strategies occurs, disables load_best_model_at_end.
    - If TypeError about unexpected keywords occurs, removes common newer keys and retries.
    """
    try:
        sig = signature(TrainingArguments.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
        filtered = {k: v for k, v in ta_kwargs.items() if k in valid_keys}
    except Exception:
        filtered = ta_kwargs.copy()

    def _try_build(kwargs):
        try:
            return TrainingArguments(**kwargs)
        except ValueError as e:
            msg = str(e)
            if 'requires the save and eval strategy to match' in msg or 'requires the save and eval' in msg:
                if 'load_best_model_at_end' in kwargs and kwargs.get('load_best_model_at_end'):
                    new_kwargs = kwargs.copy()
                    new_kwargs['load_best_model_at_end'] = False
                    return TrainingArguments(**new_kwargs)
            raise
        except TypeError:
            # Unexpected keyword(s); remove common newer keys and retry
            for k in ('evaluation_strategy', 'save_strategy', 'load_best_model_at_end', 'metric_for_best_model'):
                if k in kwargs:
                    kwargs.pop(k, None)
            return TrainingArguments(**kwargs)

    return _try_build(filtered)

class BertHFClassifier(BaseModel):
    """Adapter class wrapping HF Trainer to match BaseModel interface."""

    def __init__(self):
        # placeholders
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.num_labels = None
        self.output_dir = None

    def fit(self, X: List[str], y: List[int], **kwargs) -> None:
        """Fit using HF Trainer. Accepts kwargs: output_dir, pretrained, top_k_tokens, epochs, batch_size, max_length, fp16, nrows, learning_rate"""
        output_dir = kwargs.get('output_dir', kwargs.get('model_out', 'models/bert_finetuned'))
        pretrained = kwargs.get('pretrained', 'bert-base-chinese')
        top_k_tokens = kwargs.get('top_k_tokens', 6000)
        per_device_batch_size = kwargs.get('batch_size', kwargs.get('batch_size', 8))
        num_train_epochs = kwargs.get('epochs', 3)
        max_length = kwargs.get('max_length', 512)
        fp16 = kwargs.get('fp16', True)
        stride = kwargs.get('stride', None)
        dataloader_num_workers = kwargs.get('dataloader_num_workers', 0)

        # save for prediction time
        self.max_length = max_length
        self.stride = stride

        self.output_dir = output_dir

        texts = X
        labels = y
        # Normalize labels to contiguous integer ids required by HF Trainer / torch tensors
        processed_labels = []
        orig_to_id = {}
        id_to_orig = {}
        next_idx = 0
        for lab in labels:
            key = str(lab)
            if key in orig_to_id:
                lid = orig_to_id[key]
            else:
                lid = next_idx
                orig_to_id[key] = lid
                id_to_orig[lid] = key
                next_idx += 1
            processed_labels.append(lid)

        labels = processed_labels
        self.num_labels = next_idx
        # Save mapping to decode predictions back to original label strings
        self.id_to_label = id_to_orig
        self.label_to_id = orig_to_id
        logger.info(f"Label mapping (original -> id): {self.label_to_id}")

        # tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(ensure_pretrained_local(pretrained))
        added_tokens = get_top_k_tokens(texts, top_k_tokens)
        if added_tokens:
            self.tokenizer.add_tokens(added_tokens)

        self.model = BertForSequenceClassification.from_pretrained(ensure_pretrained_local(pretrained), num_labels=self.num_labels)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # split
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

        train_dataset = prepare_datasets(self.tokenizer, X_train, y_train, max_length=max_length, stride=stride)
        val_dataset = prepare_datasets(self.tokenizer, X_val, y_val, max_length=max_length, stride=stride)

        ta_kwargs = dict(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            dataloader_num_workers=dataloader_num_workers,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            fp16=fp16,
            logging_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )
        try:
            sig = signature(TrainingArguments.__init__)
            valid_keys = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
            filtered = {k: v for k, v in ta_kwargs.items() if k in valid_keys}
        except Exception:
            filtered = ta_kwargs.copy()

        training_args = build_training_args(ta_kwargs)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )

        self.trainer.train()

        # save
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, X: List[str]) -> List[int]:
        if self.trainer is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        device = getattr(self.trainer.args, 'device', None)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        preds = []
        for text in X:
            if getattr(self, 'stride', None):
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=getattr(self, 'max_length', 512),
                    return_overflowing_tokens=True,
                    stride=getattr(self, 'stride'),
                    padding='max_length',
                    return_attention_mask=True,
                )
                input_ids_list = enc.get('input_ids')
                attn_list = enc.get('attention_mask')
                ttype_list = enc.get('token_type_ids') if 'token_type_ids' in enc else None

                input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
                attention_mask = torch.tensor(attn_list, dtype=torch.long).to(device)
                kwargs_model = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if ttype_list is not None:
                    kwargs_model['token_type_ids'] = torch.tensor(ttype_list, dtype=torch.long).to(device)

                with torch.no_grad():
                    outputs = self.model(**kwargs_model)
                    logits = outputs.logits
                    avg_logits = logits.mean(dim=0)
                    pred = int(torch.argmax(avg_logits).cpu().numpy())
                preds.append(pred)
            else:
                enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=getattr(self, 'max_length', 512), return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = self.model(**enc)
                    logits = outputs.logits
                    pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                preds.append(pred)

        # If we have an id->original label mapping, decode to original labels (strings)
        if getattr(self, 'id_to_label', None):
            return [self.id_to_label.get(p, p) for p in preds]
        return preds

    def predict_proba(self, X: List[str]):
        if self.trainer is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        device = getattr(self.trainer.args, 'device', None)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        all_probs = []
        for text in X:
            if getattr(self, 'stride', None):
                enc = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=getattr(self, 'max_length', 512),
                    return_overflowing_tokens=True,
                    stride=getattr(self, 'stride'),
                    padding='max_length',
                    return_attention_mask=True,
                )
                input_ids_list = enc.get('input_ids')
                attn_list = enc.get('attention_mask')
                ttype_list = enc.get('token_type_ids') if 'token_type_ids' in enc else None

                input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
                attention_mask = torch.tensor(attn_list, dtype=torch.long).to(device)
                kwargs_model = {'input_ids': input_ids, 'attention_mask': attention_mask}
                if ttype_list is not None:
                    kwargs_model['token_type_ids'] = torch.tensor(ttype_list, dtype=torch.long).to(device)

                with torch.no_grad():
                    outputs = self.model(**kwargs_model)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    avg_probs = probs.mean(dim=0).cpu().numpy().tolist()
                all_probs.append(avg_probs)
            else:
                enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=getattr(self, 'max_length', 512), return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    outputs = self.model(**enc)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0].tolist()
                all_probs.append(probs)

        return all_probs

    def save(self, path: str) -> None:
        # Save HF model and tokenizer
        if self.trainer is not None:
            self.trainer.save_model(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> 'BertHFClassifier':
        inst = cls()
        # if the provided path is not a local folder, ensure it's downloaded locally
        local_path = ensure_pretrained_local(path)
        inst.tokenizer = BertTokenizer.from_pretrained(local_path)
        # load model
        inst.model = BertForSequenceClassification.from_pretrained(local_path)
        inst.trainer = None
        return inst


def ensure_pretrained_local(pretrained: str, cache_root: str = 'models/pretrained') -> str:
    """
    Ensure that `pretrained` refers to a local directory containing a HF model.
    If `pretrained` is already a local path with model files, return it unchanged.
    Otherwise, download the model snapshot from the Hub into `cache_root/<repo_id>` and return that path.
    """
    # If it already looks like a local directory, and contains expected files, return it
    p = Path(pretrained)
    expected_files = ['pytorch_model.bin', 'tf_model.h5', 'flax_model.msgpack', 'config.json']
    if p.is_dir():
        # If directory contains at least one expected weight file, return it
        for ef in expected_files:
            f = p / ef
            if f.exists():
                # If it's a safetensors file, try to open to validate header
                if ef.endswith('.safetensors') or str(f).endswith('.safetensors'):
                    try:
                        with _safe_open(str(f), framework="pt") as sf:
                            # access keys to ensure file is readable
                            _ = sf.keys()
                    except Exception as e:
                        raise RuntimeError(f"safetensors file appears corrupted: {f}. Error: {e}")
                return str(p)
        # If dir exists but no expected files, still return it (allow custom layouts)
        return str(p)

    # Not a local dir; attempt to download from HF Hub into cache_root
    repo_id = pretrained
    # sanitize repo_id into a directory name
    safe_name = repo_id.replace('/', '__')
    target_dir = Path(cache_root) / safe_name
    if target_dir.exists():
        # if target already has model files, validate and return it
        for ef in expected_files:
            f = target_dir / ef
            if f.exists():
                if ef.endswith('.safetensors') or str(f).endswith('.safetensors'):
                    try:
                        with _safe_open(str(f), framework="pt") as sf:
                            _ = sf.keys()
                    except Exception as e:
                        raise RuntimeError(f"Existing safetensors file appears corrupted: {f}. Error: {e}")
                return str(target_dir)

    # create parent
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # snapshot_download supports local_dir in newer versions; use it to download into our folder
        snapshot_download(repo_id=repo_id, local_dir=str(target_dir), local_dir_use_symlinks=False)
        return str(target_dir)
    except Exception:
        # fallback: let transformers/hf handle remote identifier (will raise later)
        return pretrained
