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
from .base import BaseModel

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


def train(
    data_path: str,
    output_dir: str,
    pretrained: str = 'bert-base-chinese',
    top_k_tokens: int = 6000,
    nrows: int = None,
    per_device_batch_size: int = 8,
    num_train_epochs: int = 3,
    max_length: int = 512,
    fp16: bool = True,
):
    texts, labels = read_csv(data_path, nrows=nrows)
    num_labels = len(set(labels))
    logger.info(f"Loaded {len(texts)} samples, {num_labels} labels")

    # choose top-K tokens from dataset and add to tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    added_tokens = get_top_k_tokens(texts, top_k_tokens)
    logger.info(f"Adding {len(added_tokens)} tokens to tokenizer (top {top_k_tokens})")
    tokenizer.add_tokens(added_tokens)

    model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=num_labels)
    model.resize_token_embeddings(len(tokenizer))

    # split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    train_dataset = prepare_datasets(tokenizer, X_train, y_train, max_length=max_length)
    val_dataset = prepare_datasets(tokenizer, X_val, y_val, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training completed. Model saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/train_set.csv')
    parser.add_argument('--output_dir', default='models/bert_finetuned')
    parser.add_argument('--top_k_tokens', type=int, default=6000)
    parser.add_argument('--nrows', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        top_k_tokens=args.top_k_tokens,
        nrows=args.nrows,
        per_device_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        fp16=args.fp16,
    )


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

        # save for prediction time
        self.max_length = max_length
        self.stride = stride

        self.output_dir = output_dir

        texts = X
        labels = y
        self.num_labels = len(set(labels))

        # tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        added_tokens = get_top_k_tokens(texts, top_k_tokens)
        if added_tokens:
            self.tokenizer.add_tokens(added_tokens)

        self.model = BertForSequenceClassification.from_pretrained(pretrained, num_labels=self.num_labels)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # split
        X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)

        train_dataset = prepare_datasets(self.tokenizer, X_train, y_train, max_length=max_length, stride=stride)
        val_dataset = prepare_datasets(self.tokenizer, X_val, y_val, max_length=max_length, stride=stride)

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy='epoch',
            save_strategy='epoch',
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
        inst.tokenizer = BertTokenizer.from_pretrained(path)
        # load model
        inst.model = BertForSequenceClassification.from_pretrained(path)
        inst.trainer = None
        return inst
