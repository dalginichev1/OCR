import re
import json
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (BertForSequenceClassification,
                          BertTokenizer,
                          Trainer,
                          TrainingArguments)


def clean_text(text):
    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def load_dataset(file_path, task=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    labels = []
    for item in data:
        if task is None or item['task'] == task:
            texts.append(clean_text(item['text']))
            labels.append(item['label'])
    return texts, labels


class EgeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }


def train_ege_evaluator(dataset_path, model_save_path, task=None):
    texts, labels = load_dataset(dataset_path, task)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'DeepPavlov/rubert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(set(labels)),
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    )

    def compute_loss(model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.1,
        save_steps=500,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=EgeDataset(train_texts, train_labels, tokenizer),
        eval_dataset=EgeDataset(val_texts, val_labels, tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"Результаты валидации для {task if task else 'всех заданий'}:", results)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Модель сохранена в {model_save_path}")


if __name__ == "__main__":
    DATASET_PATH = "bert_dataset.json"
    MODEL_SAVE_PATH = "ege_bert_model"
    train_ege_evaluator(DATASET_PATH, MODEL_SAVE_PATH)
