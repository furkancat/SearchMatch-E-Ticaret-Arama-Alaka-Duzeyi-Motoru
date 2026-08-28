"""
DistilBERT Tabanlı 2 Aşamalı Sınıflandırma Modeli
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertConfig
from config import MODEL_NAME, MAX_LENGTH, device

class SearchMatchModel(nn.Module):
    """
    DistilBERT + Dropout + Classification Head
    Stage 1: Binary (2 sınıf)
    Stage 2: Multi-class (3 sınıf)
    """
    def __init__(self, num_labels=3, dropout_rate=0.3):
        super(SearchMatchModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.config = self.distilbert.config
        hidden_size = self.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 2, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

def get_tokenizer():
    """DistilBERT tokenizer'ı döndürür."""
    return DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples, tokenizer, label_col="label_multi"):
    """
    HuggingFace datasets için tokenizasyon fonksiyonu.
    """
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = examples[label_col]
    return tokenized