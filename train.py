"""
Model Eğitim Modülü (HuggingFace Trainer API)
"""
import os
import numpy as np
from transformers import (
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from config import (
    MODEL_DIR_STAGE1, MODEL_DIR_STAGE2, 
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, WEIGHT_DECAY, 
    WARMUP_RATIO, SEED, device
)

def compute_metrics(eval_pred):
    """
    Değerlendirme metrikleri: Accuracy, Macro-F1, Weighted-F1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall
    }

def train_model(tokenized_datasets, model, tokenizer, output_dir, stage_name="Stage2"):
    """
    HuggingFace Trainer ile model eğitimi.
    """
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        report_to="none",  # WandB vb. devre dışı
        fp16=(device == "cuda"),  # GPU varsa mixed precision
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print(f"\n🚀 {stage_name} eğitimi başlıyor...")
    trainer.train()

    # En iyi modeli kaydet
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ {stage_name} tamamlandı. Model: {output_dir}")

    # Test seti değerlendirmesi
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"📊 Test Sonuçları ({stage_name}): {test_results}")

    return trainer, test_results