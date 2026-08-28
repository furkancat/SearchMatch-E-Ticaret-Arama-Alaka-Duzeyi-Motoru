"""
Model Değerlendirme ve Hız Testi Modülü
"""
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import device, INV_LABEL_MAP

def measure_inference_speed(model, tokenizer, texts, num_runs=100):
    """
    Modelin inference hızını ölçer (ms/sorgu).
    Tekli ve toplu (batch) istekler için ayrı ölçüm.
    """
    model.eval()
    model.to(device)

    # Tekli istek hızı
    single_times = []
    for _ in range(min(num_runs, len(texts))):
        text = texts[_ % len(texts)]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        single_times.append((time.perf_counter() - start) * 1000)

    # Batch istek hızı (batch_size=32)
    batch_times = []
    batch_size = 32
    for i in range(0, min(num_runs, len(texts)), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        batch_times.append(elapsed / len(batch_texts))

    results = {
        "single_mean_ms": np.mean(single_times),
        "single_std_ms": np.std(single_times),
        "batch_mean_ms": np.mean(batch_times),
        "batch_std_ms": np.std(batch_times),
    }

    print(f"⚡ Inference Hızı:")
    print(f"   Tekli istek: {results['single_mean_ms']:.2f} ± {results['single_std_ms']:.2f} ms")
    print(f"   Batch (32):  {results['batch_mean_ms']:.2f} ± {results['batch_std_ms']:.2f} ms/sorgu")

    return results

def evaluate_model(trainer, tokenized_test, stage_name="Stage2", num_labels=3):
    """
    Detaylı değerlendirme raporu üretir.
    """
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Macro-F1 (TEKNOFEST'in kullandığı metrik)
    macro_f1 = f1_score(labels, preds, average="macro")

    print(f"\n📊 {stage_name} Değerlendirme Raporu")
    print(f"   Macro-F1: {macro_f1:.4f}")
    print("\n" + classification_report(
        labels, preds, 
        target_names=[INV_LABEL_MAP[i] for i in range(num_labels)] if num_labels == 3 else ["İlişkili Değil", "İlişkili"],
        digits=4
    ))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{stage_name} - Confusion Matrix")
    plt.ylabel("Gerçek")
    plt.xlabel("Tahmin")
    plt.savefig(f"outputs/{stage_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"macro_f1": macro_f1, "predictions": preds, "labels": labels}