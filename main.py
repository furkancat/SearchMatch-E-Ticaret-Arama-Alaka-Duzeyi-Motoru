"""
SearchMatch: Ana Pipeline
-------------------------
1. Veri Yükleme ve Ön İşleme
2. Tokenizasyon
3. Stage 1 Eğitimi (Binary: İlişkili/Değil)
4. Stage 2 Eğitimi (Multi-class: Exact/Partial/Irrelevant)
5. Değerlendirme (Macro-F1 + Hız Testi)
6. Açıklanabilirlik Demo

Kullanım:
    python main.py

Gereksinimler:
    pip install -r requirements.txt
    # WANDS verisetini data/ klasörüne yerleştir:
    #   data/product.csv
    #   data/query.csv
    #   data/label.csv
"""
import os
import sys
import torch
from datasets import DatasetDict

from config import (
    MODEL_DIR_STAGE1, MODEL_DIR_STAGE2, RESULTS_DIR, 
    NUM_EPOCHS, BATCH_SIZE, device, SEED
)
from dataset import prepare_datasets
from model import SearchMatchModel, get_tokenizer, tokenize_function
from train import train_model
from evaluate import evaluate_model, measure_inference_speed

def set_seed():
    """Tekrarlanabilirlik için seed ayarı."""
    import random
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def main():
    print("=" * 60)
    print("🚀 SearchMatch: E-Commerce Query-Product Relevance Engine")
    print("=" * 60)
    print(f"Cihaz: {device.upper()}")
    print(f"Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR_STAGE1, exist_ok=True)
    os.makedirs(MODEL_DIR_STAGE2, exist_ok=True)

    # 1. Veri Hazırlama
    print("\n📦 Adım 1: Veri yükleniyor...")
    dataset, raw_df = prepare_datasets()

    # 2. Tokenizasyon
    print("\n🔤 Adım 2: Tokenizasyon...")
    tokenizer = get_tokenizer()

    # Stage 1 için binary label ile tokenize et
    tokenized_stage1 = dataset.map(
        lambda x: tokenize_function(x, tokenizer, label_col="label_binary"),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_stage1.set_format("torch")

    # Stage 2 için multi-class label ile tokenize et
    tokenized_stage2 = dataset.map(
        lambda x: tokenize_function(x, tokenizer, label_col="label_multi"),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    tokenized_stage2.set_format("torch")

    # 3. Stage 1: Binary Sınıflandırma
    print("\n" + "=" * 60)
    print("🔷 AŞAMA 1: İkili Sınıflandırma (İlişkili / İlişkili Değil)")
    print("=" * 60)

    model_stage1 = SearchMatchModel(num_labels=2)
    if device == "cuda":
        model_stage1 = model_stage1.to(device)

    trainer1, results1 = train_model(
        tokenized_stage1, model_stage1, tokenizer, 
        MODEL_DIR_STAGE1, stage_name="Stage1-Binary"
    )

    # Stage 1 Değerlendirme
    eval1 = evaluate_model(trainer1, tokenized_stage1["test"], stage_name="Stage 1", num_labels=2)

    # Hız testi
    test_texts = raw_df.iloc[:100]["text"].tolist()
    speed1 = measure_inference_speed(model_stage1, tokenizer, test_texts, num_runs=50)

    # 4. Stage 2: Multi-class Sınıflandırma
    print("\n" + "=" * 60)
    print("🔷 AŞAMA 2: Çok Sınıflı Sınıflandırma (Exact / Partial / Irrelevant)")
    print("=" * 60)

    model_stage2 = SearchMatchModel(num_labels=3)
    if device == "cuda":
        model_stage2 = model_stage2.to(device)

    trainer2, results2 = train_model(
        tokenized_stage2, model_stage2, tokenizer,
        MODEL_DIR_STAGE2, stage_name="Stage2-MultiClass"
    )

    # Stage 2 Değerlendirme
    eval2 = evaluate_model(trainer2, tokenized_stage2["test"], stage_name="Stage 2", num_labels=3)

    # Hız testi
    speed2 = measure_inference_speed(model_stage2, tokenizer, test_texts, num_runs=50)

    # 5. Sonuç Özeti
    print("\n" + "=" * 60)
    print("📊 NİHAİ SONUÇ ÖZETİ")
    print("=" * 60)
    print(f"Stage 1 (Binary) Macro-F1: {eval1['macro_f1']:.4f}")
    print(f"Stage 2 (Multi)  Macro-F1: {eval2['macro_f1']:.4f}")
    print(f"Stage 2 Tekli Inference:   {speed2['single_mean_ms']:.2f} ms")
    print(f"Stage 2 Batch Inference:   {speed2['batch_mean_ms']:.2f} ms/sorgu")
    print("=" * 60)

    # Sonuçları kaydet
    summary = f"""
SearchMatch - Eğitim Sonuçları
================================
Cihaz: {device}
Batch Size: {BATCH_SIZE}
Epochs: {NUM_EPOCHS}

Stage 1 (Binary):
  Macro-F1: {eval1['macro_f1']:.4f}
  Tekli Hız: {speed1['single_mean_ms']:.2f} ms
  Batch Hız: {speed1['batch_mean_ms']:.2f} ms

Stage 2 (Multi-class):
  Macro-F1: {eval2['macro_f1']:.4f}
  Tekli Hız: {speed2['single_mean_ms']:.2f} ms
  Batch Hız: {speed2['batch_mean_ms']:.2f} ms

Modeller kaydedildi:
  - {MODEL_DIR_STAGE1}
  - {MODEL_DIR_STAGE2}
    """

    with open(f"{RESULTS_DIR}/training_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n✅ Pipeline tamamlandı! Sonuçlar: {RESULTS_DIR}/")
    print("\n🎨 Demo çalıştırmak için:")
    print("   python app.py")

if __name__ == "__main__":
    main()