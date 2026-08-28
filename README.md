# SearchMatch: E-Commerce Query-Product Relevance Engine

> End-to-end, açıklanabilir ve yüksek performanslı bir e-ticaret arama alaka düzeyi motoru.  
> DistilBERT tabanlı 2 aşamalı derin öğrenme mimarisi ile geliştirilmiştir.

---

## 🎯 Proje Motivasyonu

Bu proje, **CV ve portfolyo** için geliştirilmiş bağımsız bir çalışmadır. E-ticaret platformlarındaki kullanıcı arama niyetleri ile ürünler arasındaki anlamsal bağın çözümlenmesi problemine; 2 aşamalı derin öğrenme mimarisi, açıklanabilir AI (XAI) ve endüstriyel düzeyde inference optimizasyonu ile yaklaşılmıştır.

**Kullanılan Veriseti:** [WANDS (Wayfair Annotation DataSet)](https://github.com/wayfair/WANDS) — 42,994 ürün, 480 sorgu, 233,448 insan tarafından etiketlenmiş query-product eşleşmesi.

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────┐
│  Kullanıcı Sorgusu + Ürün Bilgileri (Ad, Kategori, Özellik) │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Aşama 1: İkili         │
              │  İlişkili / İlişkili Değil│
              │  (DistilBERT + MLP)       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Aşama 2: Çok Sınıflı   │
              │  Exact / Partial /      │
              │  Irrelevant             │
              │  (DistilBERT + MLP)       │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼─────┐
   │  Tahmin   │   │  LIME       │   │ Attention │
   │  + Skor   │   │  Açıklaması │   │  Grafiği  │
   └───────────┘   └─────────────┘   └───────────┘
```

**Neden 2 Aşamalı?**
- Aşama 1, hızlı bir ön filtreleme yaparak alakasız ürünleri eler.
- Aşama 2, kalan ürünler arasında ince ayrım (Exact vs Partial) yapar.
- Bu yapı, hem doğruluğu artırır hem de servis maliyetini düşürür.

---

## 📈 Performans Sonuçları

**Ortam:** NVIDIA GPU (CUDA), 8GB VRAM, 32GB RAM

| Metrik | Stage 1 (Binary) | Stage 2 (Multi-class) |
|---|---|---|
| **Macro-F1** | **0.9445** | **0.8980** |
| **Accuracy** | — | **0.9244** |
| **Precision (macro)** | — | **0.9046** |
| **Recall (macro)** | — | **0.8918** |

### Sınıf Bazlı Stage 2 Sonuçları

| Sınıf | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Irrelevant** | 0.9296 | 0.9089 | **0.9192** | 9,181 |
| **Partial** | 0.9347 | 0.9496 | **0.9421** | 21,995 |
| **Exact** | 0.8495 | 0.8168 | **0.8328** | 3,842 |

### Inference Hızı

| Mod | Ortalama Süre | Standart Sapma |
|---|---|---|
| **Tekli istek** | **6.77 ms** | ± 1.19 ms |
| **Batch (32)** | **1.70 ms/sorgu** | ± 0.06 ms |

> Endüstriyel servislerde kabul edilebilir eşik genellikle <100ms\'dir. Bu model, tekli istekte 14x daha hızlıdır.

---

## 🔍 Açıklanabilirlik (XAI)

Model kararlarının nedenini anlamak için iki yöntem entegre edilmiştir:

### 1. LIME (Local Interpretable Model-agnostic Explanations)
- Modelin tahminini hangi kelimelerin destekleyip/zayıflattığını metin olarak açıklar.
- Örnek: *"wood kelimesi Exact kararını +0.32 desteklerken, blue kelimesi -0.18 zayıflatmıştır."*

### 2. Attention Görselleştirme
- `[CLS]` token\'ının diğer tokenlere olan attention ağırlıkları çubuk grafik olarak çizilir.
- Modelin dikkatini gerçek içerik kelimelerine (`wood`, `coffee`, `table`) verdiği gözlemlenmiştir.

**Önemli Not:** İlk versiyonda `[QUERY]` ve `[PRODUCT]` gibi özel tokenler kullanıldığında model dikkatini anlamsız ayraç karakterlerine veriyordu. Bu problem, modelin doğal dilinde olan `[SEP]` (separator) tokeniyle çözülmüştür.

---

## 🛠️ Teknolojiler

| Katman | Teknoloji |
|---|---|
| **Model** | `distilbert-base-uncased` (66M parametre) |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Eğitim** | HuggingFace Trainer API (Early Stopping, Mixed Precision) |
| **XAI** | LIME, Attention Weights |
| **Demo** | Gradio |
| **Metrik** | Macro-F1 |

---

## 🚀 Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonla
git clone <repo-url>
cd searchmatch

# 2. WANDS verisetini indir (sadece CSV dosyaları data/ altına)
#    https://github.com/wayfair/WANDS/tree/main/dataset
#    → product.csv, query.csv, label.csv → data/

# 3. Sanal ortam (önerilir)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate  # Windows

# 4. Bağımlılıkları yükle
pip install -r requirements.txt

# 5. Eğitimi başlat (~20-30 dk, GPU ile)
python main.py

# 6. Demo arayüzünü çalıştır
python app.py
# → http://localhost:7860
```

---

## 📂 Proje Yapısı

```
searchmatch/
├── data/                      # WANDS CSV dosyaları (product, query, label)
├── outputs/
│   ├── stage1_binary/         # Aşama 1 model ağırlıkları
│   ├── stage2_multiclass/     # Aşama 2 model ağırlıkları
│   └── results/               # Eğitim logları, confusion matrix, summary
├── config.py                  # Hyperparametreler, label mapping, cihaz ayarları
├── dataset.py                 # Veri yükleme, [SEP] formatlı text birleştirme
├── model.py                   # DistilBERT + Dropout + 2-layer MLP
├── train.py                   # HuggingFace Trainer, compute_metrics
├── evaluate.py                # Macro-F1, inference hızı ölçümü
├── explain.py                 # LIME + Attention görselleştirme
├── app.py                     # Gradio demo (tahmin + XAI)
├── main.py                    # End-to-end eğitim pipeline
├── requirements.txt           # Python bağımlılıkları
└── README.md                  # Bu dosya
```

---

## 🧪 Problem Tanımı ve Değerlendirme Kriterleri

Bu proje, e-ticaret arama alaka düzeyi değerlendirmesinde yaygın olarak kullanılan standartlara uygundur:

| Kriter | Uygulama |
|---|---|
| **2 Aşamalı Sınıflandırma** | ✅ Aşama 1 (Binary) → Aşama 2 (Multi-class) |
| **Macro-F1 Değerlendirme** | ✅ Sklearn `f1_score(average="macro")` |
| **Model Hızı** | ✅ `time.perf_counter()` ile tekli/batch ölçümü |
| **Açıklanabilirlik Arayüzü** | ✅ LIME + Attention + Gradio |
| **Teknik Rapor** | ✅ README + Kod dokümantasyonu |

---

## 📝 Notlar

- **Dil:** Model ve veriseti İngilizcedir. DistilBERT tokenizer Türkçe karakterleri (`ç, ğ, ı, ö, ş, ü`) işleyebilir ancak anlamsal bağlamı İngilizce pre-training üzerinden öğrenmiştir.
- **Transfer:** Türkçe e-ticaret verisi ile fine-tune edilerek Türkçe\'ye adapte edilebilir.
- **Veriseti Atıfı:** WANDS verisetini kullanırken lütfen orijinal makaleyi atfedin:

```bibtex
@InProceedings{wands,
  title = {WANDS: Dataset for Product Search Relevance Assessment},
  author = {Chen, Yan and Liu, Shujian and Liu, Zheng and Sun, Weiyi and Baltrunas, Linas and Schroeder, Benjamin},
  booktitle = {Proceedings of the 44th European Conference on Information Retrieval},
  year = {2022},
  numpages = {12}
}
```
