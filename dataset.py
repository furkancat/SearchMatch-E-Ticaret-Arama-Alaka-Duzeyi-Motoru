"""
WANDS Veri Yükleyici ve Ön İşleme Modülü
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from config import PRODUCT_CSV, QUERY_CSV, LABEL_CSV, LABEL_MAP, BINARY_LABEL_MAP, SEED

def load_wands_data():
    """
    WANDS CSV dosyalarını yükler ve birleştirir.
    Bozuk satırları on_bad_lines='skip' ile atlar.
    """
    # WANDS CSV'lerini oku (quoting=3 kaldırıldı)
    products = pd.read_csv(
        PRODUCT_CSV, 
        sep='\t',
        on_bad_lines='skip'
    )
    queries = pd.read_csv(
        QUERY_CSV,
        sep='\t',
        on_bad_lines='skip'
    )
    labels = pd.read_csv(
        LABEL_CSV,
        sep='\t',
        on_bad_lines='skip'
    )

    print(f"Yüklenen: {len(products)} ürün, {len(queries)} sorgu, {len(labels)} etiket")

    # Sütun isimlerinde kazara boşluk kalmışsa temizleyelim (Garanti olsun diye)
    labels.columns = labels.columns.str.strip()

    # Label'ları sayısala çevir
    labels["label_id"] = labels["label"].map(LABEL_MAP)

    # Birleştir
    merged = labels.merge(queries, on="query_id", how="left")
    merged = merged.merge(products, on="product_id", how="left")

    # Eksik değerleri temizle
    merged = merged.dropna(subset=["query", "product_name", "label_id"])
    merged["label_id"] = merged["label_id"].astype(int)

    print(f"Birleştirme sonrası: {len(merged)} geçerli örnek")

    return merged

def create_text_field(row):
    """
    Query ve ürün bilgilerini birleştirerek model girdisi oluşturur.
    Format: [QUERY] query_text [PRODUCT] name | category | features
    """
    query = str(row["query"]).strip()
    product_name = str(row["product_name"]).strip()
    category = str(row.get("category_hierarchy", "")).strip()
    features = str(row.get("product_features", "")).strip()

    product_text = f"{product_name}"
    if category and category != "nan":
        product_text += f" | Category: {category}"
    if features and features != "nan":
        # features: "attribute:value|attribute:value" formatında
        features_clean = features.replace("|", ", ")
        product_text += f" | Features: {features_clean}"

    return f"{query} [SEP] {product_text}"

def prepare_datasets(test_size=0.15, val_size=0.15):
    """
    WANDS verisini hazırlar ve train/val/test olarak bölür.
    Stage 1 (Binary) ve Stage 2 (Multi-class) için ayrı label'lar üretir.
    """
    df = load_wands_data()

    # Text birleştir
    df["text"] = df.apply(create_text_field, axis=1)

    # Stage 1: Binary labels (0: İlişkili Değil, 1: İlişkili)
    df["label_binary"] = df["label_id"].map(BINARY_LABEL_MAP)

    # Stage 2: Multi-class labels (0: Irrelevant, 1: Partial, 2: Exact)
    df["label_multi"] = df["label_id"]

    # Train/Val/Test split (stratify ederek dengeli bölme)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=SEED, stratify=df["label_multi"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(test_size / (val_size + test_size)), random_state=SEED, stratify=temp_df["label_multi"]
    )

    # HuggingFace Dataset formatına çevir
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label_binary", "label_multi"]].reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df[["text", "label_binary", "label_multi"]].reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df[["text", "label_binary", "label_multi"]].reset_index(drop=True))
    })

    print(f"\nBölünme: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    print(f"Binary dağılım: {train_df['label_binary'].value_counts().to_dict()}")
    print(f"Multi dağılım: {train_df['label_multi'].value_counts().to_dict()}")

    return dataset, df

if __name__ == "__main__":
    dataset, raw_df = prepare_datasets()
    print("\nÖrnek girdi:")
    print(raw_df.iloc[0]["text"][:300])