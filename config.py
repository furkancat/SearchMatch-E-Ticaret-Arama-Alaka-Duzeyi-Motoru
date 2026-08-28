"""
SearchMatch: Proje Konfigürasyonu
"""
import torch

# Veri yolları
DATA_DIR = "data"
PRODUCT_CSV = f"{DATA_DIR}/product.csv"
QUERY_CSV = f"{DATA_DIR}/query.csv"
LABEL_CSV = f"{DATA_DIR}/label.csv"

# Çıktı yolları
OUTPUT_DIR = "outputs"
MODEL_DIR_STAGE1 = f"{OUTPUT_DIR}/stage1_binary"
MODEL_DIR_STAGE2 = f"{OUTPUT_DIR}/stage2_multiclass"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# Model konfigürasyonu
MODEL_NAME = "distilbert-base-uncased"  # 66M parametre, 8GB VRAM için ideal
MAX_LENGTH = 256
BATCH_SIZE = 32  # 8GB VRAM'de DistilBERT için güvenli
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# Cihaz
device = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping (WANDS → Proje)
# WANDS: Exact, Partial, Irrelevant
# Stage 1 (Binary): Related (Exact+Partial) vs Irrelevant
# Stage 2 (Multi): Exact vs Partial vs Irrelevant
LABEL_MAP = {
    "Irrelevant": 0,
    "Partial": 1,
    "Exact": 2
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Stage 1 için binary map
BINARY_LABEL_MAP = {
    0: 0,  # Irrelevant -> 0 (İlişkili Değil)
    1: 1,  # Partial -> 1 (İlişkili)
    2: 1   # Exact -> 1 (İlişkili)
}