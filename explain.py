"""
Açıklanabilirlik (XAI) Modülü
LIME + Attention Görselleştirme
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from config import device, INV_LABEL_MAP

class ModelWrapper:
    """
    LIME için model wrapper. HuggingFace modelini LIME'in anlayacağı formata çevirir.
    """
    def __init__(self, model, tokenizer, num_labels=3):
        self.model = model
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.model.eval()
        self.model.to(device)

    def predict_proba(self, texts):
        """
        LIME, bu fonksiyonu çağırır. Her text için olasılık dağılımı döndürür.
        """
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy()

def explain_with_lime(model, tokenizer, text, num_features=10, num_labels=3):
    """
    LIME ile metin tabanlı açıklama üretir.
    Hangi kelimelerin modelin kararını etkilediğini gösterir.
    """
    class_names = [INV_LABEL_MAP[i] for i in range(num_labels)] if num_labels == 3 else ["İlişkili Değil", "İlişkili"]

    explainer = LimeTextExplainer(class_names=class_names)
    wrapper = ModelWrapper(model, tokenizer, num_labels)

    explanation = explainer.explain_instance(
        text, 
        wrapper.predict_proba, 
        num_features=num_features,
        top_labels=1
    )

    return explanation

def visualize_attention(model, tokenizer, text, save_path="outputs/attention_viz.png"):
    """
    Modelin attention ağırlıklarını görselleştirir.
    Hangi tokenlere daha çok odaklandığını gösterir.
    """
    model.eval()
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.distilbert(**inputs, output_attentions=True)
        attentions = outputs.attentions  # Tuple of (layers, batch, heads, seq_len, seq_len)

    # Son katman, tüm head'lerin ortalaması
    last_layer_attn = attentions[-1]  # (batch, heads, seq_len, seq_len)
    avg_attn = last_layer_attn.mean(dim=1)[0]  # (seq_len, seq_len)

    # [CLS] token'ından diğer tokenlere olan attention ( ilk satır )
    cls_attn = avg_attn[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Görselleştir
    plt.figure(figsize=(14, 4))
    x_pos = np.arange(len(tokens))
    plt.bar(x_pos, cls_attn, color="steelblue")
    plt.xticks(x_pos, tokens, rotation=90, fontsize=8)
    plt.title("Attention Ağırlıkları (CLS Token)")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return tokens, cls_attn

def generate_explanation_text(pred_label, explanation, num_features=5):
    """
    LIME çıktısından insan tarafından okunabilir açıklama metni üretir.
    Sistem token'larını (SEP, CLS vb.) gizler.
    """
    # Tüm özellikleri alıyoruz (sınırı filtrelemeden sonra uygulayacağız)
    top_features = explanation.as_list(label=explanation.available_labels()[0])

    # Görmezden gelinecek sistem kelimeleri
    ignore_words = ["SEP", "CLS", "QUERY", "PRODUCT"]

    # Hem skora göre ayırıyoruz hem de ignore_words içindekileri çöpe atıyoruz
    positive_words = [word for word, score in top_features if score > 0 and word.upper() not in ignore_words][:num_features]
    negative_words = [word for word, score in top_features if score < 0 and word.upper() not in ignore_words][:num_features]

    text = f"Model, bu ürünü **{INV_LABEL_MAP.get(pred_label, pred_label)}** olarak sınıflandırdı.\n\n"

    if positive_words:
        text += f"✅ **Eşleşmeyi destekleyen kelimeler:** {', '.join(positive_words)}\n"
    if negative_words:
        text += f"❌ **Eşleşmeyi zayıflatan kelimeler:** {', '.join(negative_words)}\n"

    return text