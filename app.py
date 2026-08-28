"""
Gradio Arayüzü: SearchMatch Demo Uygulaması
Model tahmini + Açıklanabilirlik (LIME + Attention)
"""
import os
import torch
import numpy as np
from safetensors.torch import load_file
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertConfig
from model import SearchMatchModel
from explain import explain_with_lime, visualize_attention, generate_explanation_text
from config import MODEL_DIR_STAGE1, MODEL_DIR_STAGE2, INV_LABEL_MAP, device

# Modelleri yükle (global)
print("Modeller yükleniyor...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR_STAGE2)

# Stage 2 Model (3 sınıf: Exact, Partial, Irrelevant)
config2 = DistilBertConfig.from_pretrained(MODEL_DIR_STAGE2, num_labels=3)
model_stage2 = SearchMatchModel(num_labels=3)
model_stage2.load_state_dict(load_file(f"{MODEL_DIR_STAGE2}/model.safetensors"))
model_stage2.to(device)
model_stage2.eval()

# Stage 1 Model (2 sınıf: İlişkili/Değil) - varsa yükle
if os.path.exists(f"{MODEL_DIR_STAGE1}/model.safetensors"):
    config1 = DistilBertConfig.from_pretrained(MODEL_DIR_STAGE1, num_labels=2)
    model_stage1 = SearchMatchModel(num_labels=2)
    model_stage1.load_state_dict(load_file(f"{MODEL_DIR_STAGE1}/model.safetensors"))
    model_stage1.to(device)
    model_stage1.eval()
else:
    model_stage1 = None
    print("Stage 1 modeli bulunamadı, sadece Stage 2 çalışacak.")

def predict_relevance(query, product_name, category="", features=""):
    """
    Kullanıcı girdisinden tahmin ve açıklama üretir.
    """
    # Text birleştir (dataset.py'deki ile aynı format)
    product_text = product_name
    if category:
        product_text += f" | Category: {category}"
    if features:
        product_text += f" | Features: {features}"

    text = f"{query} [SEP] {product_text}"

    # Stage 1: Binary tahmin (varsa)
    if model_stage1:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs1 = model_stage1(**inputs)
            probs1 = torch.softmax(outputs1["logits"], dim=-1)[0].cpu().numpy()
        stage1_pred = int(np.argmax(probs1))
        stage1_conf = float(probs1[stage1_pred])
        stage1_text = "İlişkili" if stage1_pred == 1 else "İlişkili Değil"
    else:
        stage1_text = "N/A"
        stage1_conf = 0.0

    # Stage 2: Multi-class tahmin
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs2 = model_stage2(**inputs)
        probs2 = torch.softmax(outputs2["logits"], dim=-1)[0].cpu().numpy()

    stage2_pred = int(np.argmax(probs2))
    stage2_conf = float(probs2[stage2_pred])
    stage2_label = INV_LABEL_MAP.get(stage2_pred, "Bilinmiyor")

    # Olasılıklar metni
    prob_text = "\n".join([f"- {INV_LABEL_MAP[i]}: {probs2[i]:.2%}" for i in range(3)])

    # LIME Açıklaması
    explanation = explain_with_lime(model_stage2, tokenizer, text, num_features=8, num_labels=3)
    lime_text = generate_explanation_text(stage2_pred, explanation, num_features=5)

    # Attention görselleştirmesi
    attn_path = "outputs/attn_temp.png"
    os.makedirs("outputs", exist_ok=True)
    visualize_attention(model_stage2, tokenizer, text, save_path=attn_path)

    result_md = f"""## 🔍 Tahmin Sonucu

| Aşama | Sonuç | Güven |
|-------|-------|-------|
| **Aşama 1 (İkili)** | {stage1_text} | {stage1_conf:.1%} |
| **Aşama 2 (Çok Sınıflı)** | **{stage2_label}** | {stage2_conf:.1%} |

### Sınıf Olasılıkları
{prob_text}

### 🧠 Model Açıklaması (LIME)
{lime_text}"""

    return result_md, attn_path

# Gradio Arayüzü
with gr.Blocks(title="SearchMatch Demo") as demo:
    gr.Markdown("# 🛒 SearchMatch: E-Ticaret Arama Alaka Düzeyi Motoru")
    gr.Markdown("Arama sorgusu ve ürün bilgilerini girin. Model, ikili ve çok sınıflı olarak alaka düzeyini tahmin edecek ve kararının nedenini açıklayacak.")

    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Arama Sorgusu (Query)", placeholder="örn: wood coffee table")
            product_input = gr.Textbox(label="Ürün Adı", placeholder="örn: Modern Coffee Table")
            category_input = gr.Textbox(label="Kategori (Opsiyonel)", placeholder="örn: Furniture / Living Room / Tables")
            features_input = gr.Textbox(label="Ürün Özellikleri (Opsiyonel)", placeholder="örn: Color:Brown, Material:Wood, Style:Modern")
            submit_btn = gr.Button("Tahmin Et", variant="primary")

        with gr.Column():
            result_output = gr.Markdown()
            attn_image = gr.Image(label="Attention Görselleştirmesi", type="filepath")

    submit_btn.click(
        fn=predict_relevance,
        inputs=[query_input, product_input, category_input, features_input],
        outputs=[result_output, attn_image]
    )

    # Örnekler
    gr.Examples(
        examples=[
            ["wood coffee table", "Modern Coffee Table", "Furniture / Living Room / Tables", "Color:Brown | Material:Wood"],
            ["red running shoes", "Blue Basketball Sneakers", "Shoes / Athletic / Basketball", "Color:Blue | Size:10"],
            ["porcelain dinner plate", "Book Lovers Pasta Bowl", "Tableware / Kitchen / Bowls", "Material:Ceramic | Style:Modern"],
        ],
        inputs=[query_input, product_input, category_input, features_input],
        label="Örnek Girdiler"
    )

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)