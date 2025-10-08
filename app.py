from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# === Konfigurasi ===
MODEL_REPO = "Rivaldi3001/bert-phishing-detector"  # Ganti dengan repo model kamu di Hugging Face
MAX_LENGTH = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenizer & Model dari Hugging Face ===
try:
    print("🔄 Memuat tokenizer dan model dari Hugging Face...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO)
    model.to(device)
    model.eval()
    print("✅ Model berhasil dimuat dari Hugging Face!")
except Exception as e:
    print("❌ Gagal memuat model dari Hugging Face:", e)
    model = None


# === Fungsi Prediksi ===
def predict_phishing(text):
    if model is None:
        return None, 0.0

    # Tokenisasi input
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    # Prediksi
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        confidence = float(probs[pred_class])

    return pred_class, confidence


# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Teks tidak boleh kosong'}), 400

    try:
        pred_class, conf = predict_phishing(email)
        if pred_class is None:
            return jsonify({'error': 'Model belum siap'}), 500

        result = "Phishing" if pred_class == 1 else "Aman"
        confidence = round(conf * 100, 2)

        return jsonify({
            'result': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === Jalankan secara lokal ===
if __name__ == '__main__':
    app.run(debug=True)
