from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# === Konfigurasi ===
MODEL_PATH = "best_model.bin"
PRETRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512  # Sesuaikan dengan yang kamu gunakan saat training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenizer ===
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# === Load Model ===
try:
    # Muat arsitektur dasar BERT untuk klasifikasi
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=2  # phishing (1) / aman (0)
    )
    # Muat bobot fine-tuned
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model BERT berhasil dimuat!")
except Exception as e:
    print("❌ Gagal memuat model:", e)
    model = None

def predict_phishing(text):
    if model is None:
        return None, 0.0

    # Tokenisasi
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        confidence = float(probabilities[predicted_class])

    return predicted_class, confidence

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email tidak boleh kosong'}), 400

    try:
        pred_class, conf = predict_phishing(email)
        if pred_class is None:
            return jsonify({'error': 'Model tidak tersedia'}), 500

        result = "Phishing" if pred_class == 1 else "Aman"
        confidence = round(conf * 100, 2)

        return jsonify({
            'result': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)