# from flask import Flask, render_template, request, jsonify
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# import os

# app = Flask(__name__)

# # === Konfigurasi ===
# MODEL_REPO = "Rivaldi3001/bert-phishing-detector"  # Ganti dengan repo model kamu di Hugging Face
# MAX_LENGTH = 512
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === Load Tokenizer & Model dari Hugging Face ===
# try:
#     print("🔄 Memuat tokenizer dan model dari Hugging Face...")
#     tokenizer = BertTokenizer.from_pretrained(MODEL_REPO)
#     model = BertForSequenceClassification.from_pretrained(MODEL_REPO)
#     model.to(device)
#     model.eval()
#     print("✅ Model berhasil dimuat dari Hugging Face!")
# except Exception as e:
#     print("❌ Gagal memuat model dari Hugging Face:", e)
#     model = None


# # === Fungsi Prediksi ===
# def predict_phishing(text):
#     if model is None:
#         return None, 0.0

#     # Tokenisasi input
#     inputs = tokenizer(
#         text,
#         truncation=True,
#         padding=True,
#         max_length=MAX_LENGTH,
#         return_tensors="pt"
#     ).to(device)

#     # Prediksi
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         pred_class = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
#         confidence = float(probs[pred_class])

#     return pred_class, confidence


# # === Routes ===
# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     email = request.form.get('email', '').strip()
#     if not email:
#         return jsonify({'error': 'Teks tidak boleh kosong'}), 400

#     try:
#         pred_class, conf = predict_phishing(email)
#         if pred_class is None:
#             return jsonify({'error': 'Model belum siap'}), 500

#         result = "Phishing" if pred_class == 1 else "Aman"
#         confidence = round(conf * 100, 2)

#         return jsonify({
#             'result': result,
#             'confidence': confidence
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# # === Jalankan secara lokal ===
# if __name__ == '__main__':
#     app.run(debug=True)
# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import os

# Download NLTK resources (pastikan sudah diunduh sekali)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# === Path file ===
MODEL_PATH = "model.pkl"
TFIDF_PATH = "tfidf.pkl"
SLANG_DICT_PATH = "slang.txt"

# === Load slang dictionary ===
slang_dict = {}
if os.path.exists(SLANG_DICT_PATH):
    with open(SLANG_DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            slang_dict[key.strip().lower()] = value.strip().lower()
else:
    print("⚠️ File slang.txt tidak ditemukan. Formalisasi slang akan dilewati.")

# === Load TF-IDF dan Model ===
try:
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model dan TF-IDF berhasil dimuat!")
except Exception as e:
    print("❌ Gagal memuat model atau TF-IDF:", e)
    model = None
    tfidf = None

# === Inisialisasi NLP tools ===
ps = PorterStemmer()
stopword_list = set(stopwords.words('english'))
stopword_list.update(["subject", "re", "fw", "fwd", "etc", "ok", "thank", "thanks", "hi", "hello", "regards", "dear"])

# === Fungsi preprocessing (sama seperti di notebook) ===
def cleaning_email(isi):
    isi = re.sub(r'@[A-Za-z0-9]+', ' ', isi)
    isi = re.sub(r'#[A-Za-z0-9]+', ' ', isi)
    isi = re.sub(r'http\S+', ' ', isi)
    isi = re.sub(r'[0-9]+', ' ', isi)
    isi = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", isi)
    return isi.strip()

def convertToSlangword(tokens):
    return [slang_dict.get(word.lower(), word.lower()) for word in tokens]

def preprocess_single_text(text):
    text = cleaning_email(text)
    text = text.encode('ascii', 'ignore').decode('ascii')  # hapus emoji/non-ascii
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # hapus karakter berulang >2
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = convertToSlangword(tokens)
    tokens = [w for w in tokens if w not in stopword_list]
    tokens = [ps.stem(w) for w in tokens]
    return " ".join(tokens)

# === Prediksi ===
def predict_phishing(text):
    if model is None or tfidf is None:
        return None, 0.0

    try:
        # Preprocessing
        clean_text = preprocess_single_text(text)
        # Transform ke TF-IDF
        tfidf_vec = tfidf.transform([clean_text])
        # Prediksi
        pred = model.predict(tfidf_vec)[0]
        # Probabilitas (jika model mendukung)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(tfidf_vec)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0
        return int(pred), confidence
    except Exception as e:
        print("Error prediksi:", e)
        return None, 0.0

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Teks email tidak boleh kosong'}), 400

    pred_class, conf = predict_phishing(email)
    if pred_class is None:
        return jsonify({'error': 'Model belum siap atau terjadi kesalahan internal'}), 500

    # Mapping label: 0 → Aman, 2 → Phishing
    result = "Aman" if pred_class == 0 else "Phishing"
    confidence = round(conf * 100, 2)

    return jsonify({
        'result': result,
        'confidence': confidence
    })

# === Jalankan ===
if __name__ == '__main__':
    app.run(debug=True)