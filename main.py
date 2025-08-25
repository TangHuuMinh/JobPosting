from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import gdown

app = Flask(__name__)

# ---- Google Drive folder link ----
FOLDER_URL = "https://drive.google.com/drive/folders/1xfBJm-FNwi34W0ySzQIovtLxycUYGqd5?usp=drive_link"
MODEL_DIR = "./phobert_job_fraud"

# ---- Tải model từ Google Drive nếu chưa có ----
if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print("📥 Downloading model from Google Drive...")
    gdown.download_folder(FOLDER_URL, output=MODEL_DIR, quiet=False, use_cookies=False)

# ---- Load model từ thư mục đã tải ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ---- Trang chủ: nhập text ----
@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Job Fraud Detection</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center">
        <div class="bg-white p-8 rounded-lg shadow-lg max-w-2xl w-full">
            <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">🔎 Job Fraud Detection</h2>
            <form action="/predict" method="post" class="space-y-4">
                <textarea name="text" rows="5" class="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500" placeholder="Nhập văn bản tuyển dụng..."></textarea>
                <button type="submit" class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition duration-200">Dự đoán</button>
            </form>
        </div>
    </body>
    </html>
    """)

# ---- API dự đoán ----
@app.route("/predict", methods=["POST"])
def predict():
    # Lấy text từ form hoặc JSON
    if request.is_json:
        text = request.json.get("text", "")
    else:
        text = request.form.get("text", "")

    if not text:
        return jsonify({"error": "No input text"}), 400

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]
        pred_class = int(torch.argmax(outputs.logits, dim=1))

    # Trả kết quả
    result = {
        "text": text,
        "prediction": "Fraudulent" if pred_class == 1 else "Real",
        "probabilities": {"Real": probs[0], "Fraudulent": probs[1]}
    }

    # Nếu gọi từ form → hiển thị HTML
    if not request.is_json:
        return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Job Fraud Detection Result</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 min-h-screen flex items-center justify-center">
            <div class="bg-white p-8 rounded-lg shadow-lg max-w-2xl w-full">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 text-center">Kết quả dự đoán</h2>
                <div class="space-y-4">
                    <p><span class="font-semibold">Văn bản:</span> {{text}}</p>
                    <p><span class="font-semibold">Kết quả:</span> <span class="{% if prediction == 'Fraudulent' %}text-red-600{% else %}text-green-600{% endif %}">{{prediction}}</span></p>
                    <p><span class="font-semibold">Xác suất:</span> Real = {{real:.4f}}, Fraudulent = {{fraud:.4f}}</p>
                    <a href="/" class="inline-block bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition duration-200">🔙 Thử lại</a>
                </div>
            </div>
        </body>
        </html>
        """, text=text, prediction=result["prediction"],
           real=probs[0], fraud=probs[1])

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
