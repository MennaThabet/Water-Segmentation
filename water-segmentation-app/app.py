import os
import io
from flask import Flask, request, jsonify, send_file, render_template
import torch
import numpy as np
from PIL import Image

from model import load_model
from inference import load_normalization_stats, preprocess_tif, predict_mask

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "weights/best_pretrained_full_finetune.pth"
MEAN_PATH = "weights/band_mean_train.npy"
STD_PATH = "weights/band_std_train.npy"

model = load_model(WEIGHTS_PATH, DEVICE)
band_mean, band_std = load_normalization_stats(MEAN_PATH, STD_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        tensor = preprocess_tif(save_path, band_mean, band_std)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

    mask = predict_mask(model, tensor, DEVICE)

    # Return as PNG image by default
    mask_img = Image.fromarray(mask)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


@app.route("/predict_json", methods=["POST"])
def predict_json():
    """Alternative endpoint returning the mask as a JSON array instead of an image."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        tensor = preprocess_tif(save_path, band_mean, band_std)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400

    mask = predict_mask(model, tensor, DEVICE)
    water_pixel_pct = float((mask > 0).mean() * 100)

    return jsonify({
        "mask": (mask // 255).tolist(),  # binary 0/1 array
        "water_pixel_percentage": round(water_pixel_pct, 2)
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)