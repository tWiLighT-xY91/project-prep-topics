from flask import Flask, render_template, request
import librosa
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model, scaler and encoder
BASE_DIR = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(BASE_DIR, "emotion_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb"))

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    f = request.files["file"]
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    try:
        features = extract_features(file_path)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        emotion = le.inverse_transform(prediction)[0]
        os.remove(file_path)
        return render_template("index.html", prediction=emotion)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
