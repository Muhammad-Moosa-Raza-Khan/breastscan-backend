import os
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- CONFIGURATION ---
# Use /tmp for cloud environments (they often don't allow writing to other folders)
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- MODEL ROUTER ---
# The path must be relative to where the script runs
MODEL_FILES = {
    'mammo': 'models/mammo_model.h5',
    'mri': 'models/mri_model.h5',
    'ultra': 'models/ultra_model.h5',
    'histo': 'models/histo_model.h5'
}

loaded_models = {}

def load_all_models():
    print(" * Loading AI Models...")
    for modality, path in MODEL_FILES.items():
        if os.path.exists(path):
            try:
                loaded_models[modality] = load_model(path)
                print(f"   [SUCCESS] Loaded {modality} model.")
            except Exception as e:
                print(f"   [ERROR] Failed to load {modality}: {e}")
        else:
            print(f"   [WAITING] No file found for {modality} at {path}")

# Load models immediately when app starts
load_all_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    # Resize to match your training (224x224)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/', methods=['GET'])
def home():
    return "BreastScan AI Server is Running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Validation
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 2. Router Logic
    modality = request.form.get('type', 'mammo')

    if modality not in loaded_models:
        return jsonify({'error': f'Model for {modality} is not available.'}), 503

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 3. Predict
            model = loaded_models[modality]
            processed_img = preprocess_image(filepath)
            prediction = model.predict(processed_img)

            # 4. Cleanup (Delete file to save space on cloud)
            os.remove(filepath)

            # 5. Format Result
            confidence_score = float(prediction[0][0])

            if confidence_score > 0.5:
                result_text = "Malignant (Abnormal)"
                confidence = f"{confidence_score * 100:.1f}%"
            else:
                result_text = "Benign (Normal)"
                confidence = f"{(1 - confidence_score) * 100:.1f}%"

            return jsonify({
                'result': result_text,
                'confidence': confidence,
                'modality': modality
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    # Gunicorn will handle the port in production, but this is safe to keep
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)