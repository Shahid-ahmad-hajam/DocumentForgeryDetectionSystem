"""
Flask Web Application for Image Forgery Detection
Run: python app.py
Then open: http://localhost:5000
"""

import os
import sys
import traceback
from io import BytesIO

import numpy as np
from PIL import Image, ImageChops
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf

# -----------------------
# Configuration
# -----------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

# Global variables for model
model = None
MODEL_PATH = 'outputs/ela_efficientnet_final_finetuned.h5'
IMAGE_SIZE = (224, 224)  # (width, height) for PIL.resize
ELA_QUALITY = 90
SCALE = 15

# -----------------------
# Model loader
# -----------------------
def load_model():
    """Load the trained model. Return True if loaded, False otherwise."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model loaded from {MODEL_PATH}")
            return True
        else:
            print(f"[ERROR] Model file {MODEL_PATH} not found!")
            model = None
            return False
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        model = None
        return False

# -----------------------
# Utility: ELA conversion
# -----------------------
def convert_to_ela_image(image, quality=ELA_QUALITY, scale=SCALE):
    """Convert a PIL Image to its ELA (Error Level Analysis) representation."""
    try:
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to buffer as JPEG with given quality
        buffer = BytesIO()
        image.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)

        # Reload and compute difference
        resaved = Image.open(buffer).convert('RGB')
        ela_image = ImageChops.difference(image, resaved)

        # Scale channels to make differences more visible
        def scale_channel(channel):
            return channel.point(lambda i: min(255, int(i * scale)))

        if ela_image.mode == 'RGB':
            r, g, b = ela_image.split()
            r, g, b = scale_channel(r), scale_channel(g), scale_channel(b)
            ela_image = Image.merge('RGB', (r, g, b))
        else:
            ela_image = scale_channel(ela_image)

        return ela_image

    except Exception as e:
        print(f"[ERROR] ELA conversion failed: {e}")
        traceback.print_exc()
        raise

# -----------------------
# Prediction logic
# -----------------------
def predict_image(image_file):
    """Process image and make prediction (robust handling)."""
    if model is None:
        return None, "Model not loaded. Please check server startup logs."

    try:
        print("[INFO] Starting prediction...")

        # Reset stream position
        try:
            image_file.stream.seek(0)
        except Exception:
            try:
                image_file.seek(0)
            except Exception:
                pass

        # Open image with PIL
        image = Image.open(image_file)
        print(f"[INFO] Image opened: mode={image.mode}, size={image.size}")

        # Convert to ELA and resize
        ela_img = convert_to_ela_image(image)
        print(f"[INFO] ELA conversion complete: mode={ela_img.mode}, size={ela_img.size}")

        ela_img = ela_img.resize(IMAGE_SIZE)
        print(f"[INFO] Image resized to {IMAGE_SIZE}")

        # Convert to numpy array and normalize
        # arr = np.array(ela_img).astype(np.float32) / 255.0------------------------>Changed
        arr = np.array(ela_img).astype(np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)


        """-------------------------------------------------------------------------->changed
        # print(f"[INFO] Array shape: {arr.shape}, dtype: {arr.dtype}")

        # # If grayscale => convert to 3 channels
        # if arr.ndim == 2:
        #     arr = np.stack([arr] * 3, axis=-1)
        #     print("[INFO] Converted single-channel to 3-channel")
        # if arr.shape[-1] != 3:
        #     raise ValueError(f"Expected 3 channels (H,W,3), got shape {arr.shape}")

        # arr = np.expand_dims(arr, axis=0)  # add batch
        # print(f"[INFO] Array shape with batch: {arr.shape}")

        # # Predict
        # print("[INFO] Running model.predict...")
        # raw_pred = model.predict(arr, verbose=0)"""
        #Ensure shape (H, W, 3)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels but got {arr.shape}")

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)

        # Predict
        raw_pred = model.predict(arr, verbose=0)
        print(f"[INFO] Raw model output: {raw_pred}")

        # Interpret different output shapes
        raw = np.array(raw_pred)
        pred = None
        if raw.size == 1:
            pred = float(raw.flatten()[0])
        elif raw.ndim >= 2 and raw.shape[-1] == 1:
            pred = float(raw.flatten()[0])
        elif raw.ndim >= 2 and raw.shape[-1] == 2:
            # Binary softmax: [prob_class0, prob_class1]
            pred = float(raw[0, 1])
        else:
            pred = float(raw.flatten()[0])
            print("[WARNING] Unexpected model output shape; using first element as score")

        print(f"[INFO] Interpreted prediction score: {pred}")

        # Map to label/confidence (threshold 0.5)
        label = "Real" if pred > 0.5 else "Fake"
        confidence = pred if pred > 0.5 else (1 - pred)

        result = {
            'label': label,
            'confidence': float(confidence * 100),
            'raw_score': float(pred)
        }

        print(f"[INFO] Prediction result: {result}")
        return result, None

    except Exception as e:
        error_msg = f"Prediction error: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return None, error_msg

# -----------------------
# Routes
# -----------------------
@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/results')
def results():
    """Render the results page."""
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        print("[INFO] Received prediction request")
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload PNG or JPEG'}), 400

        # If model isn't loaded, return 503 Service Unavailable
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503

        result, error = predict_image(file)
        if error:
            return jsonify({'error': error}), 500

        return jsonify(result)
    except Exception as e:
        error_msg = f"Server error: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

# -----------------------
# Error handlers
# -----------------------
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"[ERROR] Internal server error: {error}")
    traceback.print_exc()
    return jsonify({'error': 'Internal server error. Check server logs.'}), 500

# -----------------------
# Run server
# -----------------------
if __name__ == '__main__':
    print("[STARTUP] Loading model...")
    if not load_model():
        print("[STARTUP] Model failed to load. Exiting to avoid confusing 500s.")
        sys.exit(1)

    print("[STARTUP] Model loaded. Starting server...")
    print("[INFO] Make sure you have the following folder structure:")
    print("       templates/")
    print("         - index.html")
    print("         - results.html")
    print("       static/")
    print("         - css/style.css")
    print("         - js/script.js")
    print("         - js/results.js")
    
    # For local dev use debug=True; in production use a proper WSGI server
    app.run(debug=False, host='0.0.0.0', port=5000)