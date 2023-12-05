from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import requests
import base64
from gfpgan import GFPGANer
import traceback
import os
from werkzeug.utils import secure_filename
import tempfile
import logging

# Flask application initialization
app = Flask(__name__)

# Set the model directory
MODEL_STORE_DIR = "model_store"

# Define model paths globally
inswapper_model_path = os.path.join(MODEL_STORE_DIR, 'inswapper_128.onnx')
gfpgan_model_path = os.path.join(MODEL_STORE_DIR, 'GFPGANv1.4.pth')

# Global variables to hold the models
face_analysis_app = None
gfpgan = None

def get_face_analysis_app():
    global face_analysis_app
    if face_analysis_app is None:
        inswapper_model_path = os.path.join(MODEL_STORE_DIR, 'inswapper_128.onnx')
        face_analysis_app = FaceAnalysis(name='buffalo_l', root=os.path.dirname(inswapper_model_path))
        face_analysis_app.prepare(ctx_id=-1, det_size=(640, 640))
    return face_analysis_app

def get_gfpgan():
    global gfpgan
    if gfpgan is None:
        gfpgan_model_path = os.path.join(MODEL_STORE_DIR, 'GFPGANv1.4.pth')
        gfpgan = GFPGANer(model_path=gfpgan_model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)
    return gfpgan

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        face_app = get_face_analysis_app()
        faces = face_app.get(img)

        if len(faces) > 1:
            return jsonify({"error": "Multiple faces detected. Only single-face detection is supported."}), 400
        elif len(faces) == 1:
            return jsonify({"faces_detected": True})
        else:
            return jsonify({"error": "No face detected"}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500

@app.route('/swap-face', methods=['POST'])
def swap_face():
    try:
        source_file = request.files['user_image']
        target_image_url = request.form['generated_image_url']

        source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        response = requests.get(target_image_url)
        target_img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        target_img = cv2.imdecode(target_img_array, cv2.IMREAD_COLOR)

        face_app = get_face_analysis_app()
        source_faces = face_app.get(source_img)
        target_faces = face_app.get(target_img)

        if len(source_faces) != 1 or len(target_faces) != 1:
            return jsonify({
                "error": "Each image should have exactly one face.",
                "faces_detected": {
                    "source": len(source_faces),
                    "target": len(target_faces)
                }
            }), 400

        # Use the global path for the swapper model
        global inswapper_model_path
        swapper = insightface.model_zoo.get_model(str(inswapper_model_path), download=False, download_zip=False)

        swapped_img = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

        gfpgan = get_gfpgan()
        _, _, enhanced_img = gfpgan.enhance(swapped_img, has_aligned=False, only_center_face=False, paste_back=True)

        _, buffer = cv2.imencode('.jpg', enhanced_img)
        res_base64 = base64.b64encode(buffer).decode()

        return jsonify({'result_image': res_base64})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500

@app.route('/enhance-face', methods=['POST'])
def enhance_face():
    try:
        file = request.files['image']
        filename = secure_filename(file.filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, filename)
            file.save(file_path)
            img = cv2.imread(file_path)

            gfpgan = get_gfpgan()
            _, _, enhanced_img = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

            _, buffer = cv2.imencode('.jpg', enhanced_img)
            encoded_string = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'enhanced_image': encoded_string})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
