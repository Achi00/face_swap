from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import requests
from io import BytesIO
import base64
from gfpgan import GFPGANer
import subprocess
import traceback
import os
from werkzeug.utils import secure_filename

# Initialize the Flask application
flask_app = Flask(__name__)

# Initialize the FaceAnalysis application
face_analysis_app = FaceAnalysis(name='buffalo_l')
face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize GFPGAN for face restoration
gfpgan = GFPGANer(
    model_path='./venv/insightface/pretrained_models/GFPGANv1.4.pth',
    upscale=2,  # Change the upscale factor based on your requirements
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

# check if face exists on users image
@flask_app.route('/detect-face', methods=['POST'])
def detect_face():
    try:
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Use InsightFace for face detection
        face_app = insightface.app.FaceAnalysis()
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        faces = face_app.get(img)

        # Check the number of detected faces
        if len(faces) > 1:
            return jsonify({"error": "Multiple faces detected. Only single-face detection is supported."}), 400

        # If only one face is detected, proceed with the normal response
        elif len(faces) == 1:
            return jsonify({"faces_detected": True})

        # If no faces are detected, return the original error message
        else:
            return jsonify({"error": "No face detected"}), 400

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500



# for face swapping
@flask_app.route('/swap-face', methods=['POST'])
def swap_face():
    try:
        # Read source (user's face) and target (generated) images from the request
        source_file = request.files['user_image']  # This should be the user's face
        target_image_url = request.form['generated_image_url']  # URL of the generated image

        # Convert the source image to OpenCV format
        source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Download and convert the target image from the URL
        response = requests.get(target_image_url)
        target_img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        target_img = cv2.imdecode(target_img_array, cv2.IMREAD_COLOR)

        # Detect faces
        source_faces = face_analysis_app.get(source_img)
        target_faces = face_analysis_app.get(target_img)
        if len(source_faces) != 1 or len(target_faces) != 1:
            # Here we send a specific message back if the face count is not exactly one
            return jsonify({
                "error": "Each image should have exactly one face.",
                "faces_detected": {
                    "source": len(source_faces),
                    "target": len(target_faces)
                }
            }), 400

        # Initialize the face swapper model
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

        # Swap the face from source image onto the target image
        swapped_img = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

        # Enhance the image using the GFPGAN model
        _, _, enhanced_img = gfpgan.enhance(swapped_img, has_aligned=False, only_center_face=False, paste_back=True)

        # Convert the enhanced result to base64 to send as JSON
        _, buffer = cv2.imencode('.jpg', enhanced_img)
        res_base64 = base64.b64encode(buffer).decode()

        return jsonify({'result_image': res_base64})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500

    
@flask_app.route('/enhance-face', methods=['POST'])
def enhance_face():
    try:
        # Save uploaded image
        file = request.files['image']
        filename = secure_filename(file.filename)  # Use the secure_filename function to validate the filename
        input_path = os.path.join('inputs/upload', filename)
        file.save(input_path)

        # Read the image
        img = cv2.imread(input_path)

        # Enhance the image using the GFPGAN model
        _, _, enhanced_img = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        # Save the enhanced image
        output_path = os.path.join('results', filename)
        cv2.imwrite(output_path, enhanced_img)

        # Encode the image to base64 to return as JSON
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return jsonify({'enhanced_image': encoded_string})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500



if __name__ == '__main__':
    flask_app.run(debug=True, host='0.0.0.0', port=5000)