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
from google.cloud import storage
import tempfile
import logging

# Initialize the Flask application
flask_app = Flask(__name__)

# Initialize the storage client
storage_client = storage.Client()

# Define the download_blob function
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    logging.info(f"Starting download of '{source_blob_name}' from bucket '{bucket_name}' to '{destination_file_name}'")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Finished download of '{source_blob_name}'")

# Define the bucket name and model names
bucket_name = 'inswapper'
inswapper_model_blob_name = 'inswapper_128.onnx'
gfpgan_model_blob_name = 'GFPGANv1.4.pth'
gfpgan_weights_blob_names = ['gfpgan/weights/parsing_parsenet.pth', 'gfpgan/weights/detection_Resnet50_Final.pth']

# Define the destination paths for the models
inswapper_model_path = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False).name
gfpgan_model_path = tempfile.NamedTemporaryFile(suffix='.pth', delete=False).name
gfpgan_weights_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(gfpgan_weights_dir, 'weights'), exist_ok=True)

# Download the model files from GCS to the local filesystem
download_blob(bucket_name, inswapper_model_blob_name, inswapper_model_path)
download_blob(bucket_name, gfpgan_model_blob_name, gfpgan_model_path)
for weight_blob_name in gfpgan_weights_blob_names:
    weight_name = os.path.basename(weight_blob_name)
    weight_path = os.path.join(gfpgan_weights_dir, 'weights', weight_name)
    download_blob(bucket_name, weight_blob_name, weight_path)

# Initialize the FaceAnalysis application with the downloaded model
face_analysis_app = FaceAnalysis(name='buffalo_l', root=os.path.dirname(inswapper_model_path))
face_analysis_app.prepare(ctx_id=-1, det_size=(640, 640))

# Initialize GFPGAN for face restoration with the downloaded model file
gfpgan = GFPGANer(
    model_path=gfpgan_model_path,
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

         # Load the face swapper model from the downloaded path
        swapper = insightface.model_zoo.get_model(str(inswapper_model_path), download=False, download_zip=False)

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
    flask_app.run(debug=True, host='0.0.0.0', port=5000)  # Comment out or remove this line
    # pass  # Or remove the entire if block
