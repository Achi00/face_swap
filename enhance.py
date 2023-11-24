from flask import Flask, request, send_from_directory
import subprocess
import os

app = Flask(__name__)

@app.route('/enhance-face', methods=['POST'])
def enhance_face():
    # Save uploaded image
    file = request.files['image']
    input_path = 'inputs/upload/your_image.jpg'
    file.save(input_path)

    # Run the enhancement script
    subprocess.run([
        'python', 'enhance.py',
        '-i', 'inputs/upload',
        '-o', 'results',
        '-v', '1.3',
        '-s', '2',
        '--bg_upsampler', 'realesrgan'
    ], check=True)

    # Return the enhanced image
    return send_from_directory('results', 'your_image.jpg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)