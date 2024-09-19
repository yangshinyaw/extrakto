from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import os
import time
import torch

app = Flask(__name__)

processor = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    global processor, model

    start_time = time.time()  # Start profiling time

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Lazy load the model and processor only when necessary
    if processor is None or model is None:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


        # Quantize the model to reduce memory usage and speed up inference
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # Retrieve the file and process it as an image
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert("L")  # Convert to grayscale to reduce size
    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid image format.'}), 400
    except Exception as e:
        return jsonify({'error': f'Error loading image: {str(e)}'}), 500

    # Resize image to reduce memory usage and speed up processing
    max_size = (384, 384)  # You can experiment with smaller sizes
    img.thumbnail(max_size, Image.LANCZOS)

    print(f"Image processing time: {time.time() - start_time} seconds")

    try:
        # Process the image using TrOCR
        pixel_values = processor(images=img, return_tensors="pt").pixel_values

        with torch.no_grad():  # Disable gradient computation to save memory
            generated_ids = model.generate(pixel_values)

        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return jsonify({'error': f'Error during model inference: {str(e)}'}), 500

    print(f"Model inference time: {time.time() - start_time} seconds")
    print(f"Total request handling time: {time.time() - start_time} seconds")

    return jsonify({'text': predicted_text})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Check PORT env variable
    print(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port)
