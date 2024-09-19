from flask import Flask, request, jsonify, render_template
from PIL import Image
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
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

        # Quantize the model to reduce memory usage and speed up inference
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Resize image to reduce memory usage and speed up processing
    max_size = (512, 512)
    img.thumbnail(max_size, Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS

    print(f"Image processing time: {time.time() - start_time} seconds")

    # Process the image using TrOCR
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():  # Disable gradient computation to save memory
        generated_ids = model.generate(pixel_values)

    print(f"Model inference time: {time.time() - start_time} seconds")

    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Total request handling time: {time.time() - start_time} seconds")

    return jsonify({'text': predicted_text})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable from Render
    app.run(host='0.0.0.0', port=port)
