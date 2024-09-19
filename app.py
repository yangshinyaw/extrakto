from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import os

app = Flask(__name__)

processor = None
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    global processor, model

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Lazy load the model and processor only when necessary
    if processor is None or model is None:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Process the image using TrOCR
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({'text': predicted_text})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT env variable from Render
    app.run(host='0.0.0.0', port=port)
