from flask import Flask, request, jsonify, render_template
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io

app = Flask(__name__)

# Lazy load TrOCR model (smaller version)
processor = None
model = None

def load_trocr_model():
    global processor, model
    if processor is None or model is None:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    # Lazy load the model when needed
    load_trocr_model()

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Process the image using TrOCR
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({'text': predicted_text})

@app.route('/batch_extract', methods=['POST'])
def batch_extract_text():
    # Batch processing to extract words
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    load_trocr_model()
    results = []

    for file in files:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(predicted_text)

    return jsonify({'texts': results})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
