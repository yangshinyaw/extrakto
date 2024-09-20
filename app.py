import os
from flask import Flask, request, jsonify, render_template, send_file, g
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import openpyxl
from openpyxl import Workbook
import tempfile

app = Flask(__name__)

# Set a file size limit of 16 MB for uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize TrOCR model on first use, using Flask's global `g` object for persistence
def get_trocr_model():
    if 'trocr_model' not in g:
        g.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
        g.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return g.processor, g.model

# Error handler for file size exceeding limit
@app.errorhandler(413)
def handle_large_file(error):
    return jsonify({'error': 'File size exceeds the 16MB limit'}), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        max_size = (1000, 1000)
        img.thumbnail(max_size)
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    # Retrieve the model and processor
    processor, model = get_trocr_model()

    # Process the image using TrOCR
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({'text': predicted_text})

@app.route('/save_to_excel', methods=['POST'])
def save_to_excel():
    data = request.json
    if 'words' not in data:
        return jsonify({'error': 'No words provided'}), 400

    words = data['words']
    combined_words = " ".join(words)

    # Create a new Excel workbook in a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_file:
        excel_file_path = tmp_file.name

        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted Words"
        ws.cell(row=1, column=1, value=combined_words)

        wb.save(excel_file_path)

    response = send_file(excel_file_path, as_attachment=True, download_name="extracted_words.xlsx")
    
    # Delete the file after sending it to avoid storage accumulation
    os.remove(excel_file_path)

    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
