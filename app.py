import os
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
import openpyxl
from openpyxl import Workbook
import tempfile

app = Flask(__name__)

# Initialize TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert("RGB")

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
    combined_words = " ".join(words)  # Merge words into a single line

    # Define the path to the Excel file
    excel_file_path = 'extracted_words.xlsx'

    # Check if the Excel file already exists
    if os.path.exists(excel_file_path):
        # Open the existing Excel file
        wb = openpyxl.load_workbook(excel_file_path)
        ws = wb.active
    else:
        # Create a new Excel file and set the active sheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Extracted Words"

    # Find the next empty row
    next_row = ws.max_row + 1

    # Write the merged words to the next row
    ws.cell(row=next_row, column=1, value=combined_words)

    # Save the Excel file
    wb.save(excel_file_path)

    # Return the Excel file as a response
    return send_file(excel_file_path, as_attachment=True, download_name="extracted_words.xlsx")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
