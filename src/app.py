from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import logging
import csv
import traceback
from model_training import train_all_models, format_metrics_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def detect_delimiter(file):
    try:
        sample = file.read(2048).decode('utf-8')  # Read a larger sample and decode to string
        file.seek(0)  # Reset file pointer to the beginning
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        if delimiter:
            return delimiter
    except Exception as e:
        logger.error(f"Error detecting delimiter: {str(e)}")
    return ','  # Default to comma if detection fails

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        logger.info(f"Received file: {file.filename}")
        
        if not file or file.filename == '':
            logger.error("No selected file")
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            logger.error("Invalid file type")
            return jsonify({'success': False, 'error': 'Please upload a CSV file'}), 400

        try:
            # Detect the delimiter
            delimiter = detect_delimiter(file)

            # Read the CSV file with the detected delimiter
            data = pd.read_csv(file, delimiter=delimiter)
            logger.info(f"Successfully read CSV file with shape: {data.shape}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({'success': False, 'error': 'Error reading CSV file. Please check the file format.'}), 400
        
        if data.empty:
            logger.error("Empty CSV file")
            return jsonify({'success': False, 'error': 'The uploaded CSV file is empty'}), 400
        
        # Get the last column as target
        target_column = data.columns[-1]
        logger.info(f"Using {target_column} as target column")
        
        # Train models and get metrics
        try:
            models, metrics = train_all_models(data, target_column)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error during model training',
                'details': str(e)
            }), 500
        
        # Format metrics for response
        try:
            formatted_metrics = format_metrics_response(metrics)
            logger.info(f"Formatted metrics: {formatted_metrics}")
        except Exception as e:
            logger.error(f"Error formatting metrics: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error formatting metrics',
                'details': str(e)
            }), 500
        
        return jsonify({
            'success': True,
            'metrics': formatted_metrics,
            'message': 'Analysis completed successfully',
            'target_column': target_column
        })
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred during processing',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)