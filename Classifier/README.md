# Classifier Module

This directory contains the plant classification model for the GreenThumb project. The module provides both web interface and API endpoints for plant type classification using deep learning.

## Directory Structure

- `src/`: Source code for the classifier
  - `app.py`: Web application using Gradio for interactive plant classification
  - `API_test.py`: FastAPI implementation for RESTful API endpoints
  - `Connector.py`: Plant prediction service for handling image uploads and classification
  - `data_loader.py`: Utilities for loading and processing training data
  - `evaluate.py`: Code for evaluating model performance
  - `model.py`: Model architecture definition (CNN and VGG16 transfer learning)
  - `predict.py`: Command-line tool for making predictions
  - `train.py`: Training script for the model
- `transfer_model.h5`: Pre-trained model file (excluded from version control)
- `class_indices.json`: Mapping between class indices and plant names (excluded from version control)
- `base_dir/`: Raw data directory (excluded from version control)
  - `train_dir/`: Training data
  - `val_dir/`: Validation data
  - `test_dir/`: Test data

## Features

1. **Interactive Web Interface**: Using Gradio for easy plant classification through browser
2. **RESTful API**: FastAPI implementation for integration with other applications
3. **Command-line Tools**: Scripts for training, evaluation, and prediction
4. **Transfer Learning**: Option to use VGG16 pre-trained model for better performance

## Usage

### Web Interface

Run the Gradio web interface:

```bash
cd src
python app.py
```

### API Service

Start the FastAPI server:

```bash
cd src
python API_test.py
```

The API will be available at `http://localhost:8000/predict`

### Training

Train a new model:

```bash
cd src
python train.py --data_dir ../base_dir --output_model ../my_model.h5 --epochs 20
```

For transfer learning:

```bash
python train.py --data_dir ../base_dir --output_model ../transfer_model.h5 --epochs 10 --transfer --trainable_blocks block5_conv
```

### Evaluation

Evaluate model performance:

```bash
cd src
python evaluate.py --data_dir ../base_dir --model_path ../transfer_model.h5
```

### Command-line Prediction

Make predictions on a specific image:

```bash
cd src
python predict.py --model_path ../transfer_model.h5 --image_path path/to/your/image.jpg
```

## API Documentation

The API endpoint `/predict` accepts POST requests with form data containing an image file. The response includes:

```json
{
  "image_base64": "base64_encoded_image",
  "predictions": [
    {"plant_name": "rose", "confidence": 0.95},
    {"plant_name": "tulip", "confidence": 0.03}
  ],
  "top_prediction": {"plant_name": "rose", "confidence": 0.95}
}
```

## Excluded Files

The following files are excluded from version control:
- Virtual environment (`venv/`)
- Model files (`.h5`, `.hdf5`)
- Raw data (`base_dir/`)
- Class indices file (`class_indices.json`)
- Cached Python files (`__pycache__/`)