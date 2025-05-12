# 🌿 GreenThumb (Plant Care Assistant - AI-Driven Plant Health Diagnosis System)

This project helps plant lovers take better care of their plants using AI. It combines image analysis and multimodal large language models to offer insights into plant type, health condition, and care suggestions.

## 🔧 Features
- Upload a photo of your plant
- Add optional notes (e.g., symptoms, environment)
- Get back a detailed diagnosis and care recommendation powered by GPT-4o

## 🚀 Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn Classifier.src.API_test:app --reload

# Launch Gradio UI
python MLLM/app.py
```

## Reference
Kaggle Dataset: https://www.kaggle.com/datasets/yudhaislamisulistya/plants-type-datasets?utm_source=chatgpt.com