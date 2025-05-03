import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
import json

model = load_model('transfer_model.h5')

# Load the class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Create a mapping from class index to class name
index_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img):
    img = img.resize((64, 64))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    return {index_to_class[i]: float(preds[i]) for i in range(len(preds))}

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    title='Plant type Classifier',
    description='Upload picture to show its type.'
)

if __name__=='__main__':
    iface.launch()