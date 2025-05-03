import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--image_path', required=True)
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Load the class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # Create a mapping from class index to class name
    index_to_class = {v: k for k, v in class_indices.items()}

    # Load and preprocess the image
    img = image.load_img(args.image_path, target_size=(64,64))
    x = image.img_to_array(img) / 255.
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    
    idx = np.argmax(preds)
    label = index_to_class[idx]
    prob = preds[idx]
    print(f'Prediction: {label} ({prob:.3f})')

    # Display the prediction results
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()