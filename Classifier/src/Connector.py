import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import base64
from io import BytesIO

class PlantPredictor:
    """
    This class allows users to upload images, and the system will predict the plant type using VGG16 pre-trained model, returning the uploaded images and the prediction results.
    """
    
    def __init__(self, model_path='transfer_model.h5', class_indices_path='class_indices.json'):
        """
        Initialize the class object
        
        Parameters:
            model_path (str): the path to the pre-trained model
            class_indices_path (str): the path to the class indices JSON file
        """
        # Load the pre-trained model
        self.model = load_model(model_path)
        
        # Load the class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
            
        # Create a mapping from class index to class name
        self.index_to_class = {v: k for k, v in self.class_indices.items()}
    
    def predict(self, image_data, return_top=5):
        """
        Predict the plant type from the uploaded image data
        
        Parameters:
            image_data (bytes): the binary data of the uploaded image
            return_top (int): the number of top predictions to return
            
        Returns:
            dict: a dictionary containing the processed image data(base64 encoded image) and prediction results
        """
        img = image.load_img(BytesIO(image_data), target_size=(64, 64))
        
        # Image preprocessing
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        
        preds = self.model.predict(x)[0]
        
        # Get the top N predictions
        top_indices = preds.argsort()[-return_top:][::-1]
        results = []
        
        for idx in top_indices:
            plant_name = self.index_to_class[idx]
            confidence = float(preds[idx])
            results.append({
                "plant_name": plant_name,
                "confidence": confidence
            })
        
        # Convert the image to base64 format
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return {
            "image_base64": image_base64, # Base64 encoded image(for API response)
            "predictions": results,
            "top_prediction": {
                "plant_name": self.index_to_class[top_indices[0]],
                "confidence": float(preds[top_indices[0]])
            }
        }
    
    async def process_upload(self, uploaded_file, return_top=5):
        """
        Process the uploaded file and return the prediction results
        
        Parameters:
            uploaded_file: the uploaded file object
            return_top (int): the number of top predictions to return
            
        Returns:
            dict: a dictionary containing the processed image data(base64 encoded image) and prediction results
        """
        # Read the uploaded file
        image_data = await uploaded_file.read()
        
        return self.predict(image_data, return_top)