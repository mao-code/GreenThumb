from fastapi import FastAPI, File, UploadFile, HTTPException
from Connector import PlantPredictor
import uvicorn
import traceback


app = FastAPI()

predictor = PlantPredictor(
    model_path='../transfer_model.h5',
    class_indices_path='../class_indices.json'
)

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    try:
        print(f"File uploaded: {file.filename}, Type: {file.content_type}")
        result = await predictor.process_upload(file)
        
        return {
            "image_base64": result["image_base64"],
            "predictions": result["predictions"],
            "top_prediction": result["top_prediction"]
        }
    except Exception as e:
        print(f"Error occured: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)