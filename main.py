# app/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from models.efficient_net import load_model
from utils.utils import preprocess_image
import json
import uvicorn

app = FastAPI(title="FER2013 Emotion Recognition API")

# Load class_to_idx and create idx_to_class mapping
with open('./models/class_to_idx.json', 'r') as f:
    class_to_idx = json.load(f)

# Create index to class mapping
idx_to_class = {int(v): k for k, v in class_to_idx.items()}  # Ensure keys are integers

# Load the model at startup
@app.on_event("startup")
def startup_event():
    global model
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './models/efficient_net.pth'  # Path to your .pth file
    model = load_model(model_path, device)
    print(f"Model loaded on {device}.")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict the emotion from an uploaded image.
    
    Args:
        file (UploadFile): Image file uploaded by the user.
    
    Returns:
        JSONResponse: Predicted class and confidence score.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess the image (detect and crop face)
        input_tensor = preprocess_image(image_bytes)
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # Get class name
        predicted_class = idx_to_class[predicted.item()]
        confidence_score = confidence.item()

        # Return the prediction
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence_score
        })
    
    except ValueError as ve:
        # Handle face detection errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Handle other exceptions
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "FER2013 Emotion Recognition API is up and running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
