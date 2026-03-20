from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# This allows your index.html to talk to this Python script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the brain you just trained
model = tf.keras.models.load_model('weather_classifier.h5')
CLASS_NAMES = ['Cloudy', 'Rainy', 'Sunny']

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert('RGB')
        image = image.resize((150, 150))
        
        # This division by 255.0 is the "Secret Sauce" - it MUST match the training
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        score = np.max(predictions)
        label = CLASS_NAMES[np.argmax(predictions)]
        
        return {
            "status": "success",
            "prediction": label,
            "confidence": round(float(score) * 100, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)