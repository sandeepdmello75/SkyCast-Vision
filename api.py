from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os

# Smart Import for TFLite (Render vs Laptop)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = FastAPI()

# CRITICAL: This allows your HTML to talk to your Python code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Lite Model
MODEL_PATH = "model.tflite"
interpreter = None

if os.path.exists(MODEL_PATH):
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    print(f"ERROR: {MODEL_PATH} not found!")

CLASS_NAMES = ['Cloudy', 'Rainy', 'Sunny']

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and resize image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB').resize((150, 150))
        
        # Preprocess for AI
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run Prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(output_data[0])
        confidence = float(output_data[0][index]) * 100
        
        return {
            "status": "success", 
            "prediction": CLASS_NAMES[index], 
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}