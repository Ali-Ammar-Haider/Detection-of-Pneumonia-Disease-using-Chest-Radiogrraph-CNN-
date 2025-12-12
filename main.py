from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import shutil

app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load trained model
MODEL_PATH = "pneumonia_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]

    # Corrected based on class_indices
    return "NORMAL" if pred >= 0.5 else "PNEUMONIA"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def upload(file: UploadFile = File(...)):
    filepath = "uploaded.jpg"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = predict_pneumonia(filepath)
    return JSONResponse(content={"prediction": result})
