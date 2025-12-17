from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# Загружаем модель один раз при старте сервера
model = YOLO("results_of_model/YOLO_480_results/best.pt") 

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.3):
    # Читаем изображение из запроса
    content = await file.read()
    img = Image.open(io.BytesIO(content))

    # Инференс
    results = model.predict(img, conf=conf)
    
    # Формируем список найденных объектов
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "name": model.names[int(box.cls[0].item())],
            "conf": float(box.conf[0].item()),
            "bbox": [x1, y1, x2, y2]
        })
    
    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)