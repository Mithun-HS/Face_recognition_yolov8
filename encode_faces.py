import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

# Load YOLOv8 face detection model (can use yolov8n for lightweight)
model = YOLO("yolov8n-face.pt")  # Trained specifically on face dataset

print("Model classes:", model.names)

def get_embedding(image):
    try:
        embedding = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Failed to get embedding: {e}")
        return None

face_db = {}
dataset_path = "dataset"

for person in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person)
    embeddings = []
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        img = cv2.imread(img_path)

        # Detect face with YOLO
        results = model(img)
        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img[y1:y2, x1:x2]
                emb = get_embedding(face)
                if emb is not None:
                    embeddings.append(emb)
    if embeddings:
        face_db[person] = np.mean(embeddings, axis=0)

# Save face_db to a file
np.save("face_db.npy", face_db)
print("Embeddings saved!")
