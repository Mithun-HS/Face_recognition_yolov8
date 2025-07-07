import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

face_db = np.load("face_db.npy", allow_pickle=True).item()
model = YOLO("yolov8n-face.pt")

def get_embedding(image):
    try:
        embedding = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        return None

def recognize_face(embedding, threshold=0.5):
    best_match = "Unknown"
    highest_sim = threshold
    for name, db_emb in face_db.items():
        sim = cosine_similarity([embedding], [db_emb])[0][0]
        if sim > highest_sim:
            highest_sim = sim
            best_match = name
    return best_match

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            emb = get_embedding(face)

            if emb is not None:
                name = recognize_face(emb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
