import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import csv
import os
import pyttsx3  # Text-to-speech

# Initialize Text-to-Speech engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# Load face database
face_db = np.load("face_db.npy", allow_pickle=True).item()
model = YOLO("yolov8n-face.pt")

# Log file setup
log_file = "recognition_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

# Track names already logged
logged_names = set()

def get_embedding(image):
    try:
        embedding = DeepFace.represent(img_path=image, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        return np.array(embedding)
    except Exception:
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

# Start webcam
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

                if name != "Unknown":
                    status = "Present"
                    color = (0, 255, 0)  # Green box
                else:
                    status = "Unknown"
                    color = (0, 0, 255)  # Red box

                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Log only if recognized and not already logged
                if name != "Unknown" and name not in logged_names:
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")

                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, date, time, status])

                    logged_names.add(name)

                    # Announce
                    tts.say(f"{name} present")
                    tts.runAndWait()

    # Show the result frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
