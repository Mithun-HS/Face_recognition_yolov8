import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8n-face.pt if you trained for faces

# Load known face embeddings
face_db = np.load("face_db.npy", allow_pickle=True).item()

# Attendance function
def mark_attendance(name):
    filename = "attendance.xlsx"
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        if os.path.exists(filename):
            df = pd.read_excel(filename)
        else:
            df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

        if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
            new_entry = pd.DataFrame([[name, date_str, time_str, "Present"]], columns=df.columns)
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel(filename, index=False)
            print(f"✅ Attendance marked for {name}")
        else:
            print(f"ℹ️ {name} already marked today")

    except Exception as e:
        print(f"❌ Error: {e}")

# Function to find best match from DB
def find_best_match(face_embedding, face_db, threshold=7.5):
    min_distance = float("inf")
    best_match = "Unknown"

    for name, db_embedding in face_db.items():
        distance = np.linalg.norm(face_embedding - db_embedding)
        if distance < min_distance:
            min_distance = distance
            best_match = name

    print(f"🧠 Min distance: {min_distance:.3f}")
    return best_match if min_distance < threshold else "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        try:
            embedding = DeepFace.represent(face_crop, model_name='Facenet', enforce_detection=False)[0]["embedding"]
            embedding = np.array(embedding)

            name = find_best_match(embedding, face_db)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = name

            # Mark attendance only once
            if name != "Unknown":
                mark_attendance(name)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception as e:
            print(f"Face processing error: {e}")

    cv2.imshow("YOLOv8 Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
