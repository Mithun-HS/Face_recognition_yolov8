import numpy as np

face_db = np.load("face_db.npy", allow_pickle=True).item()
print("Faces stored in face_db.npy:")
print(face_db.keys())
