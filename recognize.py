import cv2
import numpy as np
import os
from knn import KNN

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# Load training data
data_dir = "data"
face_data = []
labels = []
names = {}
class_id = 0

for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        data_item = np.load(os.path.join(data_dir, file))
        face_data.append(data_item)
        names[class_id] = file[:-4]

        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        class_id += 1

# Prepare dataset
face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape(-1, 1)

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    for (x, y, w, h) in faces[-1:]:
        face_section = gray[y:y + h, x:x + w]
        face_section = cv2.resize(face_section, (100, 100))

        pred = KNN(face_dataset, face_labels, face_section)
        pred_name = names[int(pred)]

        cv2.putText(frame, pred_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
