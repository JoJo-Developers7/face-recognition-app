import cv2
import numpy as np
import os

# Create directory if not exist
dirpath = "data"
os.makedirs(dirpath, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

face_data = []
skip = 0
name = input("Enter your name: ")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = sorted(faces, key=lambda f: f[2]*f[3])

    for (x, y, w, h) in faces[-1:]:
        face_section = gray[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100))

        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        if skip % 10 == 0:
            face_data.append(face_section)

    skip += 1
    cv2.imshow("Collecting Face Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

np.save(os.path.join(dirpath, f"{name}.npy"), face_data)

cap.release()
cv2.destroyAllWindows()

print(f"[INFO] Saved {len(face_data)} images for {name}")
