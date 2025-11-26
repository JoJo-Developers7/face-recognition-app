# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import os
import numpy as np
from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mtcnn import MTCNN
from PIL import Image
import pickle

# -----------------------------
# LOAD MODELS
# -----------------------------
embedder = FaceNet()
detector = MTCNN()

# -----------------------------
# FACE EXTRACTION
# -----------------------------
def extract_face(img_path):
    img = Image.open(img_path).convert('RGB')
    pixels = np.array(img)

    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']
    face = pixels[y:y+h, x:x+w]
    face = Image.fromarray(face).resize((160, 160))

    return np.array(face)


# -----------------------------
# GET EMBEDDING
# -----------------------------
def get_embedding(face_array):
    face_array = face_array.astype('float32') / 255.0
    embedding = embedder.embeddings([face_array])
    return embedding[0]


# -----------------------------
# LOAD DATASET & CREATE EMBEDDINGS
# -----------------------------
dataset_path = "dataset"

X, y = [], []

for person in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, person)

    for img_name in os.listdir(folder):
        path = os.path.join(folder, img_name)

        face = extract_face(path)
        if face is None:
            continue

        emb = get_embedding(face)
        X.append(emb)
        y.append(person)

X = np.asarray(X)
y = np.asarray(y)

# -----------------------------
# LABEL ENCODING
# -----------------------------
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# -----------------------------
# TRAIN CLASSIFIER
# -----------------------------
clf = SVC(kernel='linear', probability=True)
clf.fit(X, y_enc)

# -----------------------------
# SAVE MODELS
# -----------------------------
os.makedirs("models", exist_ok=True)

np.savez("models/embeddings.npz", X=X, y=y)
pickle.dump(clf, open("models/svm_model.pkl", "wb"))
pickle.dump(encoder, open("models/label_encoder.pkl", "wb"))

print("Training completed! Models saved in /models folder.")
