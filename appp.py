import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from keras_facenet import FaceNet
from mtcnn import MTCNN
import pickle

# Load models
embedder = FaceNet()
detector = MTCNN()

clf = pickle.load(open("svm_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Extract face
def extract_face(frame):
    results = detector.detect_faces(frame)
    if len(results) == 0:
        return None, None
    
    x, y, w, h = results[0]['box']
    face = frame[y:y+h, x:x+w]
    return face, (x, y, w, h)

# Embedding
def get_embedding(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32') / 255
    emb = embedder.embeddings([face])
    return emb[0]

# Streamlit Video Transformer
class FaceRecognition(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        face, box = extract_face(img)
        if face is not None:
            emb = get_embedding(face)
            pred = clf.predict([emb])[0]
            name = encoder.inverse_transform([pred])[0]

            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img

# UI
st.title("Real-Time Face Recognition")
st.write("Webcam live detection using FaceNet + SVM")

webrtc_streamer(key="face-recognition", video_transformer_factory=FaceRecognition)

