# src/train_model.py

import face_recognition
import os
import pickle

def train_model(image_dir, model_save_path):
    known_face_encodings = []
    known_face_names = []

    # Load each image file and learn face encodings
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(os.path.splitext(image_name)[0])

    # Save the face encodings and names to a file
    with open(model_save_path, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

if __name__ == "__main__":
    image_dir = './images'
    model_save_path = './models/face_encodings.pkl'
    train_model(image_dir, model_save_path)
