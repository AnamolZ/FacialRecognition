import os
import cv2
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
import joblib
import time

def obtain_data(model_filename, image_directory):
    if os.path.exists(model_filename):
        data = joblib.load(model_filename)
        encodings, names = data['encodings'], data['names']
    else:
        encodings, names = extract_face_encodings(image_directory)
        data = {'encodings': encodings, 'names': names}
        joblib.dump(data, model_filename)
    return encodings, names

def prepare_model(encodings, names):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(encodings, names)
    return model

def recognize(frame, model, encodings, names, threshold):
    locations = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, locations)

    recognized = []
    for encoding in encodings:
        distances = face_recognition.face_distance(encodings, encoding)
        min_idx = distances.argmin()
        min_distance = distances[min_idx]

        if min_distance <= threshold:
            name = names[min_idx]
            recognized.append(name)
        else:
            recognized.append("Unknown")

    return recognized

def display(frame, recognized, duration):
    start_time = time.time()

    for name in recognized:
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return start_time

def process(model, encodings, names, threshold, frequency, duration):
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FPS, 144)

    frame_count = 0
    start_time = None

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture frame.")
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        if frame_count % frequency == 0:
            recognized = recognize(frame, model, encodings, names, threshold)

            if not recognized:
                recognized = ["Unknown"]

            start_time = display(frame, recognized, duration)

        if start_time is not None:
            time_since = time.time() - start_time
            if time_since > duration:
                start_time = None

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_directory = 'training_dataset'
    model_filename = 'training_dataset/model/face_recognition_model.joblib'
    frequency = 144
    threshold = 0.6
    duration = 5

    encodings, names = obtain_data(model_filename, image_directory)
    model = prepare_model(encodings, names)
    process(model, encodings, names, threshold, frequency, duration)
