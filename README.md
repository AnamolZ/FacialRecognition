# Facial Recognition

This facial recognition project is designed to train a model on a set of labeled images and then use the model to recognize and identify faces in real-time through a webcam feed.

## Project Structure

The project is organized as follows:

- `images/`: Contains labeled images for training. Each image should be named after the person in the image (e.g., `ElonMusk.png`).
- `src/`: Contains the source code.
  - `train_model.py`: Script to train the face recognition model.
  - `recognize_faces.py`: Script for real-time face recognition.
- `models/`: Stores the trained model file.
- `requirements.txt`: Lists the Python dependencies.

## Setup

### Prerequisites

- Python 3.6 or higher
- A webcam for real-time face recognition

### Installation

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/AnamolZ/FacialRecognition.git
   cd FacialRecognition
   ```

2. Set up a virtual environment (optional but recommended):
   ```
   python -m venv env
   .\env\Scripts\activate  # On Windows
   source env/bin/activate  # On Unix or MacOS
   ```

. Set up dlib installation:
   ```
   pip install dlib_file_name.whl
   pip install cmake
   pip install face_recognition
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Place your labeled image files in the `images/` directory. Ensure each image is named after the person in the image.

2. Run `train_model.py` to train the model. This will create a model file in the `models/` directory:
   ```
   python src/train_model.py
   ```

3. Run `recognize_faces.py` to start the real-time face recognition:
   ```
   python src/recognize_faces.py
   ```

## Contributing

Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

Specify your license here. Common choices include MIT, GPL, and Apache 2.0.
```
