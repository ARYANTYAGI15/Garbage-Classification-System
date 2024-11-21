from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pre-trained model
model = load_model('Garbage_classification.h5')

# Define the class labels (based on your dataset)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'general waste']

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to preprocess a frame for real-time classification
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Resize to model's input size
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    frame_reshaped = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    return frame_reshaped

# Function to classify a single frame
def classify_frame(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

@app.route('/')
def index():
    return '''
        <h1>Classify Your Trash</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
        <br>
        <a href="/realtime">Start Real-Time Classification</a>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image and make a prediction
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    return jsonify({'material': predicted_class})

@app.route('/realtime')
def realtime_classification():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    if not cap.isOpened():
        return "Error: Could not access the camera.", 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Classify the frame
        label = classify_frame(frame)

        # Display the classification on the frame
        cv2.putText(frame, f"Class: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Garbage Classification', frame)

        # Exit the real-time classification on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Real-time classification ended.", 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
