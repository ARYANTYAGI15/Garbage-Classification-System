from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your pre-trained model
model = load_model('Garbage_classification.h5')

# Define the class labels (based on your dataset)
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Change target size if required
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image (if your model expects this)
    return img_array

@app.route('/')
def index():
    return '''
        <h1>Classify Your Trash</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
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

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
