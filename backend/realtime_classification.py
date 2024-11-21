import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("backend/Garbage_classification.h5")
class_labels = {0: 'Cardboard', 1: 'Glass', 2: 'Metal', 3: 'Paper', 4: 'Plastic', 5: 'Trash', 6: 'General Waste'}

def preprocess_image(image):
    # Resize to model input shape
    img = cv2.resize(image, (224, 224))  # Update size based on your model
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_frame(frame):
    processed_img = preprocess_image(frame)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class]

def realtime_classification():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Classify the frame
        label = classify_frame(frame)

        # Display the classification on the frame
        cv2.putText(frame, f"Class: {label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Garbage Classification', frame)

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
