import cv2
import tensorflow as tf
import numpy as np
import os
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
import csv

app = Flask(__name__)

# Load the face cascade and the pre-trained model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("CNN.model")

# Define paths
history_csv = "history.csv"
DATADIR = 'dataset'



@app.route('/')
def index():
    links = [
        {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "thumbnail": "static/images/1.jpeg"},
        {"url": "https://www.youtube.com/watch?v=3fumBcKC6RE", "thumbnail": "static/images/2.jpeg"},
        {"url": "https://www.youtube.com/watch?v=xyz12345", "thumbnail": "static/images/3.jpeg"},
        {"url": "https://www.youtube.com/watch?v=abc67890", "thumbnail": "static/images/4.jpeg"}
    ]
    return render_template('index.html', links=links)


# Function to save history
def save_history(url):
    with open(history_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([url, time.strftime("%Y-%m-%d %H:%M:%S")])

# Function to predict age category
def predict_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        im = gray[y:y + h, x:x + w]
        im_array = cv2.resize(im, (50, 50))

        # Convert grayscale to RGB by stacking the grayscale image 3 times
        im_array = np.stack((im_array,) * 3, axis=-1)

        im_array = np.expand_dims(im_array, axis=0)  # Add batch dimension

        # Predict the age category
        predictions = model.predict(im_array)
        t = np.argmax(predictions[0])  # Get the index of the highest prediction
        return t

    return -1  # No face detected

@app.route('/check_age', methods=['POST'])
def check_age():
    if request.method == 'POST':
        # Capture frame from webcam
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()

        if ret:
            age_category = predict_age(frame)
            
            if age_category < 3:  # Assuming 0 is for under 18
                print("Predicted Age Category: Under 18 (Access Denied)")
                return jsonify({"status": "denied"})
            else:
                url = request.form['url']
                save_history(url)
                print(f"Predicted Age Category: {age_category} (Access Allowed)")
                return jsonify({"status": "allowed", "url": url})
    
    return jsonify({"status": "error"})

if __name__ == '__main__':
    app.run(debug=True)

