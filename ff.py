import cv2
import tensorflow as tf
import numpy as np
import time
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

DATADIR = 'data'
CATEGORIES = os.listdir(DATADIR)

sample_frames = 150  # Set the number of frames to capture
frame_counter = 0
image_samples = []

while frame_counter < sample_frames:
    ret, img = cam.read()
    img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w + 50, y + h + 50), (255, 0, 0), 2)
        im = gray[y:y + h, x:x + w]

    cv2.imshow('image', img)

    if 'im' in locals() and frame_counter < sample_frames:
        im_array = cv2.resize(im, (150, 150))
        im_array = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        im_array = np.expand_dims(im_array, axis=0)  # Add batch dimension
        image_samples.append(im_array)
        frame_counter += 1

    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()

# Convert the list of image samples to a numpy array
image_samples = np.concatenate(image_samples, axis=0)

# Load your model
model = tf.keras.models.load_model("CNN.model")

# Make prediction on the entire sample
predictions = model.predict(image_samples)
prediction = list(predictions[0])
print(prediction)
t=prediction.index(max(prediction))
print(CATEGORIES[prediction.index(max(prediction))])



