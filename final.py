import cv2
import tensorflow as tf
import numpy as np
import time
import os
import sys
import csv
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *

# Load face cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
cam = cv2.VideoCapture(0)

# Data directory and categories
DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR)

# Frame capturing settings
sample_frames = 150
frame_counter = 0
image_samples = []

# Capture frames from the camera
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
t = prediction.index(max(prediction))
print(CATEGORIES[prediction.index(max(prediction))])
print(t)

# Define paths for CSV files
block_18_csv = "block_18.csv"
block_20_csv = "block_20.csv"
history_csv = "history.csv"

# Function to load sites from CSV
def load_sites_from_csv(csv_file):
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        sites = [row[0] for row in csv_reader]
    return sites

# Choose the CSV file based on age prediction
print(t)
if t < 1:
    sites_to_block = load_sites_from_csv(block_18_csv)
else:
    sites_to_block = load_sites_from_csv(block_20_csv)


# GUI part
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://google.com'))
        self.setCentralWidget(self.browser)
        self.showMaximized()

        # Navbar
        navbar = QToolBar()
        self.addToolBar(navbar)

        back_btn = QAction('Back', self)
        back_btn.triggered.connect(self.browser.back)
        navbar.addAction(back_btn)

        forward_btn = QAction('Forward', self)
        forward_btn.triggered.connect(self.browser.forward)
        navbar.addAction(forward_btn)

        reload_btn = QAction('Reload', self)
        reload_btn.triggered.connect(self.browser.reload)
        navbar.addAction(reload_btn)

        home_btn = QAction('Home', self)
        home_btn.triggered.connect(self.navigate_home)
        navbar.addAction(home_btn)

        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        navbar.addWidget(self.url_bar)

        self.browser.urlChanged.connect(self.update_url)

    def navigate_home(self):
        self.browser.setUrl(QUrl('http://google.com'))

    def navigate_to_url(self):
        url = self.url_bar.text()
        if url not in sites_to_block:
            url="https://"+url
            print("hi")
            print(sites_to_block)
            self.save_history(url)
        else:
            url="https://www.google.co.in/error"
            print("bye")
        self.browser.setUrl(QUrl(url))
        
        # Save the history to CSV
        

    def update_url(self, q):
        self.url_bar.setText(q.toString())

    def save_history(self, url):
        with open(history_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([url, time.strftime("%Y-%m-%d %H:%M:%S")])

app = QApplication(sys.argv)
QApplication.setApplicationName('Safe Browser')
window = MainWindow()
app.exec_()

