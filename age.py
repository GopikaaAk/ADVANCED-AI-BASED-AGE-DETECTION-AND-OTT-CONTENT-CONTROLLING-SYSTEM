import cv2
import time
import csv
import sys
import tensorflow as tf  # Import TensorFlow for CNN model
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# CSV file paths
block_18_csv = "block_18.csv"
block_20_csv = "block_20.csv"
history_csv = "history.csv"

# Function to load sites from CSV
def load_sites_from_csv(csv_file):
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        sites = [row[0] for row in csv_reader]
    return sites

# MainWindow class for browser interface
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://google.com'))
        self.setCentralWidget(self.browser)
        self.showMaximized()

        # Navbar setup
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
            url = "https://" + url
            self.save_history(url)
        else:
            self.save_history(url)
            url = "https://www.google.co.in/error"
        self.browser.setUrl(QUrl(url))

    def update_url(self, q):
        self.url_bar.setText(q.toString())

    def save_history(self, url):
        with open(history_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([url, time.strftime("%Y-%m-%d %H:%M:%S")])

# Function to get the face bounding box
def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes

# Placeholder for loading a CNN model
def load_cnn_model():
    model = tf.keras.models.load_model("CNN.model")
    model = None  # Since you don't want it to work, just set to None
    return model

def predict_with_cnn(model, face_image):
    return 0  

# Load the CNN model (even though we don't use it)
cnn_model = load_cnn_model()

# Model paths and parameters
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the models
ageNet = cv2.dnn.readNet(ageModel, ageProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize camera
cap = cv2.VideoCapture(0)
padding = 20

while True:
    # Read frame
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # Create a smaller frame for better optimization
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Get face bounding box
    frameFace, bboxes = getFaceBox(faceNet, small_frame)
    if not bboxes:
        print("No face detected, checking next frame")
        continue

    for bbox in bboxes:
        # Extract face
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                           max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        # Preprocess face for age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        
        age = ageList[agePreds[0].argmax()]
        label = "{}".format(age)
        cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)

    # Display the image with bounding boxes
    cv2.imshow("Age Detection", frameFace)

    # Wait for 'q' key to capture the age and block sites
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        ai = agePreds[0].argmax()
        print("Captured Age Index:", ai)

        if ai <= 3:
            sites_to_block = load_sites_from_csv(block_18_csv)
        else:
            sites_to_block = load_sites_from_csv(block_20_csv)

        break

    # Exit loop if 'ESC' key is pressed
    if key & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

# Launch the browser application
app = QApplication(sys.argv)
QApplication.setApplicationName('Safe Browser')
window = MainWindow()
app.exec_()

