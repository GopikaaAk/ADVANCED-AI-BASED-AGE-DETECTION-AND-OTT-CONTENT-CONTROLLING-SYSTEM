import os
import cv2
import numpy as np
import random
import pickle

# Directory containing your hand sign images (subfolders for each class)
DATADIR = 'dataset'
CATEGORIES = os.listdir(DATADIR)  # Get all subfolder names as categories
IMG_SIZE = 150

training_data = []

def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATADIR, category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert to RGB
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X*3).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Set 3 channels for color images
y = np.array(y*3)

# Save the processed data to files
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

