import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

data = pd.read_csv('/content/test_labels.csv')

X = data[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
y = data['filename']
print(data.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_dataset(path):
    images = []
    labels = []
    for i in range(1, 101):
        for j in range(1, 101):
            img = cv2.imread(f"{path}/{i}_{j}.png", cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(i-1)
    return np.array(images), np.array(labels)

def extract_features(images):
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        features.append(des)
    return np.array(features)

def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def plot_histogram(data):
    plt.figure(figsize=(10, 7))
    sns.histplot(data, bins=30, kde=True)
    plt.title('Histogram of Data')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_histogram(data)
