import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU
from tensorflow.math import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns

# Set seed for reproducibility
tf.random.set_seed(3)

# Define folder paths and file names
folders_names = [r'D:/prodigy tasks/task4/train']  # Update with your folder path
files_names = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb']


# Function to create training data
def create_training_data():
    training_data = []
    for folder in folders_names:
        Class_num = folder[-1]
        print('Class', Class_num)
        for file in files_names:
            path = os.path.join(folder, file)
            print('Class', Class_num, file)
            for img in tqdm(os.listdir(path)):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                training_data.append([img_array, int(Class_num)])


create_training_data()


# Function to check image sizes
def check_image_sizes():
    first_img_shape = None
    for folder in folders_names:
        for file in files_names:
            path = os.path.join(folder, file)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if first_img_shape is None:
                    first_img_shape = img_array.shape
                elif img_array.shape != first_img_shape:
                    print("Image sizes are not consistent.")
                    return False
    print("All images have the same size:", first_img_shape)


check_image_sizes()

# Shuffle the training data
random.shuffle(training_data)

# Print class numbers for a few images
for i in range(5):
    print("Class number for image", i + 1, ":", training_data[i][1])

# Extract features and labels
X = [feature for feature, _ in training_data]
y = [label for _, label in training_data]
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the model architecture
model = Sequential([
    Flatten(input_shape=(240, 640)),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dense(32),
    LeakyReLU(alpha=0.1),
    Dense(16),
    LeakyReLU(alpha=0.1),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=3, validation_split=0.1, batch_size=32, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Training Loss: {loss:.4f}")
print(f"Training Accuracy: {accuracy * 100:.2f}%")

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Testing Loss: {loss:.4f}")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Plot the training history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred = [np.argmax(i) for i in y_pred]

# Generate classification report
print(classification_report(y_test, y_pred))

# Generate confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='bone')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()
