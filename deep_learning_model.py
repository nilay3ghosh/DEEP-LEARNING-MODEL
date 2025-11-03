# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 18:31:28 2025

@author: USER
"""

"""
CodTech Internship - Task 2
Deep Learning Model for Image Classification using TensorFlow (Keras)
Author: Nilay Ghosh
"""

# ------------------------------------------------------
# 1️⃣ Import Libraries
# ------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------
# 2️⃣ Load Dataset (MNIST - Handwritten Digits)
# ------------------------------------------------------
print("📥 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape for CNN: (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ------------------------------------------------------
# 3️⃣ Build CNN Model
# ------------------------------------------------------
print("🧠 Building CNN model...")
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# ------------------------------------------------------
# 4️⃣ Compile Model
# ------------------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ------------------------------------------------------
# 5️⃣ Train Model
# ------------------------------------------------------
print("🚀 Training model...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ------------------------------------------------------
# 6️⃣ Evaluate Model
# ------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ------------------------------------------------------
# 7️⃣ Visualize Training Results
# ------------------------------------------------------
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 8️⃣ Save Model
# ------------------------------------------------------
model.save("mnist_cnn_model.h5")
print("\n💾 Model saved as 'mnist_cnn_model.h5'")
