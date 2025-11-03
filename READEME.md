# 🖼️ MNIST Handwritten Digit Classification using CNN

## 📋 Project Overview

**CodTech Internship - Task 2**

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras for image classification on the MNIST dataset. The model achieves **99.03% accuracy** on test data, demonstrating the effectiveness of deep learning for computer vision tasks.

**Author**: Nilay Ghosh  
**Date**: November 3, 2025

## 🎯 Project Objectives

- Build a deep learning model for handwritten digit recognition (0-9)
- Implement CNN architecture with TensorFlow/Keras
- Train and evaluate model performance
- Visualize training metrics (accuracy & loss)
- Save trained model for future inference

## 📊 Dataset Information

### MNIST Dataset
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image dimensions**: 28×28 pixels (grayscale)
- **Classes**: 10 digits (0-9)
- **Source**: TensorFlow Keras datasets

The dataset is automatically downloaded on first run from:
```
https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

## 🏗️ Model Architecture

```
Input Layer: 28×28×1 (grayscale images)
    ↓
Conv2D (32 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filters, 3×3, ReLU)
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (64 units, ReLU)
    ↓
Dense (10 units, Softmax) → Output
```

### Layer Details

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32) | (26, 26, 32) | 320 |
| MaxPooling2D | (13, 13, 32) | 0 |
| Conv2D (64) | (11, 11, 64) | 18,496 |
| MaxPooling2D | (5, 5, 64) | 0 |
| Flatten | (1600) | 0 |
| Dense (64) | (64) | 102,464 |
| Dense (10) | (10) | 650 |

**Total Parameters**: ~122,000

## 📈 Performance Metrics

### Training Results (5 Epochs)

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1 | 95.54% | 0.1445 | 98.20% | 0.0587 |
| 2 | 98.54% | 0.0475 | 98.81% | 0.0385 |
| 3 | 99.00% | 0.0332 | 98.86% | 0.0339 |
| 4 | 99.21% | 0.0246 | 99.12% | 0.0279 |
| 5 | 99.37% | 0.0197 | **99.03%** | 0.0312 |

### Final Test Performance
- **Test Accuracy**: 99.03%
- **Test Loss**: 0.0312

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- ~200MB free disk space for dataset

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd MACHINE_LEARNING___PYTHON/DATAPIPELINE_DEVELOPMENT
   ```

2. **Create virtual environment (recommended)**
   ```bash
   conda create -n ml_env python=3.9
   conda activate ml_env
   ```

3. **Install required packages**
   ```bash
   pip install tensorflow matplotlib numpy
   ```

4. **Verify installation**
   ```python
   import tensorflow as tf
   print(tf.__version__)  # Should be 2.x
   ```

## 🚀 Usage

### Running the Script

**Method 1: IDE (Spyder/PyCharm)**
```python
runfile('D:/MACHINE_LEARNING___PYTHON/DATAPIPELINE_DEVELOPMENT/untitled2.py',
        wdir='D:/MACHINE_LEARNING___PYTHON/DATAPIPELINE_DEVELOPMENT')
```

**Method 2: Command Line**
```bash
python untitled2.py
```

### Expected Output

```
📥 Loading MNIST dataset...
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 3s

🧠 Building CNN model...
🚀 Training model...

Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 16s - accuracy: 0.9554 - loss: 0.1445

...

✅ Test Accuracy: 0.9903
💾 Model saved as 'mnist_cnn_model.h5'
```

### Output Files

1. **mnist_cnn_model.h5** - Trained model (can be loaded for inference)
2. **Training visualization plot** - Accuracy and loss curves

## 📊 Visualizations

The script generates a dual-plot figure showing:

1. **Model Accuracy**
   - Training accuracy vs epochs
   - Validation accuracy vs epochs
   - Shows model learning progression

2. **Model Loss**
   - Training loss vs epochs
   - Validation loss vs epochs
   - Indicates convergence and overfitting

## 💻 Code Structure

```python
# 1️⃣ Import Libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 2️⃣ Load & Preprocess Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # Normalize to [0,1]

# 3️⃣ Build CNN Model
model = models.Sequential([...])

# 4️⃣ Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 5️⃣ Train Model
history = model.fit(x_train, y_train, epochs=5)

# 6️⃣ Evaluate & Save
test_acc = model.evaluate(x_test, y_test)
model.save("mnist_cnn_model.h5")
```

## 🔧 Customization Options

### Adjust Training Epochs
```python
history = model.fit(x_train, y_train, epochs=10)  # Train for 10 epochs
```

### Modify Model Architecture
```python
# Add more layers
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),  # Add dropout
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  # Increase neurons
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Change Optimizer
```python
model.compile(optimizer='sgd',  # Use SGD instead of Adam
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Add Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, epochs=20, 
                   validation_data=(x_test, y_test),
                   callbacks=[early_stop])
```

## 🔍 Loading & Using Saved Model

```python
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Make predictions
predictions = model.predict(x_test[:5])
predicted_labels = np.argmax(predictions, axis=1)

print(f"Predicted: {predicted_labels}")
print(f"Actual: {y_test[:5]}")
```

## 🐛 Troubleshooting

### Common Issues

**1. TensorFlow Installation Error**
```bash
# For CPU-only version
pip install tensorflow

# For GPU version (requires CUDA)
pip install tensorflow-gpu
```

**2. Memory Error During Training**
```python
# Reduce batch size
history = model.fit(x_train, y_train, batch_size=32, epochs=5)
```

**3. Model Format Warning**
```
WARNING: This file format is considered legacy
```
**Solution**: Use .keras format instead
```python
model.save("mnist_cnn_model.keras")  # New format
```

**4. oneDNN Warnings**
These are informational messages about CPU optimizations and can be safely ignored. To suppress:
```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

## 📚 Key Concepts

### Convolutional Neural Networks (CNN)
- **Convolution layers**: Extract features from images
- **Pooling layers**: Reduce spatial dimensions
- **Fully connected layers**: Perform classification

### Data Preprocessing
- **Normalization**: Scale pixel values from [0,255] to [0,1]
- **Reshaping**: Add channel dimension for grayscale images

### Training Process
- **Optimizer**: Adam (adaptive learning rate)
- **Loss function**: Sparse categorical crossentropy
- **Batch size**: 32 (default)
- **Validation split**: Using separate test set

## 🎓 Learning Outcomes

From this project, you will learn:
- ✅ Building CNN architectures with TensorFlow/Keras
- ✅ Image data preprocessing and normalization
- ✅ Training deep learning models
- ✅ Evaluating model performance
- ✅ Visualizing training metrics
- ✅ Saving and loading trained models

## 🚀 Next Steps

### Improvements
1. **Data Augmentation**: Add rotation, zoom, shift
2. **Hyperparameter Tuning**: Optimize learning rate, batch size
3. **Batch Normalization**: Add between layers
4. **Transfer Learning**: Use pre-trained models
5. **Model Deployment**: Create REST API or web app

### Advanced Projects
- Fashion-MNIST classification
- CIFAR-10/100 classification
- Custom dataset training
- Real-time digit recognition with webcam

## 📖 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras CNN Tutorial](https://keras.io/examples/vision/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Understanding CNNs](https://cs231n.github.io/)

## 📝 License

This project is for educational purposes as part of CodTech Internship.

## 👤 Author

**Nilay Ghosh**  
CodTech Internship - Task 2  
Created: November 3, 2025

## 📧 Support

For issues or questions:
- Check TensorFlow documentation
- Review troubleshooting section
- Verify Python and package versions

---

**Model Performance**: 99.03% Test Accuracy 🎯  
**Framework**: TensorFlow 2.x + Keras  
**Last Updated**: November 3, 2025