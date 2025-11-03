
# 🤖 CodTech Internship — Task 2  
## Deep Learning Model for Image Classification (TensorFlow CNN)

### 📘 Overview
This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow (Keras)** to classify handwritten digits from the **MNIST dataset**.  
It is developed as part of the **CodTech Data Science Internship – Task 2: Deep Learning Model Implementation**.

---

### 🧠 Model Architecture
The CNN automatically extracts visual features and learns to classify images into 10 digit classes (0–9).

| Layer | Type | Filters/Units | Activation |
|--------|------|---------------|-------------|
| 1 | Conv2D | 32 filters (3×3) | ReLU |
| 2 | MaxPooling2D | 2×2 | – |
| 3 | Conv2D | 64 filters (3×3) | ReLU |
| 4 | MaxPooling2D | 2×2 | – |
| 5 | Flatten | – | – |
| 6 | Dense | 64 units | ReLU |
| 7 | Dense | 10 units | Softmax |

---

### ⚙️ Training Details
- **Dataset:** MNIST (60 000 training + 10 000 testing images)  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  
- **Epochs:** 5  
- **Batch Size:** 32  

---

### 📈 Performance
| Metric | Value |
|---------|-------|
| **Training Accuracy** | 99.3 % |
| **Validation Accuracy** | 99.0 % |
| **Validation Loss** | 0.0312 |

✅ The model achieves **~99 % accuracy**, demonstrating excellent learning and generalization.

---

### 📊 Visualizations
The script automatically displays training curves:

- **Accuracy vs. Epochs**
- **Loss vs. Epochs**

Example output:

Epoch 5/5
accuracy: 0.9937 val_accuracy: 0.9903
✅ Test Accuracy: 0.9903 Run the Model
python deep_learning_model.py

3. Output

Trains the CNN for 5 epochs

Displays plots of accuracy and loss

Saves the trained model as mnist_cnn_model.h5

💾 Files in this Repository
File	Description
deep_learning_model.py	Python script with complete TensorFlow CNN implementation
requirements.txt	List of dependencies for reproducibility
README.md	Documentation (this file)
mnist_cnn_model.h5 (optional)	Saved trained model file
🧩 Extensions / Future Work

You can adapt this model for:

CIFAR-10 (colored image dataset)

Fashion-MNIST (clothing classification)

Custom datasets using your own images

👨‍💻 Author

Name: Nilay Ghosh
Internship: CodTech Data Science Internship — Task 2
Tools: Python, TensorFlow, Keras, Matplotlib
Accuracy: ≈ 99 %
