# 🧠 CNN for Image Classification - Assignment 2

### 👤 Student: Oscar Quintero

### 🆔 Student ID: c0922321

---

## 📌 Project Description

This project focuses on building a **Convolutional Neural Network (CNN)** to classify natural scene images using the **Intel Image Classification** dataset.

The model is trained using TensorFlow and Keras, and it includes:

- Image data preprocessing with augmentation
- A custom CNN architecture with Batch Normalization
- Training/validation split
- Model evaluation and performance visualization
- Feature map interpretation and misclassification analysis

---

## 📁 Dataset

- Source: [Intel Image Classification on Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification)
- Classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- Downloaded using KaggleHub API

---

## ⚙️ Model Architecture

- Three Conv2D + MaxPooling blocks with Batch Normalization
- Dropout for regularization
- Flatten + Dense layers for classification
- Final layer with Softmax for 6-class output

---

## 📊 Results & Visualizations

The notebook includes:

- ✅ Accuracy/Loss plots during training
- ✅ Confusion matrix with seaborn
- ✅ Precision, Recall, F1-score using `classification_report`
- ✅ Misclassified images with actual vs predicted labels
- ✅ Feature maps of the first convolutional layer to interpret learned features

---

## 🛠 Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 🚀 Run Instructions

1. Install requirements:
   ```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn
   ```
