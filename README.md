# Pneumonia-EfficientNetB0-XAI

## EfficientNetV2B0-Based Multi-Class Pneumonia Classification with Explainable AI (Grad-CAM) and Model Calibration

This repository contains the implementation of a deep learning framework for multi-class pneumonia classification using chest X-ray images. The project utilizes EfficientNetV2B0 with transfer learning to classify chest X-rays into three categories:

- NORMAL
- BACTERIA
- VIRUS

The framework also incorporates Explainable AI (Grad-CAM) and model calibration techniques to improve interpretability and reliability for clinical decision support.

---

# 📂 Project Structure

```
Pneumonia-EfficientNetB0-XAI/
│
├── dataset/
├── docs/
├── models/
├── notebooks/
│   ├── 01_Setup_Data_Verification.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   ├── 04_GradCAM.ipynb
│   └── 05_Calibration.ipynb
│
├── results/
├── utils/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 📊 Dataset

Dataset: **Processed Chest X-ray Dataset**

Classes:

- NORMAL
- BACTERIA
- VIRUS

Dataset Split:

- Training: 70%
- Validation: 15%
- Testing: 15%

---

# 🧠 Model

- EfficientNetV2B0
- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Early Stopping
- ReduceLROnPlateau
- Model Checkpoint
- Resume Training Support

---

# 📈 Evaluation Metrics

The trained model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC Score

---

# 🔬 Explainable AI

The project includes Explainable AI using:

- Grad-CAM

This helps visualize which regions of chest X-ray images contribute most to the model's predictions.

---

# 📏 Model Calibration

Calibration analysis will include:

- Expected Calibration Error (ECE)
- Reliability Diagram
- Confidence Distribution

---

# 📌 Project Status

## ✅ Completed

- Project initialization
- Repository setup
- Dataset preparation
- Three-class dataset organization
- Stratified Train/Validation/Test split
- Dataset verification notebook
- Google Colab environment setup
- Exploratory Data Analysis (EDA)

---

## 🔄 In Progress

- EfficientNetV2B0 training pipeline
- Transfer learning
- Resume training implementation
- Hyperparameter optimization

---

## ⏳ Upcoming

- Model evaluation
- Grad-CAM visualization
- Model calibration analysis
- Performance comparison
- Final research paper

---

# 🚀 Development Workflow

1. Dataset Verification
2. Model Training
3. Model Evaluation
4. Explainable AI
5. Model Calibration
6. Research Paper Preparation

---

# ⚙️ Frameworks

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV

---

# 📄 License

This project is licensed under the MIT License.

---

## 🚧 Project Status

**Under Active Development**
