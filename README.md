# 🧠 Brain Tumor Detection using CNN

An AI-based brain MRI image classification project using **Convolutional Neural Networks (CNNs)** and **TensorFlow/Keras** to classify MRI images into four categories.

## 🚀 Project Overview

This project uses deep learning to classify brain MRI images into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

**Model:** Custom CNN  
**Framework:** TensorFlow / Keras  
**Input Image Size:** 224 × 224 × 3  
**Number of Classes:** 4

## ✨ Features

- Brain MRI image classification into four categories.
- Image preprocessing and normalization.
- CNN model training and evaluation.
- Training and validation performance visualization.
- Grad-CAM visualization for model interpretability.
- Interactive Streamlit web application.
- Prediction confidence and class-wise probabilities.

## 📊 Model Performance

| Metric | Result |
|---|---:|
| Training Accuracy | 98.12% |
| Validation Accuracy | 91.61% |
| Test Accuracy | 87.19% |
| Test Loss | 0.8680 |

The model was trained for 10 epochs. The difference between training and test performance indicates some overfitting, so further improvements are possible.

## 🏗️ CNN Architecture

The model uses a Sequential CNN architecture consisting of:

1. Input layer (224 × 224 × 3)
2. Conv2D + MaxPooling2D (32 filters)
3. Conv2D + MaxPooling2D (64 filters)
4. Conv2D + MaxPooling2D (128 filters)
5. Flatten layer
6. Dense layer (128 neurons)
7. Dropout (0.5)
8. Softmax output layer (4 classes)

**Total Parameters:** 11,169,476

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow
- Streamlit
- Google Colab
- VS Code

## 🌐 Streamlit Web Application

An interactive web interface allows users to upload brain MRI images and receive model predictions.

### Application Features

- Upload JPG, JPEG, and PNG images.
- Predict one of four brain MRI categories.
- Display predicted class and confidence score.
- Show class-wise prediction probabilities.

### Run the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

**Important:** The trained model file `brain_tumor_model.h5` must be placed inside the `notebooks/` directory. It is excluded from this repository because of its large size.

## 📁 Project Structure

```text
brain-tumor-detection-cnn/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── notebooks/
│   ├── brain_tumor_detection.ipynb
│   ├── brain_tumor_model.h5
│   ├── training_performance.png
│   └── all_datasets/
│
└── screenshots/
```

Note: The trained model and dataset are kept locally and are not included in the GitHub repository.

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/sanchij624-oss/brain-tumor-detection-cnn.git
cd brain-tumor-detection-cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Required Files

- Place the MRI dataset in `notebooks/all_datasets/`.
- Place the trained model at `notebooks/brain_tumor_model.h5`.

### 4. Run the Application

```bash
streamlit run app.py
```

## 🔮 Future Improvements

- Improve model generalization and reduce overfitting.
- Improve classification performance for individual tumor classes.
- Enhance Grad-CAM visualizations.
- Experiment with transfer learning architectures.
- Evaluate on additional independent MRI datasets.

## ⚠️ Disclaimer

This project is developed

🙏 Acknowledgements

Thanks to the contributors of publicly available brain MRI datasets used for educational and research purposes.
