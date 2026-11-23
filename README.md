# Railway Defect Detection Using Deep Learning

## 1. Project Overview

Railway Defect Detection is a Deep Learning-based image classification project developed to identify railway track defects from uploaded images.

The project uses Convolutional Neural Networks (CNN) and MobileNetV2 Transfer Learning to classify railway images into two categories:

- Defective
- Non defective

A FastAPI backend connects the trained model with a web-based frontend, allowing users to upload railway images and receive prediction results.

## 2. Problem Statement

Railway track defects can affect railway safety and operational reliability. Manual inspection requires time and human effort.

This project explores an image-based Deep Learning approach to classify railway track images and assist the inspection process.

The system is a prototype and is not a replacement for professional railway inspection.

## 3. Project Objectives

- Analyze and understand the railway image dataset.
- Perform image preprocessing.
- Develop a CNN classification model.
- Apply MobileNetV2 Transfer Learning.
- Evaluate model performance.
- Build a prediction pipeline using Python.
- Develop a REST API using FastAPI.
- Create a web interface for image upload and prediction.

## 4. Dataset Information

The project uses a railway defect image dataset.

Dataset directory:

```text
dataset/
├── raw/
│   └── RailwayDefectDetectionDatabase V2/
│
└── processed/
    ├── Defective/
    └── Non defective/
```

Dataset distribution:

| Category | Number of Images |
|---|---:|
| Defective | 478 |
| Non defective | 471 |
| Total | 949 |

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical operations |
| Pandas | Data analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| TensorFlow / Keras | Deep Learning |
| CNN | Image classification |
| MobileNetV2 | Transfer Learning |
| Scikit-learn | Model evaluation |
| Pillow | Image processing |
| FastAPI | Backend API |
| Uvicorn | API server |
| HTML | Frontend |
| CSS | Web interface styling |
| JavaScript | Frontend API integration |
| VS Code | Development environment |

## 6. Project Workflow

1. Dataset Analysis
2. Image Preprocessing
3. CNN Model Development
4. Transfer Learning Using MobileNetV2
5. Model Evaluation
6. Model Saving and Loading
7. Image Prediction
8. FastAPI Development
9. Frontend Development
10. API and Frontend Integration

## 7. Project Folder Structure

```text
RAILWAY_DEFECT_DETECTION/
│
├── Artifacts/
│   └── Saved model files and configurations
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── Notebooks/
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Image_Preprocessing.ipynb
│   ├── 03_CNN_Model.ipynb
│   ├── 04_Transfer_Learning.ipynb
│   └── 05_Model_Evaluation.ipynb
│
├── Reports/
│   └── Evaluation and prediction reports
│
├── sample_images/
│
├── Src/
│   ├── config.py
│   ├── data_loader.py
│   ├── dataset_audit.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── api/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation and Environment Setup

Clone or download the project and open its root directory in VS Code.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## 9. Running the Backend API

Start the FastAPI application from the project root directory:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | API home |
| GET | /health | Model health check |
| POST | /predict | Railway image prediction |
| GET | /api-info | API information |

## 10. Running the Frontend

Open the following file in Visual Studio Code:

```text
frontend/index.html
```

Right-click and select:

**Open with Live Server**

The frontend allows users to:

- Select a railway image.
- Preview the uploaded image.
- Send the image to the FastAPI backend.
- View the predicted category.
- View the confidence percentage and class probabilities.

Keep the FastAPI server running while using the frontend.

## 11. Model Prediction

The system classifies an uploaded railway image into one of two categories:

- Defective
- Non defective

The prediction response includes:

- Image name
- Predicted class
- Confidence percentage
- Class probabilities
- Prediction timestamp

## 12. Model Evaluation

The project includes model evaluation using classification metrics and visualization techniques.

Evaluation methods include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

Actual evaluation values should be taken from the saved evaluation results.

## 13. Expected Application Output

The web application displays the uploaded railway image and the corresponding prediction.

The system provides a prediction category and confidence percentage for the selected image.

## 14. Future Enhancements

- Improve classification performance using additional railway images.
- Perform independent train, validation and test evaluation.
- Explore object detection for identifying defect locations.
- Develop a more advanced inspection dashboard.
- Deploy the application on a cloud platform.
- Integrate with real-time railway inspection systems.

## 15. Conclusion

This project demonstrates the application of Deep Learning and Transfer Learning for railway image classification.

By integrating CNN, MobileNetV2, Python, FastAPI and a web frontend, the project provides an end-to-end prototype for classifying railway track images.

The project also demonstrates the practical implementation of a Deep Learning model through an API and a user-friendly web interface.