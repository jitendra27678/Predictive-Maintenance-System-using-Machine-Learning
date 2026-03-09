# Predictive Maintenance using Machine Learning

## Project Overview
This project predicts industrial machine failures using machine learning models trained on sensor data. The goal is to reduce unexpected downtime and maintenance costs.

## Problem Statement
Unexpected machine failures in industries can lead to production downtime and financial loss. Predictive maintenance helps detect failures before they occur.

## Dataset
The dataset contains operational features such as:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

Target Variable:
- 0 → No Failure
- 1 → Failure

## Machine Learning Workflow
1. Data preprocessing
2. Feature engineering
3. Model training
4. Hyperparameter tuning using GridSearchCV
5. Model evaluation
6. Deployment preparation

## Project Architecture

User Input → Data Preprocessing → Trained ML Model → Failure Prediction

The system takes machine sensor data as input, processes it using the trained pipeline, and predicts whether a machine failure is likely to occur.

## Models Used
- Random Forest
- Support Vector Machine (SVM)

## Model Performance

| Model | Accuracy | ROC-AUC |
|------|------|------|
| Random Forest | 98% | 0.977 |
| SVM | 98% | 0.963 |

Random Forest was selected as the final model.

## Deployment
The trained model was saved using joblib and deployed using a Streamlit application.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- Streamlit

## Author
Jitendra Khandelwal
