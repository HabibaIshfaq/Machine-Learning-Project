Diabetes Prediction Project
This project aims to predict diabetes using various machine learning models. It involves data preprocessing, exploratory data analysis, and applying multiple classifiers to make predictions.

Introduction
Diabetes is a chronic disease characterized by high levels of sugar in the blood. It can lead to serious health complications, including heart disease, stroke, kidney failure, and blindness. Early detection and management of diabetes are crucial to prevent these complications.

Dataset
The dataset used in this project is the "Diabetes Prediction Dataset," which contains several health-related features for individuals. The goal is to use these features to predict whether an individual has diabetes or not.

Features
age: Age of the individual.
gender: Gender of the individual.
bmi: Body Mass Index, a measure of body fat based on height and weight.
HbA1c_level: Average blood sugar levels over the past three months.
blood_glucose_level: Blood glucose level at the time of measurement.
smoking_history: Smoking history of the individual.
diabetes: Target variable indicating whether the individual has diabetes (1) or not (0). By leveraging machine learning models, we aim to build a robust predictive model that can assist healthcare providers in identifying individuals at risk of diabetes.
Table of Contents
Installation
Usage
Project Structure
Data Preprocessing
Modeling
Evaluation
Dependencies
Installation
Clone the repository:
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
Usage
To run the project, execute the following command: python app.py

Project Structure
app.py: Main script to run the project. diabetes_prediction_dataset.csv: Dataset used for training and testing. README.md: Project documentation. requirements.txt: List of dependencies.

Data Preprocessing
Import Necessary Libraries
import numpy as np import pandas as pd import matplotlib.pyplot as plt import seaborn as sns from sklearn.model_selection import train_test_split

Load Dataset
df = pd.read_csv("diabetes_prediction_dataset.csv") X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

Data Overview
df.info() df.describe() df.isnull().sum()

Detect and Remove Outliers
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

def detect_outliers(df, cols): outliers = {} for col in cols: Q1 = df[col].quantile(0.25) Q3 = df[col].quantile(0.75) IQR = Q3 - Q1 lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)] return outliers

outliers = detect_outliers(df, numerical_cols)

def remove_outliers(df, column): Q1 = df[column].quantile(0.25) Q3 = df[column].quantile(0.75) IQR = Q3 - Q1 lower_bound = Q1 - 1.5 * IQR upper_bound = Q3 + 1.5 * IQR filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)] return filtered_df

for col in numerical_cols: df = remove_outliers(df, col)

Feature Engineering and Normalization
from sklearn.preprocessing import StandardScaler, LabelEncoder

data = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

scaler = StandardScaler() data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.fit_transform(data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']])

label_encoder = LabelEncoder() data['gender'] = label_encoder.fit_transform(data['gender'])

Modeling
Random Forest Classifier:
from sklearn.ensemble import RandomForestClassifier rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) rf_classifier.fit(X_train, y_train) y_pred_rf = rf_classifier.predict(X_test)

Decision Tree Classifier:
from sklearn.tree import DecisionTreeClassifier dt_classifier = DecisionTreeClassifier(random_state=42) dt_classifier.fit(X_train, y_train) y_pred_dt = dt_classifier.predict(X_test)

Naive Bayes Classifier:
from sklearn.naive_bayes import GaussianNB nb_classifier = GaussianNB() nb_classifier.fit(X_train, y_train) y_pred_nb = nb_classifier.predict(X_test)

Logistic Regression:
from sklearn.linear_model import LogisticRegression lr_classifier = LogisticRegression(random_state=42, max_iter=1000) lr_classifier.fit(X_train, y_train) y_pred_lr = lr_classifier.predict(X_test)

K-Nearest Neighbors:
from sklearn.neighbors import KNeighborsClassifier knn_classifier = KNeighborsClassifier(n_neighbors=5) knn_classifier.fit(X_train, y_train) y_pred_knn = knn_classifier.predict(X_test)

Support Vector Machine:
from sklearn.svm import SVC svc_classifier = SVC(random_state=42) svc_classifier.fit(X_train, y_train) y_pred_svc = svc_classifier.predict(X_test)

Evaluation
Evaluate each model using accuracy score, classification report, and confusion matrix: from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf) print("\nRandom Forest Classifier Accuracy:", accuracy_rf) print("\nClassification Report:") print(classification_report(y_test, y_pred_rf)) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_rf))

Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt) print("\nDecision Tree Classifier Accuracy:", accuracy_dt) print("\nClassification Report:") print(classification_report(y_test, y_pred_dt)) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_dt))

Naive Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb) print("\nNaive Bayes Classifier Accuracy:", accuracy_nb) print("\nClassification Report:") print(classification_report(y_test, y_pred_nb)) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_nb))

Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr) print("\nLogistic Regression Classifier Accuracy:", accuracy_lr) print("\nClassification Report:") print(classification_report(y_test, y_pred_lr)) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_lr))

KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn) print("\nKNN Classifier Accuracy:", accuracy_knn) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_knn)) print("\nClassification Report:") print(classification_report(y_test, y_pred_knn))

SVM
accuracy_svc = accuracy_score(y_test, y_pred_svc) print("\nSVM Classifier Accuracy:", accuracy_svc) print("\nConfusion Matrix:") print(confusion_matrix(y_test, y_pred_svc)) print("\nClassification Report:") print(classification_report(y_test, y_pred_svc))

Dependencies
The project uses the following libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
scipy These can be installed via requirements.txt
