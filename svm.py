import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_dataset = pd.read_csv('/content/diabetes.csv')

# Splitting data and labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# Data standardization
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

# Replace x with standardized data
x = standardized_data

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Training the SVM model with linear kernel
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Predicting on test data
y_pred = classifier.predict(x_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score of the model:", accuracy)
