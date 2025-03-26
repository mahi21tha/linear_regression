import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
df = load_iris()

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(x_train, y_train)

# Evaluate model performance
accuracy = model.score(x_test, y_test)
print("Model Accuracy:", accuracy)

# Make predictions
predictions = model.predict(x_test)

# Loop through predictions and print meaningful output
for i in predictions:
    if i == 0:
        print("Predicted class: Setosa")
    elif i == 1:
        print("Predicted class: Versicolor")
    else:
        print("Predicted class: Virginica")
