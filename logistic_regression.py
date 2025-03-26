import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load dataset
df = pd.read_csv("/insurance_data.csv")

# Scatter plot of raw data
plt.scatter(df['age'], df['bought_insurance'], marker='o', color='red', label='Data points')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.legend()
plt.show()

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df['bought_insurance'], test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Generate predictions over a range of ages
age_range = np.linspace(df['age'].min(), df['age'].max(), 100).reshape(-1, 1)
predicted_probs = model.predict_proba(age_range)[:, 1]  # Get probability of class 1 (bought insurance)

# Scatter plot of actual data
plt.scatter(df['age'], df['bought_insurance'], marker='o', color='red', label='Data points')

# Plot logistic regression curve
plt.plot(age_range, predicted_probs, color='blue', label='Logistic Regression Curve')

plt.xlabel("Age")
plt.ylabel("Probability of Buying Insurance")
plt.legend()
plt.show()
model.predict(x_test)
