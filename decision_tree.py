import pandas as pd

# Load dataset
df = pd.read_csv("/content/salaries.csv")
print(df.head())

# Separate features and target variable
inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

# Encode categorical features
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])  # Fixed label encoder
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])  # Fixed label encoder

# Drop original categorical columns
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')  # Fixed column names

# Split data into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)

# Train Decision Tree model
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

# Evaluate the model
accuracy = model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
model.predict(x_test)
