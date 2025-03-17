import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv("/canada_per_capita_income.csv")
%matplotlib inline
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.scatter(df['year'],df['per capita income (US$)'])
reg=linear_model.LinearRegression()
reg.fit(df[['year']],df[['per capita income (US$)']])
plt.plot(df['year'],df['per capita income (US$)'],color='blue',)
y=int(input("enter year:"))
print(reg.predict(y))
