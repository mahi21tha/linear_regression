import pandas as pd
import numpy as np
from sklearn import linear_model
df=pd.read_csv("/content/sample_data/homeprices.csv")
median=df.bedrooms.median()
df.bedrooms=df.bedrooms.fillna(median)
reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
a=int(input('enter area:'))
b=int(input('enter bedroom:'))
age=int(input('enter age :'))
predicted_price = reg.predict([[a, b, age]])
print("Predicted price:" ,predicted_price[0])
