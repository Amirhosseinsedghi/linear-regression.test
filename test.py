#install libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


#import file
data= pd.read_csv('D:\education\ML\linear regression\linear-regression.test\Housing.csv')
print(type(data)) 
print(data.columns)

#cleaning data
print(data.info())
print(data.describe())
print(data.head())
data= data.dropna()


X= data [['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']] #featurs
y= data ['price'] # dependent variable

scaler= StandardScaler()
X_scaled= scaler.fit_transform(X)



#seprating data into training and testing
X_train, X_test, y_train, y_test= train_test_split(X_scaled, y,test_size= 0.3, random_state=42)


#making regression model
model=LinearRegression()
model.fit(X_train, y_train) #مدل را با داده های اموزشیم اموزش میدهیم

#model prediction
y_prediction=model.predict(X_test)

#mean squered error ارزیابی مدل با استفاده ار معیار میانگین مربعات خطا
mse=mean_squared_error(y_test, y_prediction)
print(mse)