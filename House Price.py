import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data.csv")

df.head()

df2 =df[(df['price'] > 0) & (df['price'] < 1000000)]
df2.head()


category = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement']
X = df2[category]
y = df2['price']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

model = lr.fit(X_train, y_train)

pickle.dump(model, open('modelprice.pkl','wb'))
