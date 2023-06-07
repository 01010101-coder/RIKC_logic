import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np


df = pd.DataFrame(data = {"first": [38, 21,35,34,61,72,80,29,64,27,22,58,28,44,46,73,75,75,50, 51, 57, 22, 34, 27, 24, 30],
                   "max" : [80,50,50,50,80,80,80,80,80,60,60,80,50,50,50,80,80,80,60, 80, 80, 50, 50, 60, 60, 60],
                   'final' : [50,67,79,73,73,84,100,40,76,61,58,70,67,89,96,85,89,89,90, 63, 83, 60, 78, 61, 58, 55]})

X = df.drop(['final'], axis = 1)
Y = df['final']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=12345)

data = [X_train, X_test, y_train, y_test]

rf = RandomForestRegressor()
linear = LinearRegression()


def train_fit(model, data):
    model.fit(data[0], data[2])
    print('y test', list(y_test))
    print('y predicted', model.predict(data[1]).astype(int))

print('Linear Regression')
train_fit(linear, data)

print()

print('Random Forest')
train_fit(rf, data)
np.array([36,40]).reshape(-1,1)
print(linear.predict(np.array([46,50]).reshape(1,-1)))
print(rf.predict(np.array([35,40]).reshape(1,-1)))





