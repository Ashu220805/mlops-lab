import pandas as pd
from sklearn.linear_model import LinearRegression

# Simple dataset
X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[5]])

print("Prediction for 5:", prediction[0])