import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def run_regression_baseline(X, y, degree=25):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X.T)
    model = LinearRegression()
    model.fit(X_poly, y.T)
    y_pred = model.predict(X_poly).T
    loss = mean_squared_error(y.flatten(), y_pred.flatten())
    return y_pred, loss
