from flask import Flask, request, jsonify
from sklearn.datasets import  fetch_california_housing
import numpy as np
from sklearn.base import BaseEstimator
from flask_cors import CORS
app = Flask(__name__)
#CORS(app)
#CORS(app, origins='http://localhost:3000')

# Enable CORS for the house prediction endpoint
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Enable CORS for the diabetes prediction endpoint
CORS(app, resources={r"/predict/diabetes": {"origins": "http://localhost:3000"}})


class KnnRegressor(BaseEstimator):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.sqrt(np.sum((x - self.X_train) ** 2, axis=1))
            sorted_indices = np.argsort(distances)
            knn_indices = sorted_indices[:self.k]
            knn_labels = self.y_train[knn_indices]
            y_pred.append(np.mean(knn_labels))
        return np.array(y_pred)


# Load the trained models and other necessary variables
california_housing = fetch_california_housing()
X_cali, y_cali = california_housing.data, california_housing.target
knn_reg_cali = KnnRegressor(k=10)
knn_reg_cali.fit(X_cali, y_cali)


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    
    # Extract the features from the user input
    feature_values = [
        float(user_input['feature1']),
        float(user_input['feature2']),
        float(user_input['feature3']),
        float(user_input['feature4']),
        float(user_input['feature5']),
        float(user_input['feature6']),
        float(user_input['feature7']),
        float(user_input['feature8']),

    ]
    
    # Convert the feature values to a numpy array
    input_data = np.array(feature_values).reshape(1, -1)
    
    # Make the prediction using the trained model
    prediction = knn_reg_cali.predict(input_data)[0]

    # Return the prediction result as JSON
    result = {'prediction': prediction}
    return jsonify(result)

# Diabetes Prediction Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, shuffle=True, train_size=0.8, random_state=0)

linear = LinearRegression().fit(x_train, y_train)
ridge = Ridge(alpha=0.1).fit(x_train, y_train)
lasso = Lasso(alpha=0.1).fit(x_train, y_train)


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    user_input = request.json

    feature_values = [
        float(user_input['feature1']),
        float(user_input['feature2']),
        float(user_input['feature3']),
        float(user_input['feature4']),
        float(user_input['feature5']),
        float(user_input['feature6']),
        float(user_input['feature7']),
        float(user_input['feature8']),
        float(user_input['feature9']),
        float(user_input['feature10']),
    ]

    input_data = np.array(feature_values).reshape(1, -1)

    model_predictions = {
        'Linear Regression': linear.predict(input_data)[0],
        'Ridge Regression': ridge.predict(input_data)[0],
        'Lasso Regression': lasso.predict(input_data)[0]
    }

    result = {'prediction': model_predictions}
    return jsonify(result)


if __name__ == '__main__':
    app.run()
