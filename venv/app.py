from flask import Flask, request, jsonify
from sklearn.datasets import  fetch_california_housing
import numpy as np
from sklearn.base import BaseEstimator
from flask_cors import CORS
app = Flask(__name__)
#CORS(app)
CORS(app, origins='http://localhost:3000')


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

if __name__ == '__main__':
    app.run()
