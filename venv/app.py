from flask import Flask, request, jsonify
from sklearn.datasets import load_diabetes, fetch_california_housing
import numpy as np
from sklearn.base import BaseEstimator
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

# Define the route for getting the feature names
@app.route('/feature_names', methods=['GET'])
def get_feature_names():
    feature_names = ["Feature 1", "Feature 2"]  # Replace with actual feature names
    return jsonify(feature_names)

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
diabetes = load_diabetes()
X_diab, y_diab = diabetes.data, diabetes.target
knn_reg_diab = KnnRegressor(k=10)
knn_reg_diab.fit(X_diab, y_diab)

california_housing = fetch_california_housing()
X_cali, y_cali = california_housing.data, california_housing.target
knn_reg_cali = KnnRegressor(k=10)
knn_reg_cali.fit(X_cali, y_cali)


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json

    # Perform the prediction using the trained models and user input
    user_input_diab = user_input.get('diabetes', None)
    user_input_cali = user_input.get('california', None)

    prediction_diab = None
    prediction_cali = None

    if user_input_diab is not None:
        user_data_diab_arr = np.array(list(user_input_diab.values()))
        prediction_diab = knn_reg_diab.predict([user_data_diab_arr])

    if user_input_cali is not None:
        user_data_cali_arr = np.array(list(user_input_cali.values()))
        prediction_cali = knn_reg_cali.predict([user_data_cali_arr])

    # Return the prediction results as JSON
    results = {
        'diabetes_prediction': prediction_diab.tolist(),
        'california_prediction': prediction_cali.tolist()
    }
    return jsonify(results)


if __name__ == '__main__':
    app.run()
