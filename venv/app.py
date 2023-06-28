from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.datasets import  fetch_california_housing
import numpy as np
from sklearn.base import BaseEstimator
from flask_cors import CORS
import yfinance as yf
from decimal import Decimal


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


# Enable CORS for the stock prediction endpoint
CORS(app, resources={
     r"/stockprediction": {"origins": "http://localhost:3000"}})


@app.route('/stockprediction', methods=['POST'])
def predict_stock():
    user_input = request.json

    # Extract the stock symbol from the user input
    symbol = user_input['symbol']

    # Get stock information
    stock = yf.Ticker(symbol)

    # Get current date
    from datetime import datetime, timedelta


    # Get the current date
    current_date = datetime.now().date()

    # Check if the current date falls on a weekend (Saturday or Sunday)
    if current_date.weekday() == 5:  # Saturday
        # Subtract one day to get to Friday
        current_date -= timedelta(days=1)
    elif current_date.weekday() == 6:  # Sunday
        # Subtract two days to get to Friday
        current_date -= timedelta(days=2)

    # Format the current date as "yyyy-mm-dd"
    most_recent_trading_date = current_date.strftime("%Y-%m-%d")
    
    # Generate first trading days for some stocks:

    '''
    Netflix (NFLX): 2002-05-23

    NVIDIA (NVDA): 1999-01-22

    Google (GOOGL/GOOG): 2004-08-19

    Microsoft (MSFT): 1986-03-13

    Amazon (AMZN): 1997-05-15

    Facebook (FB): 2012-05-18

    Apple (AAPL): 1980-12-12

    Blackstone (BX): 2007-06-22

    BlackRock (BLK): 1988-10-01

    JPMorgan Chase (JPM): 1972-06-01

    Goldman Sachs (GS): 1999-05-04

    Costco (COST): 1985-07-09

    '''
    stock_ticker = 'X'  # Replace 'X' with the desired stock ticker

    if symbol == 'NFLX':
        start = '2002-05-23'
    elif symbol == 'NVDA':
        start = '1999-01-22'
    elif symbol == 'GOOGL' or symbol == 'GOOG':
        start = '2004-08-19'
    elif symbol == 'MSFT':
        start = '1986-03-13'
    elif symbol == 'AMZN':
        start = '1997-05-15'
    elif symbol == 'FB':
        start = '2012-05-18'
    elif symbol == 'AAPL':
        start = '1980-12-12'
    elif symbol == 'BX':
        start = '2007-06-22'
    elif symbol == 'BLK':
        start = '1988-10-01'
    elif symbol == 'JPM':
        start = '1972-06-01'
    elif symbol == 'GS':
        start = '1999-05-04'
    elif symbol == 'COST':
        start = '1985-07-09'
    else:
        start = '1980-12-12'

    # Load historical stock data
    stock_data = stock.history(start=start, end=most_recent_trading_date,
                               interval="1d")

    stock_data['Close(t-1)'] = stock_data['Close'].shift(1)
    stock_data = stock_data.dropna()

    # Split the adjusted data into features (X) and target variable (y)
    X = stock_data.drop('Close', axis=1)  # Features
    y = stock_data['Close']  # Target variable

    # Create an instance of the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the data
    scaler.fit(X)

    # Transform the data using the scaler
    X_scaled = scaler.transform(X)

    # Split the adjusted data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=1)

    # Train the model
    neurons_range = range(1, 10, 1)
    test_errors = []

    for n in neurons_range:
        model = MLPRegressor(hidden_layer_sizes=(
            n,n,n), max_iter=2000, tol=1e-4, activation='relu', solver='adam', alpha=0.001)
        model.fit(X_train, y_train)
        error = 1 - model.score(X_test, y_test)
        test_errors.append(error)
        print(f'Number of neurons: {n}, Test error: {error:.6f}')

    # Neural network Model Selection
    test_errors = np.array(test_errors)
    indices = list(range(len(test_errors)))
    list_neurons = [neurons_range[i] for i in indices]
    list_errors = [test_errors[i] for i in indices]
    min_error = min(list_errors)
    min_index = list_errors.index(min_error)
    best_neuron_count = list_neurons[min_index]

    # Create an instance of the MLPRegressor with the best_neuron_count
    model = MLPRegressor(hidden_layer_sizes=(best_neuron_count, 
                                             best_neuron_count,
                                             best_neuron_count), max_iter=2000,
                         tol=1e-4, activation='relu', solver='adam', alpha=0.001)
    model.fit(X_scaled, y)

    # Make predictions
    predicted_price = model.predict(X_test)[-1]
    formatted_price = Decimal(predicted_price).quantize(Decimal('0.0000'))
    most_recent_price = stock.history(period="1d")["Close"].iloc[-1]
    formatted_recent_price = Decimal(most_recent_price).quantize(Decimal('0.0000'))

    # Stock XGBoost
    import xgboost as xgb
    from xgboost import DMatrix
    ###################
    #boost_model = xgb.XGBRegressor(objective = 'reg:squarederror')
    #boost_model.fit(X_train, y_train)
    #xg_predicted_price = boost_model.predict(X_test)[-1]
    #xg_formatted_price = float(xg_predicted_price)
    #XG_formatted_price = Decimal(
    #    xg_formatted_price).quantize(Decimal('0.0000'))
    ########################
    # Define the hyperparameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    # Create an instance of the XGBoost regressor
    boost_model = xgb.XGBRegressor(objective='reg:squarederror')

    from sklearn.model_selection import GridSearchCV

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=boost_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Use the best model for prediction
    best_model = grid_search.best_estimator_
    predicted_price = best_model.predict(X_test)[-1]
    xg_formatted_price = float(predicted_price)
    XG_formatted_price = Decimal(
        xg_formatted_price).quantize(Decimal('0.0000'))





    # Create a dictionary to hold the predicted price
    result = {
        'predicted_price': str(formatted_price),
        'xgboost_predicted_price': str(XG_formatted_price),
        'most_recent_price': str(formatted_recent_price)
    }

    # Return the predicted price as JSON
    return jsonify(result)


if __name__ == '__main__':
    app.run()
