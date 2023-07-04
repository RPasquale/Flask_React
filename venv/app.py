from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
from sklearn.base import BaseEstimator
import yfinance as yf
from decimal import Decimal
from flask_cors import CORS
from flask import Flask, request, jsonify, redirect, url_for
from db import connect_to_database
from user import User
from flask_dance.contrib.google import make_google_blueprint, google
import secrets
from bson import ObjectId


app = Flask(__name__)
mongo = connect_to_database()

app.secret_key = secrets.token_hex(16)


# Enable CORS for the house login endpoint
CORS(app, resources={r"/login": {"origins": "http://localhost:3000"}})

# Enable CORS for the house register endpoint
CORS(app, resources={r"/register": {"origins": "http://localhost:3000"}})

# Enable CORS for the house Google login endpoint
CORS(app, resources={r"/login/google": {"origins": "http://localhost:3000"}})

google_bp = make_google_blueprint(
    client_id="840094957080-u0rcair0evjmv6uhbm5qcilv6ure9cp0.apps.googleusercontent.com",
    client_secret="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",
    scope=["profile", "email"],
    redirect_url="http://localhost:5000/login/google/callback"
)

app.register_blueprint(google_bp, url_prefix="/login")

@app.route("/register", methods=["POST"])
def register():
    # Extract the registration details from the request
    full_name = request.json.get("full_name")
    email = request.json.get("email")
    password = request.json.get("password")

    # Check if the email is already registered
    if mongo.db.users.find_one({"email": email}):
        return jsonify({"message": "Email already registered"}), 409

    # Create a new user document
    user = User(full_name=full_name, email=email)
    user.set_password(password)

    # Insert the user document into the "users" collection
    mongo.db.users.insert_one(user.__dict__)

    return jsonify({"message": "User registered successfully"}), 201


import jwt


@app.route("/login", methods=["POST"])
def login():
    # Extract the login details from the request
    secret_login__key = secrets.token_hex(16)
    email = request.json.get("email")
    password = request.json.get("password")

    # Retrieve the user from the database based on the provided email
    user_dict = mongo.db.users.find_one({"email": email})
    print("Retrieved user from the database:", user_dict)  # Add this line for debugging


    # Check if the user exists and the password is correct
    if user_dict:
        # Create a User object from the retrieved dictionary
        user = User(**user_dict)

        # Verify the password check
        is_password_match = user.check_password(password)
        print("Is password match:", is_password_match)  # Add this line for verification

        if is_password_match:
            # Generate the authentication token
            payload = {"user_id": str(user._id)}
            secret_key = secret_login__key  # Replace with your own secret key
            algorithm = "HS256"
            token = jwt.encode(payload, secret_key, algorithm=algorithm)

            print("Generated token:", token) 
            print("User's full name:", user.full_name)
            # Return the token to the client
            return jsonify({"token": token, "fullName": user.full_name}), 200

    # If the user does not exist or the password is incorrect, return an error response
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/login/google")
def login_google():
    if not google.authorized:
        return redirect(url_for("google.login"))

    # Handle the authorized state
    # You can access the user's information using `google.get("/oauth2/v2/userinfo")`
    google_user_info = google.get("/oauth2/v2/userinfo").json()
    email = google_user_info["email"]

    # Check if the email is already registered
    user_dict = mongo.db.users.find_one({"email": email})
    if user_dict:
        return jsonify({"message": "Email already registered"}), 409

    # Create a new user document
    user = User(full_name=google_user_info["name"], email=email)

    # Insert the user document into the "users" collection
    mongo.db.users.insert_one(user.__dict__)

    return jsonify({"message": "User registered successfully"}), 201


# CORS(app)
# CORS(app, origins='http://localhost:3000')
# Enable CORS for the house prediction endpoint
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

# Enable CORS for the diabetes prediction endpoint
CORS(app, resources={
     r"/predict/diabetes": {"origins": "http://localhost:3000"}})


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
    HP_result = {'prediction': prediction,
                 'Median Income of Households in a block': float(user_input['feature1']),
                 'Median Age of a House within a block': float(user_input['feature2']),
                 'Average Number of Rooms in a Block': float(user_input['feature3']),
                 'Average Number of Bedrooms in a Block': float(user_input['feature4']),
                 'Total Number of people residing within a block': float(user_input['feature5']),
                 'Average Occupancy': float(user_input['feature6']),
                 'Latitude': float(user_input['feature7']),
                 'Longitude': float(user_input['feature8'])}

    try:
        # Save the result in the MongoDB collection
        mongo.db.HP_results.insert_one(HP_result)
        print("Result saved in the database")
    except Exception as e:
        print("Error saving result in the database:", str(e))

    HP_result['_id'] = str(HP_result['_id'])  # Convert ObjectId to string

    # Return the prediction result as JSON
    result = {'prediction': prediction}
    return jsonify(result)




from flask_login import login_required
import jwt




def load_user(user_id):
    print("Inside load_user function", flush=True)
    print("User ID:", user_id, flush=True)
    # Retrieve the user from the database based on the user_id
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    print("Retrieved user:", user, flush=True)
    
    # If the user exists, create a User object with the necessary attributes
    if user:
        user_object = User(user['_id'], user['full_name'], user['email'], user['password_hash'])
        print("User object:", user_object, flush=True)

        return user_object

    return None


# Enable CORS for the diabetes prediction endpoint
CORS(app, resources={r"/save-prediction": {"origins": "http://localhost:3000"}})

@app.route('/save-prediction', methods=['POST'])
def save_prediction():
    # Extract the prediction data from the request
    prediction_data = request.json

    # Retrieve the user's profile from the database
    if 'user_id' in prediction_data:
        # Add the prediction data to the user's profile
        user_id = prediction_data['user_id']
        user = mongo.db.users.find_one({'UserID': user_id})
        user['prediction_data'].append(prediction_data)

        # Save the updated profile data in the database
        mongo.db.users.update_one({'UserID': prediction_data['user_id']}, {'$set': user})

        return jsonify(message='Prediction saved successfully')
    else:
        return jsonify(message='User not found'), 404



shared_house_predictions = []  # Define the variable as a global list

# Enable CORS for the stock prediction endpoint
CORS(app, resources={
    r"/shared/house": {"origins": "http://localhost:3000"}
})


@app.route("/shared/house", methods=["POST"])
def share_house_prediction():
    data = request.json
    prediction = data.get('prediction')
    if prediction is not None:
        try:
            prediction = float(prediction)  # Convert prediction to a number
        except (ValueError, TypeError):
            return jsonify(error='Invalid prediction value'), 400
    else:
        return jsonify(error='Missing or invalid prediction value'), 400

    median_income = data.get('Median Income of Households in a block')
    median_age = data.get('Median Age of a House within a block')
    num_rooms = data.get('Average Number of Rooms in a Block')
    num_bedrooms = data.get('Average Number of Bedrooms in a Block')
    num_residents = data.get('Total Number of people residing within a block')
    avg_occupancy = data.get('Average Occupancy')
    latitude = data.get('Latitude')
    longitude = data.get('Longitude')

    # Create a new shared prediction object
    shared_house_prediction = {
        '_id': len(shared_house_predictions) + 1,
        'prediction': prediction,
        'Median Income of Households in a block': median_income,
        'Median Age of a House within a block': median_age,
        'Average Number of Rooms in a Block': num_rooms,
        'Average Number of Bedrooms in a Block': num_bedrooms,
        'Total Number of people residing within a block': num_residents,
        'Average Occupancy': avg_occupancy,
        'Latitude': latitude,
        'Longitude': longitude
    }

    # Add the shared prediction to the list
    shared_house_predictions.append(shared_house_prediction)

    return jsonify(shared_house_prediction)

@app.route('/shared/house', methods=['GET'])
def get_shared_house_predictions():
    return jsonify(shared_house_predictions)



################################

# Diabetes Prediction Model


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
        'Lasso Regression': lasso.predict(input_data)[0],
        'Age': float(user_input['feature1']),
        'Sex': float(user_input['feature2']),
        'Body Mass Index (BMI)': float(user_input['feature3']),
        'Average Blood Pressure': float(user_input['feature4']),
        'Total Serum Cholesterol': float(user_input['feature5']),
        'Low Density Lipoproteins': float(user_input['feature6']),
        'High Density Lipoproteins': float(user_input['feature7']),
        'Total Cholesterol / HDL': float(user_input['feature8']),
        'Log of Serum Triglicerydes Level': float(user_input['feature9']),
        'Blood Sugar Level': float(user_input['feature10']),
    }

    try:
        # Save the result in the MongoDB collection
        mongo.db.LR_results.insert_one(model_predictions)
        print("Result saved in the database")
    except Exception as e:
        print("Error saving result in the database:", str(e))

    # Convert ObjectId to string
    model_predictions['_id'] = str(model_predictions['_id'])

    LR_predictions = {
        'Linear Regression': linear.predict(input_data)[0],
        'Ridge Regression': ridge.predict(input_data)[0],
        'Lasso Regression': lasso.predict(input_data)[0]
    }

    LR_result = {'prediction': LR_predictions}

    return jsonify(LR_result)


shared_diab_predictions = []  # Define the variable as a global list

# Enable CORS for the stock prediction endpoint
CORS(app, resources={
    r"/shared/diabetes": {"origins": "http://localhost:3000"}
})


@app.route("/shared/diabetes", methods=["POST"])
def share_diabetes_prediction():
    data = request.json
    LinearRegression = data.get('Linear Regression')
    Ridge_Regression = data.get('Ridge Regression')
    Lasso_Regression = data.get('Lasso Regression')
    Age = data.get('Age')
    Sex = data.get('Sex')
    BMI = data.get('Body Mass Index (BMI)')
    Average_Blood_Pressure = data.get('Average Blood Pressure')
    Total_Serum_Cholesterol = data.get('Total Serum Cholesterol')
    Low_Density_Lipoproteins = data.get('Low Density Lipoproteins')
    High_Density_Lipoproteins = data.get('High Density Lipoproteins')
    Total_Cholesterol_HDL = data.get('Total Cholesterol / HDL')
    Log_of_Serum_Triglicerydes_Level = data.get(
        'Log of Serum Triglicerydes Level')
    Blood_Sugar_Level = data.get('Blood Sugar Level')

    # Create a new shared prediction object
    shared_diabetes_prediction = {
        '_id': len(shared_diab_predictions) + 1,
        'Linear Regression': LinearRegression,
        'Ridge Regression': Ridge_Regression,
        'Lasso Regression': Lasso_Regression,
        'Age': Age,
        'Sex': Sex,
        'Body Mass Index (BMI)': BMI,
        'Average Blood Pressure': Average_Blood_Pressure,
        'Total Serum Cholesterol': Total_Serum_Cholesterol,
        'Low Density Lipoproteins': Low_Density_Lipoproteins,
        'High Density Lipoproteins': High_Density_Lipoproteins,
        'Total Cholesterol / HDL': Total_Cholesterol_HDL,
        'Log of Serum Triglicerydes Level': Log_of_Serum_Triglicerydes_Level,
        'Blood Sugar Level': Blood_Sugar_Level,
    }

    # Add the shared prediction to the list
    shared_diab_predictions.append(shared_diabetes_prediction)

    return jsonify(shared_diabetes_prediction)


@app.route('/shared/diabetes', methods=['GET'])
def get_shared_diab_predictions():
    return jsonify(shared_diab_predictions)


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
    y = np.log(stock_data['Close'])  # Log-transformed target variable

    # Create an instance of the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the data
    scaler.fit(X)

    # Transform the data using the scaler
    X_scaled = scaler.transform(X)

    def mean_squared_error(y, y_hat, length):
        return sum(((y-y_hat)**2))/length

    # Split the adjusted data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=1)

    # Train the model
    neurons_range = range(1, 10, 1)
    test_errors = []

    for n in neurons_range:
        model = MLPRegressor(hidden_layer_sizes=(
            n, n, n), max_iter=2000, tol=1e-4, activation='relu', solver='adam', alpha=0.001)
        model.fit(X_train, y_train)
        # model.predict(X_test)
        # error = 1 - model.score(X_test, y_test)

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

    NN_MSE = mean_squared_error(y_test, model.predict(X_test), len(X_test))

    # Make predictions
    predicted_price = model.predict(X_test)[-1]
    formatted_price = Decimal(predicted_price).quantize(Decimal('0.0000'))
    most_recent_price = np.log(stock.history(period="1d")["Close"].iloc[-1])
    formatted_recent_price = Decimal(
        most_recent_price).quantize(Decimal('0.0000'))

    # Stock XGBoost
    import xgboost as xgb
    from xgboost import DMatrix

    ########################
    # Define the hyperparameter grid
    param_grid = {
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [3, 5, 7, 10, 13],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    # Create an instance of the XGBoost regressor
    boost_model = xgb.XGBRegressor(objective='reg:squarederror')

    from sklearn.model_selection import GridSearchCV

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=boost_model, param_grid=param_grid, cv=10)
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
        'most_recent_price': str(formatted_recent_price),

    }

    try:
        # Save the result in the MongoDB collection
        mongo.db.results.insert_one(result)
        print("Result saved in the database")
    except Exception as e:
        print("Error saving result in the database:", str(e))

    result['_id'] = str(result['_id'])  # Convert ObjectId to string

    # Return the predicted price as JSON
    return jsonify(result)


shared_predictions = []
CORS(app)
# Enable CORS for the stock prediction endpoint
CORS(app, resources={
     r"/share": {"origins": "http://localhost:3000"}})


@app.route('/share', methods=['POST'])
def share_prediction():
    data = request.get_json()
    symbol = data.get('symbol')
    predicted_price = data.get('predicted_price')
    xgboost_predicted_price = data.get('xgboost_predicted_price')
    most_recent_price = data.get('most_recent_price')

    # Create a new shared prediction object
    shared_prediction = {
        '_id': len(shared_predictions) + 1,
        'symbol': symbol,
        'predicted_price': predicted_price,
        'xgboost_predicted_price': xgboost_predicted_price,
        'most_recent_price': most_recent_price
    }

    # Add the shared prediction to the list
    shared_predictions.append(shared_prediction)

    return jsonify(shared_prediction)


@app.route('/shared', methods=['GET'])
def get_shared_predictions():
    return jsonify(shared_predictions)


if __name__ == '__main__':
    app.run()
