from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import numpy as np



diabetes = load_diabetes()
diabetes.feature_names, diabetes.data.shape


x_train, x_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, shuffle=True, train_size=0.8, random_state=0)

linear = LinearRegression().fit(x_train, y_train)
ridge = Ridge(alpha=0.1).fit(x_train, y_train)
lasso = Lasso(alpha=0.1).fit(x_train, y_train)

# Training Errors
print(mean_squared_error(y_train, linear.predict(x_train)), mean_squared_error(
    y_train, ridge.predict(x_train)), mean_squared_error(y_train, lasso.predict(x_train)))

# Test Errors
print(mean_squared_error(y_test, linear.predict(x_test)), mean_squared_error(
    y_test, ridge.predict(x_test)), mean_squared_error(y_test, lasso.predict(x_test)))


class SGDLinearRegressor:
    def __init__(self, batch_size=15, eta=0.01, 
                 tau_max=10000, epsilon=0.0001, 
                 random_state=None, alpha=1.0):
        self.eta = eta
        self.tau_max = tau_max
        self.epsilon = epsilon
        self.random_state = random_state
        self.batch_size = int(batch_size)
        self.alpha = alpha

    def fit(self, x, y):
        RNG = np.random.default_rng(self.random_state)
        n, p = x.shape
        self.w_ = np.zeros(p)
        self.w_list_ = [self.w_]
        for tau in range(1, int(self.tau_max) + 1):
            idx = RNG.choice(n, size=self.batch_size, replace=True)
            grad = (
                x[idx].T.dot(x[idx].dot(self.w_) - y[idx]) / self.batch_size
                + 2 * self.alpha * self.w_
            )
            self.w_ -= self.eta * grad
            self.w_list_.append(self.w_)
            if np.linalg.norm(self.w_list_[-1] - self.w_list_[-2]) < self.epsilon:
                break
        self.coef_ = self.w_
        self.w_list_ = np.array(self.w_list_)
        return self


    def predict(self, x):
        return x.dot(self.coef_)
    

sgd = SGDLinearRegressor(batch_size=15, eta=0.01,
                         tau_max=10000, epsilon=0.0001, random_state=0)
sgd.fit(x_train, y_train)

# Training Error
print(sgd.predict(x_train))
print(mean_squared_error(y_train, sgd.predict(x_train)))

# Test Error
print(sgd.predict(x_test))
print(mean_squared_error(y_test, sgd.predict(x_test)))