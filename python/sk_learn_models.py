import numpy as np
import json

from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras.utils import normalize


def run_sk_models(x_train, x_test, y_train, y_test):
    print("+ SKLearn Models")

    main_models(x_train, x_test, y_train, y_test)


# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def main_models(x_train, x_test, y_train, y_test):
    # few features are important?
    # YES
    lasso(x_train, x_test, y_train, y_test)
    lasso_cv(x_train, x_test, y_train, y_test)

    elastic_net(x_train, x_test, y_train, y_test)

    # NO
    ridge_regression(x_train, x_test, y_train, y_test)
    svr(x_train, x_test, y_train, y_test)

    # last case scenario
    ensemble_regressors(x_train, x_test, y_train, y_test)

    # was having good results
    decision_tree_regressor(x_train, x_test, y_train, y_test)

    # neural networks
    neural_network_sklearn(x_train, x_test, y_train, y_test)


def svr(x_train, x_test, y_train, y_test):
    reg = SVR(kernel="linear")
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("SVR linear")
    print(score)

    reg_2 = SVR(kernel="rbf")
    reg_2.fit(x_train, y_train)

    score_2 = reg_2.score(x_test, y_test)

    print("SVR rbf")
    print(score_2)


def ensemble_regressors(x_train, x_test, y_train, y_test):
    pass


# 1
def supervised_learning(x_train, x_test, y_train, y_test):
    linear_models(x_train, x_test, y_train, y_test)
    decision_tree_regressor(x_train, x_test, y_train, y_test)
    # isotonic_regression()


# 1.1
def linear_models(x_train, x_test, y_train, y_test):
    ordinary_least_squares(x_train, x_test, y_train, y_test)
    ridge_regression(x_train, x_test, y_train, y_test)
    lasso(x_train, x_test, y_train, y_test)
    elastic_net(x_train, x_test, y_train, y_test)
    lars(x_train, x_test, y_train, y_test)


# 1.1.1
def ordinary_least_squares(x_train, x_test, y_train, y_test):
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Ordinary Least Squares")
    print(score)


# 1.1.2
def ridge_regression(x_train, x_test, y_train, y_test):
    reg = linear_model.Ridge(max_iter=10000, tol=0.0001)
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)

    print("Ridge Regression")
    print(score)

    reg_2 = linear_model.RidgeCV(alphas=np.arange(0.01, 1.0, 0.01), cv=5)
    reg_2.fit(x_train, y_train)
    score_2 = reg_2.score(x_test, y_test)

    print("Ridge Regression CV")
    print(score_2)


# 1.1.3
def lasso(x_train, x_test, y_train, y_test):
    with open("./tuning/lasso.json") as lasso_params_file:
        file_data = json.load(lasso_params_file)

    param_grid = file_data["param_grid"]

    lasso_instance = linear_model.Lasso()

    gscv = GridSearchCV(lasso_instance, param_grid)
    gscv.fit(x_train, y_train)

    score_gscv = gscv.score(x_test, y_test)
    print(score_gscv)


    # reg = linear_model.Lasso(alpha=1.0, max_iter=1000000, tol=0.5)
    # reg.fit(x_train, y_train)
    #
    # score = reg.score(x_test, y_test)
    #
    # print("Lasso")
    # print(score)
    #
    # reg_2 = linear_model.LassoCV(tol=0.75, cv=5)
    # reg_2.fit(x_train, y_train)
    #
    # score_2 = reg_2.score(x_test, y_test)
    #
    # print("Lasso CV")
    # print(score_2)


def lasso_cv(x_train, x_test, y_train, y_test):
    with open("./tuning/lasso_cv.json") as lasso_cv_params_file:
        file_data = json.load(lasso_cv_params_file)

    param_grid = file_data["param_grid"]

    lasso_cv_instance = linear_model.LassoCV()

    rscv = RandomizedSearchCV(lasso_cv_instance, param_grid, n_iter=20)
    rscv.fit(x_train, y_train)

    score_gscv = rscv.score(x_test, y_test)
    print(score_gscv)

# 1.1.5
def elastic_net(x_train, x_test, y_train, y_test):
    reg = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5, tol=0.5)
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Elastic Net")
    print(score)

    reg_2 = linear_model.ElasticNetCV(l1_ratio=0.5, tol=0.5, cv=5)
    reg_2.fit(x_train, y_train)

    score_2 = reg_2.score(x_test, y_test)

    print("Elastic Net CV")
    print(score_2)


# 1.1.7
def lars(x_train, x_test, y_train, y_test):
    reg = linear_model.Lars()
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Lars")
    print(score)

    reg_2 = linear_model.LarsCV(max_iter=500, cv=None)
    reg_2.fit(x_train, y_train)

    score_2 = reg_2.score(x_test, y_test)

    print("Lars CV")
    print(score_2)


# 1.10.2
def decision_tree_regressor(x_train, x_test, y_train, y_test):
    reg_1 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10)
    reg_2 = DecisionTreeRegressor(max_depth=10, min_samples_leaf=4)
    reg_3 = DecisionTreeRegressor()

    reg_1.fit(x_train, y_train)
    reg_2.fit(x_train, y_train)
    reg_3.fit(x_train, y_train)

    score_1 = reg_1.score(x_test, y_test)
    score_2 = reg_2.score(x_test, y_test)
    score_3 = reg_3.score(x_test, y_test)

    print("Decision Tree Regressor")
    print(score_1)
    print(score_2)
    print(score_3)


# 1.15
def isotonic_regression(x_train, x_test, y_train, y_test):
    reg = IsotonicRegression()
    # reg.fit(x_train, y_train)
    x_new = reg.fit_transform(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Isotonic Regression")
    print(score)


# 2
def unsupervised_learning(x_train, x_test, y_train, y_test):
    pass


def neural_network_sklearn(x_train, x_test, y_train, y_test):
    tf_x_train = np.asarray(x_train[:])
    tf_y_train = np.asarray(y_train[:])
    tf_x_test = np.asarray(x_test[:])
    tf_y_test = np.asarray(y_test[:])

    tf_x_train = normalize(tf_x_train, axis=1)
    tf_x_test = normalize(tf_x_test, axis=1)

    regr = MLPRegressor(random_state=1, max_iter=50000, learning_rate='adaptive').fit(tf_x_train, tf_y_train)
    regr.predict(tf_x_test[:2])

    score = regr.score(tf_x_test, tf_y_test)
    print("Neural Network Sklearn")
    print(score)
