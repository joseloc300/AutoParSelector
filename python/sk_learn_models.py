import numpy as np
import json
from joblib import dump, load
from os import mkdir, path

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV, ParameterGrid

from tensorflow.keras.utils import normalize


# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def run_sk_regression(x_train, x_test, y_train, y_test, main_params, selected_feature_indices):
    print("+ SKLearn Regression Models")

    regression_params = main_params["create_models"]["regression"]
    model_processing_params = create_model_processing_params(main_params["data_processing"], selected_feature_indices)

    # simplest model, linear
    if regression_params["ordinary_least_squares"]:
        ordinary_least_squares(x_train, x_test, y_train, y_test, model_processing_params)

    # few features are important?
    # YES
    if regression_params["lasso"]:
        lasso(x_train, x_test, y_train, y_test, model_processing_params)
    if regression_params["elastic_net"]:
        elastic_net(x_train, x_test, y_train, y_test, model_processing_params)

    # NO
    if regression_params["svr"]:
        svr(x_train, x_test, y_train, y_test, model_processing_params)
    if regression_params["ridge_regression"]:
        ridge_regression(x_train, x_test, y_train, y_test, model_processing_params)

    # was having good results
    if regression_params["decision_tree_regressor"]:
        decision_tree_regressor(x_train, x_test, y_train, y_test, model_processing_params)

    # neural networks
    if regression_params["neural_network_sklearn"]:
        neural_network_sklearn(x_train, x_test, y_train, y_test)


def ordinary_least_squares(x_train, x_test, y_train, y_test, model_processing_params):
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    r2_score = reg.score(x_test, y_test)

    print("\nOrdinary Least Squares")
    print("R2 Score: " + str(r2_score))

    if not path.exists("./models/regression/ordinary_least_squares"):
        mkdir("./models/regression/ordinary_least_squares")
    dump(reg, "./models/regression/ordinary_least_squares/model.dump")

    model_stats = {
        "r2_score": r2_score
    }

    with open("./models/regression/ordinary_least_squares/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/ordinary_least_squares/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def lasso(x_train, x_test, y_train, y_test, model_processing_params):
    with open("tuning/regression/lasso.json") as lasso_params_file:
        file_data = json.load(lasso_params_file)

    param_grid = file_data["param_grid"]

    lasso_instance = linear_model.Lasso()

    best_score = 0
    best_params = None
    first_iter = True

    for g in ParameterGrid(param_grid):
        lasso_instance.set_params(**g)
        lasso_instance.fit(x_train, y_train)

        new_score = lasso_instance.score(x_test, y_test)

        # save if best
        if first_iter:
            best_score = new_score
            best_params = g
            first_iter = False
        elif new_score > best_score:
            best_score = new_score
            best_params = g

    all_x = np.vstack((x_train, x_test))
    all_y = np.append(y_train, y_test)
    r2_score = lasso_instance.score(all_x, all_y)

    print("\nLasso")
    print("R2 score: " + str(r2_score))
    print("Best Score:" + str(best_score))
    print("Best params:")
    print(best_params)

    if not path.exists("./models/regression/lasso"):
        mkdir("./models/regression/lasso")
    dump(lasso_instance, "./models/regression/lasso/model.dump")

    model_stats = {
        "r2_score": r2_score,
        "best_score": best_score,
        "best_params": best_params
    }

    with open("./models/regression/lasso/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/lasso/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def elastic_net(x_train, x_test, y_train, y_test, model_processing_params):
    with open("tuning/regression/elastic_net.json") as elastic_net_params_file:
        file_data = json.load(elastic_net_params_file)

    param_grid = file_data["param_grid"]

    elastic_net_instance = linear_model.ElasticNet()

    best_score = 0
    best_params = None
    first_iter = True

    for g in ParameterGrid(param_grid):
        elastic_net_instance.set_params(**g)
        elastic_net_instance.fit(x_train, y_train)

        new_score = elastic_net_instance.score(x_test, y_test)

        # save if best
        if first_iter:
            best_score = new_score
            best_params = g
            first_iter = False
        elif new_score > best_score:
            best_score = new_score
            best_params = g

    all_x = np.vstack((x_train, x_test))
    all_y = np.append(y_train, y_test)
    r2_score = elastic_net_instance.score(all_x, all_y)

    print("\nElastic Net")
    print("R2 score: " + str(r2_score))
    print("Best Score:" + str(best_score))
    print("Best params:")
    print(best_params)

    if not path.exists("./models/regression/elastic_net"):
        mkdir("./models/regression/elastic_net")
    dump(elastic_net_instance, "./models/regression/elastic_net/model.dump")

    model_stats = {
        "r2_score": r2_score,
        "best_score": best_score,
        "best_params": best_params
    }

    with open("./models/regression/elastic_net/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/elastic_net/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def svr(x_train, x_test, y_train, y_test, model_processing_params):
    with open("tuning/regression/svr.json") as svr_params_file:
        file_data = json.load(svr_params_file)

    param_grid = file_data["param_grid"]

    svr_instance = SVR()

    best_score = 0
    best_params = None
    first_iter = True

    for g in ParameterGrid(param_grid):
        svr_instance.set_params(**g)
        svr_instance.fit(x_train, y_train)

        new_score = svr_instance.score(x_test, y_test)

        # save if best
        if first_iter:
            best_score = new_score
            best_params = g
            first_iter = False
        elif new_score > best_score:
            best_score = new_score
            best_params = g

    all_x = np.vstack((x_train, x_test))
    all_y = np.append(y_train, y_test)
    r2_score = svr_instance.score(all_x, all_y)

    print("\nSVR")
    print("R2 score: " + str(r2_score))
    print("Best Score:" + str(best_score))
    print("Best params:")
    print(best_params)

    if not path.exists("./models/regression/svr"):
        mkdir("./models/regression/svr")
    dump(svr_instance, "./models/regression/svr/model.dump")

    model_stats = {
        "r2_score": r2_score,
        "best_score": best_score,
        "best_params": best_params
    }

    with open("./models/regression/svr/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/svr/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def ridge_regression(x_train, x_test, y_train, y_test, model_processing_params):
    with open("tuning/regression/ridge_regression.json") as ridge_regression_params_file:
        file_data = json.load(ridge_regression_params_file)

    param_grid = file_data["param_grid"]

    ridge_regression_instance = linear_model.Ridge()

    best_score = 0
    best_params = None
    first_iter = True

    for g in ParameterGrid(param_grid):
        ridge_regression_instance.set_params(**g)
        ridge_regression_instance.fit(x_train, y_train)

        new_score = ridge_regression_instance.score(x_test, y_test)

        # save if best
        if first_iter:
            best_score = new_score
            best_params = g
            first_iter = False
        elif new_score > best_score:
            best_score = new_score
            best_params = g

    all_x = np.vstack((x_train, x_test))
    all_y = np.append(y_train, y_test)
    r2_score = ridge_regression_instance.score(all_x, all_y)

    print("\nRidge Regression")
    print("R2 score: " + str(r2_score))
    print("Best Score:" + str(best_score))
    print("Best params:")
    print(best_params)

    if not path.exists("./models/regression/ridge_regression"):
        mkdir("./models/regression/ridge_regression")
    dump(ridge_regression_instance, "./models/regression/ridge_regression/model.dump")

    model_stats = {
        "r2_score": r2_score,
        "best_score": best_score,
        "best_params": best_params
    }

    with open("./models/regression/ridge_regression/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/ridge_regression/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def decision_tree_regressor(x_train, x_test, y_train, y_test, model_processing_params):
    with open("tuning/regression/decision_tree_regressor.json") as decision_tree_regression_params_file:
        file_data = json.load(decision_tree_regression_params_file)

    param_grid = file_data["param_grid"]

    decision_tree_regression_instance = DecisionTreeRegressor()

    best_score = 0
    best_params = None
    first_iter = True

    for g in ParameterGrid(param_grid):
        decision_tree_regression_instance.set_params(**g)
        decision_tree_regression_instance.fit(x_train, y_train)

        new_score = decision_tree_regression_instance.score(x_test, y_test)

        # save if best
        if first_iter:
            best_score = new_score
            best_params = g
            first_iter = False
        elif new_score > best_score:
            best_score = new_score
            best_params = g

    all_x = np.vstack((x_train, x_test))
    all_y = np.append(y_train, y_test)
    r2_score = decision_tree_regression_instance.score(all_x, all_y)

    print("\nDecision Tree Regressor")
    print("R2 score: " + str(r2_score))
    print("Best Score:" + str(best_score))
    print("Best params:")
    print(best_params)

    if not path.exists("./models/regression/decision_tree_regressor"):
        mkdir("./models/regression/decision_tree_regressor")
    dump(decision_tree_regression_instance, "./models/regression/decision_tree_regressor/model.dump")

    model_stats = {
        "r2_score": r2_score,
        "best_score": best_score,
        "best_params": best_params
    }

    with open("./models/regression/decision_tree_regressor/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/regression/decision_tree_regressor/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


# TODO not up to date. lacking parameterGrid support and model, stats, and params output to files.
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


# https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
def run_sk_classification(loop_features, loop_targets, main_params, selected_feature_indices):
    print("+ SKLearn Classification Models")

    classification_params = main_params["create_models"]["classification"]
    model_processing_params = create_model_processing_params(main_params["data_processing"], selected_feature_indices)

    if classification_params["svc"]:
        svc(loop_features, loop_targets, model_processing_params)
    if classification_params["kn_classifier"]:
        kn_classifier(loop_features, loop_targets, model_processing_params)


def svc(loop_features, loop_targets, model_processing_params):
    print("\nSVC classifier")

    with open("tuning/classification/svc.json") as svc_params_file:
        file_data = json.load(svc_params_file)

    param_grid = file_data["param_grid"]

    model = Pipeline([
        ('sampling', SMOTE()),
        ('classification', svm.SVC())
    ])

    gscv = GridSearchCV(model, param_grid, scoring="roc_auc")
    gscv.fit(loop_features, loop_targets)

    if not path.exists("./models/classification/svc"):
        mkdir("./models/classification/svc")
    dump(gscv, "./models/classification/svc/model.dump")

    score_gscv = gscv.score(loop_features, loop_targets)
    print("ROC AUC: " + str(score_gscv))
    print("Best score: " + str(gscv.best_score_))
    print("Best params:")
    print(gscv.best_params_)

    y_test_pred = gscv.predict(loop_features)

    print("Classification report:")
    print(classification_report(loop_targets, y_test_pred))

    model_stats = {
        "roc_auc": score_gscv,
        "best_score": gscv.best_score_,
        "best_params": gscv.best_params_,
        "classification_report": classification_report(loop_targets, y_test_pred, output_dict=True)
    }

    with open("./models/classification/svc/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/classification/svc/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def kn_classifier(loop_features, loop_targets, model_processing_params):
    print("\nkN classifier")

    with open("tuning/classification/kn_classifier.json") as kn_classifier_params_file:
        file_data = json.load(kn_classifier_params_file)

    param_grid = file_data["param_grid"]

    model = Pipeline([
        ('sampling', SMOTE()),
        ('classification', KNeighborsClassifier())
    ])

    gscv = GridSearchCV(model, param_grid, scoring="roc_auc")
    gscv.fit(loop_features, loop_targets)

    if not path.exists("./models/classification/kn_classifier"):
        mkdir("./models/classification/kn_classifier")
    dump(gscv, "./models/classification/kn_classifier/model.dump")

    score_gscv = gscv.score(loop_features, loop_targets)
    print("ROC AUC: " + str(score_gscv))
    print("Best score: " + str(gscv.best_score_))
    print("Best params:")
    print(gscv.best_params_)

    y_test_pred = gscv.predict(loop_features)

    print("Classification report:")
    print(classification_report(loop_targets, y_test_pred))

    model_stats = {
        "roc_auc": score_gscv,
        "best_score": gscv.best_score_,
        "best_params": gscv.best_params_,
        "classification_report": classification_report(loop_targets, y_test_pred, output_dict=True)
    }

    with open("./models/classification/kn_classifier/model_stats.json", "w") as outfile:
        json.dump(model_stats, outfile, indent=4)

    with open("./models/classification/kn_classifier/model_processing_params.json", "w") as outfile:
        json.dump(model_processing_params, outfile, indent=4)


def make_predictions(loop_features, loop_targets, loop_targets_orig, prediction_params, loop_info):
    print("\n+ Make predictions")
    model = load(prediction_params["model_folder_path"] + "/model.dump")
    y_pred = model.predict(loop_features)
    np_loop_info = np.array(loop_info)
    model_predictions = {
        "modelName": prediction_params["model_name"],
        "modelClassification": prediction_params["is_classification"],
        "targetIds": (np_loop_info[:, 0]).tolist(),
        "targetProblemSizeFlags": (np_loop_info[:, 2]).tolist(),
        "targetOmpPragmas": (np_loop_info[:, 1]).tolist(),
        "targetPredictions": y_pred.tolist(),
        "targetValuesPostProcess": loop_targets,
        "targetValuesOrig": loop_targets_orig,
        "classification_stats": [],
        "r2_score": []
    }

    if len(loop_targets) > 0:
        if prediction_params["is_classification"]:
            model_predictions["classification_stats"] = {
                "roc_auc": model.score(loop_features, loop_targets),
                "classification_report": classification_report(loop_targets, y_pred, output_dict=True)
            }
        else:
            model_predictions["r2_score"] = model.score(loop_features, loop_targets)

    with open("./predictions/" + prediction_params["model_name"] + ".json", "w") as outfile:
        json.dump(model_predictions, outfile, indent=4)


def create_model_processing_params(processing_params, selected_features_indices):
    model_processing_params = {
        "threshold_target": processing_params["threshold_target"],
        "handle_invalid_inputs": processing_params["handle_invalid_inputs"],
        "scale_data_algorithm": processing_params["scale_data_algorithm"],
        "feature_selection_params": processing_params["feature_selection_params"],
        "selected_features_indices": selected_features_indices
    }

    return model_processing_params
