import json

from data_loading import load_data
from data_loading import load_main_params
from data_analysis import analyze_data
from data_processing import process_data, smote_oversampling_regression, prediction_data_processing
from sk_learn_models import run_sk_regression
from sk_learn_models import run_sk_classification
from sk_learn_models import make_predictions
# from tf_models import run_tf_regression
# from tf_models import run_tf_classification
import utils
import numpy as np
# from sklearn.model_selection import train_test_split


def main():
    main_params = load_main_params()
    loop_features, loop_targets, loop_info, feature_names = load_data(main_params["read_data_prefixes"], False)

    # pre-processing data analysis
    analyze_data(loop_features.copy(), loop_targets.copy(), feature_names.copy(), main_params["data_analysis"],
                 True, None)

    if main_params["create_models"]["regression"]["enabled"]:
        create_regression_models(loop_features.copy(), loop_targets.copy(), feature_names.copy(), main_params)
    if main_params["create_models"]["classification"]["enabled"]:
        create_classification_models(loop_features.copy(), loop_targets.copy(), feature_names.copy(), main_params)

    if main_params["make_prediction"]["enabled"]:
        loop_features, loop_targets, loop_info, feature_names = load_data(main_params["read_data_prefixes"], True)
        make_prediction_from_model(loop_features.copy(), loop_targets.copy(), main_params["make_prediction"],
                                   loop_info.copy())

    exit(0)


def create_regression_models(loop_features, loop_targets, feature_names, main_params):
    loop_features, loop_targets, new_feature_names, new_feature_indices = process_data(loop_features, loop_targets,
                                                                                       feature_names,
                                                                                       main_params["data_processing"],
                                                                                       False)

    if np.shape(np.array(loop_features))[1] == 0:
        print("Canceling create_regression_models due to lack of features after data processing")
        return

    analyze_data(loop_features, loop_targets, new_feature_names, main_params["data_analysis"], False, False)

    x_train, x_test, y_train, y_test = utils.stratified_regressions_sampling(loop_features, loop_targets,
                                                                             main_params["sampling_params"])

    if main_params["data_processing"]["regression_smote"]:
        x_train, y_train = smote_oversampling_regression(x_train, y_train, new_feature_names)

    run_sk_regression(x_train, x_test, y_train, y_test, main_params, new_feature_indices)
    # run_tf_regression(x_train, x_test, y_train, y_test)


def create_classification_models(loop_features, loop_targets, feature_names, main_params):
    loop_features, loop_targets, new_feature_names, new_feature_indices = process_data(loop_features, loop_targets,
                                                                                       feature_names,
                                                                                       main_params["data_processing"],
                                                                                       True)

    if np.shape(np.array(loop_features))[1] == 0:
        print("Canceling create_classification_models due to lack of features after data processing")
        return

    analyze_data(loop_features, loop_targets, new_feature_names, main_params["data_analysis"], False, True)

    # test_ratio = 1 - main_params["sampling_params"]["train_ratio"]
    # x_train, x_test, y_train, y_test = train_test_split(loop_features, loop_targets, test_size=test_ratio)

    run_sk_classification(loop_features, loop_targets, main_params, new_feature_indices)
    # run_tf_classification(x_train, x_test, y_train, y_test)


def make_prediction_from_model(loop_features, loop_targets, prediction_params, loop_info):
    with open(prediction_params["model_folder_path"] + "/model_processing_params.json") as f:
        prediction_processing_params = json.load(f)
    loop_targets_orig = loop_targets.copy()
    loop_features, loop_targets = prediction_data_processing(loop_features, loop_targets,
                                                             prediction_params["is_classification"],
                                                             prediction_processing_params)

    make_predictions(loop_features, loop_targets, loop_targets_orig, prediction_params, loop_info)


main()
