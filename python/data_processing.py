import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from data_analysis import get_correlation_matrix


def process_data(loop_features, loop_targets, data_processing_params):
    print("+ Data processing")

    if data_processing_params["handle_invalid_inputs"]:
        loop_features = handle_invalid_values(loop_features)
    if data_processing_params["scale_data"]:
        loop_features = scale_data(loop_features)
    if data_processing_params["feature_selection_params"]["enabled"]:
        loop_features = feature_selection(loop_features, loop_targets,
                                          data_processing_params["feature_selection_params"])

    return loop_features, loop_targets


def handle_invalid_values(loop_features):
    print("++ Handle Invalid Inputs")

    for i in range(len(loop_features)):
        for j in range(len(loop_features[i])):
            feature = loop_features[i][j]
            if feature is None or feature < 0:
                loop_features[i][j] = 0
                print("invalid_value_changed")
    return loop_features


def scale_data(x_data):
    print("++ Scale Data")

    # x_data_scaled = preprocessing.scale(x_data)
    x_data_scaled = preprocessing.minmax_scale(x_data)
    return x_data_scaled


# casos em q variancia
def feature_selection(loop_features, loop_targets, feature_selection_params):
    print("++ Feature Selection:")

    print("Loop Features before selection: " + str(np.shape(loop_features)[1]))

    if feature_selection_params["algorithm"] == -1:
        loop_features = fs_manual(loop_features, feature_selection_params["manual_selection"])
    elif feature_selection_params["algorithm"] == 0:
        loop_features = fs_variance_threshold(loop_features, feature_selection_params["variance_threshold"])
    elif feature_selection_params["algorithm"] == 1:
        loop_features = fs_target_correlation(loop_features, loop_targets,
                                              feature_selection_params["target_correlation"], feature_selection_params)

    print("Loop Features after selection: " + str(np.shape(loop_features)[1]))
    return loop_features


def fs_manual(loop_features, manual_feature_selection):
    print("+++ Manual Feature Selection")

    feature_selection_list = list(manual_feature_selection.values())

    for i in range(len(feature_selection_list) - 1, -1, -1):
        if not feature_selection_list[i]:
            loop_features = np.delete(loop_features, i, axis=1)

    return loop_features


def fs_variance_threshold(loop_features, variance_threshold):
    print("+++ Variance Threshold Feature Selection")

    print(variance_threshold)
    sel = VarianceThreshold(threshold=variance_threshold)
    loop_features = sel.fit_transform(loop_features)

    return loop_features

#TODO remove last arg after testing
def fs_target_correlation(loop_features, loop_targets, target_correlation_params, feature_selection_params):
    print("+++ Target Correlation Feature Selection")

    corr_matrix = get_correlation_matrix(loop_features, loop_targets)
    corr_matrix_np = corr_matrix.to_numpy()

    matrix_size = np.shape(corr_matrix_np)[0]
    corr_matrix_np = np.delete(corr_matrix_np, matrix_size - 1, axis=0)
    target_correlation = np.take(corr_matrix_np, matrix_size - 1, axis=1)

    use_threshold = target_correlation_params["use_threshold"]
    mode = target_correlation_params["mode"]
    target_correlation_threshold = target_correlation_params["target_correlation_threshold"]
    top_x_correlation = target_correlation_params["top_x_correlation"]

    if mode == "abs":
        target_correlation = np.abs(target_correlation)

    selected_features_indices = []
    if use_threshold:
        for i in range(len(target_correlation)):
            if (mode == "neg" and target_correlation[i] < -target_correlation_threshold) or \
                    (mode == "pos" or mode == "abs") and target_correlation[i] > target_correlation_threshold:
                selected_features_indices.append(i)
    else:
        ordered_indices = np.argsort(target_correlation)
        ordered_target_correlation = target_correlation[ordered_indices]
        print(target_correlation)
        print(ordered_target_correlation)

        if mode == "neg":
            selected_features_indices = np.take(ordered_indices, range(top_x_correlation))
        elif mode == "pos" or mode == "abs":
            selected_features_indices = np.take(ordered_indices, range(len(ordered_indices) - 1,
                                                                       len(ordered_indices) - 1 - top_x_correlation))

    #TODO not working
    # feaure_names = (feature_selection_params["manual_selection"]).keys()
    # print(feaure_names[selected_features_indices])
    loop_features = np.take(loop_features, selected_features_indices, axis=1)

    return loop_features
