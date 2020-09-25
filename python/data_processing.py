import json

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from data_analysis import get_correlation_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import smogn


def process_data(loop_features, loop_targets, feature_names, data_processing_params, classification):
    if classification:
        print("+ Data processing classification")
        loop_targets = convert_target_to_classes(loop_targets)
    else:
        print("+ Data processing regression")
        if data_processing_params["threshold_target"]["enabled"]:
            loop_targets = threshold_target(loop_targets, data_processing_params["threshold_target"]["threshold"])

    if data_processing_params["handle_invalid_inputs"]:
        loop_features = handle_invalid_values(loop_features)

    if data_processing_params["scale_data_algorithm"] != -1:
        loop_features = scale_data(loop_features, data_processing_params["scale_data_algorithm"])

    new_feature_names = feature_names.copy()
    new_feature_indices = list(range(0, len(new_feature_names)))

    if data_processing_params["feature_selection_params"]["enabled"]:
        loop_features, new_feature_names, new_feature_indices = feature_selection(loop_features, loop_targets,
                                                                                  feature_names,
                                                                                  data_processing_params[
                                                                                      "feature_selection_params"])

    return loop_features, loop_targets, new_feature_names, new_feature_indices


def prediction_data_processing(loop_features, loop_targets, is_classification, prediction_processing_params):
    print("+ Data processing")
    if len(loop_targets) > 0 and is_classification:
        loop_targets = convert_target_to_classes(loop_targets)
    elif len(loop_targets) > 0:
        if prediction_processing_params["threshold_target"]["enabled"]:
            loop_targets = threshold_target(loop_targets, prediction_processing_params["threshold_target"]["threshold"])

    if prediction_processing_params["handle_invalid_inputs"]:
        loop_features = handle_invalid_values(loop_features)

    if prediction_processing_params["scale_data_algorithm"] != -1:
        loop_features = scale_data(loop_features, prediction_processing_params["scale_data_algorithm"])

    if prediction_processing_params["feature_selection_params"]["enabled"]:
        loop_features = fs_prediction(loop_features, prediction_processing_params["selected_features_indices"])

    return loop_features, loop_targets


def handle_invalid_values(loop_features):
    print("++ Handle Invalid Inputs")

    invalid_counter = 0

    for i in range(len(loop_features)):
        for j in range(len(loop_features[i])):
            feature = loop_features[i][j]
            if feature is None or feature < 0:
                loop_features[i][j] = 0
                invalid_counter += 1

    print("Invalid values found/changed: " + str(invalid_counter))

    return loop_features


def scale_data(x_data, algorithm):
    print("++ Scale Data")

    x_data_scaled = x_data

    if algorithm == 0:
        x_data_scaled = preprocessing.scale(x_data)
    elif algorithm == 1:
        scaler = StandardScaler()
        scaler.fit(x_data)
        x_data_scaled = scaler.transform(x_data)
    elif algorithm == 2:
        x_data_scaled = preprocessing.minmax_scale(x_data)
    elif algorithm == 3:
        rbX = RobustScaler()
        x_data_scaled = rbX.fit_transform(x_data)

    return x_data_scaled


def threshold_target(y_data, threshold):
    for i in range(len(y_data)):
        if y_data[i] < threshold:
            y_data[i] = 0
    return y_data


def convert_target_to_classes(y_data):
    for i in range(len(y_data)):
        if y_data[i] <= 1:
            y_data[i] = 0
        else:
            y_data[i] = 1
    return y_data


def feature_selection(loop_features, loop_targets, feature_names, feature_selection_params):
    print("++ Feature Selection:")

    print("Loop Features before selection: " + str(np.shape(loop_features)[1]))

    new_feature_names = feature_names.copy()
    new_feature_indices = list(range(0, len(new_feature_names)))

    if feature_selection_params["algorithm"] == 0:
        loop_features, new_feature_names, new_feature_indices = fs_variance_threshold(loop_features, feature_names,
                                                                                      feature_selection_params[
                                                                                          "variance_threshold"])
    elif feature_selection_params["algorithm"] == 1:
        loop_features, new_feature_names, new_feature_indices = fs_target_correlation(loop_features, loop_targets,
                                                                                      feature_names,
                                                                                      feature_selection_params[
                                                                                          "target_correlation"])

    print("Loop Features after selection: " + str(np.shape(loop_features)[1]))
    return loop_features, new_feature_names, new_feature_indices


def fs_variance_threshold(loop_features, feature_names, variance_threshold):
    print("+++ Variance Threshold Feature Selection")

    print("Variance Threshold: " + str(variance_threshold))
    sel = VarianceThreshold(threshold=variance_threshold)
    loop_features = sel.fit_transform(loop_features)

    indices = sel.get_support(indices=True)
    new_feature_names = list(np.array(feature_names)[indices])
    new_feature_indices = indices.tolist()

    print("Selected features:")
    print(new_feature_names)

    to_file = {
        "variance_threshold": variance_threshold,
        "selected_features": new_feature_names,
        "selected_features_indices": new_feature_indices
    }

    with open("./results/fs_variance_threshold.json", "w") as outfile:
        json.dump(to_file, outfile, indent=4)

    return loop_features, new_feature_names, new_feature_indices


def fs_target_correlation(loop_features, loop_targets, feature_names, target_correlation_params):
    print("+++ Target Correlation Feature Selection")

    corr_matrix = get_correlation_matrix(loop_features, loop_targets, feature_names)
    corr_matrix_np = corr_matrix.to_numpy()

    matrix_size = np.shape(corr_matrix_np)[0]
    corr_matrix_np = np.delete(corr_matrix_np, matrix_size - 1, axis=0)
    target_correlation = np.take(corr_matrix_np, matrix_size - 1, axis=1)

    use_threshold = target_correlation_params["use_threshold"]
    mode = target_correlation_params["mode"]
    target_correlation_threshold = target_correlation_params["target_correlation_threshold"]
    top_x_correlation = target_correlation_params["top_x_correlation"]

    original_target_correlation = target_correlation
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
        if mode == "neg":
            selected_features_indices = np.take(ordered_indices, list(range(top_x_correlation)))
        elif mode == "pos" or mode == "abs":
            selected_features_indices = np.take(ordered_indices,
                                                list(range(np.shape(ordered_indices)[0] - 1,
                                                           np.shape(ordered_indices)[0] - 1 - top_x_correlation, -1)))

        selected_features_indices = selected_features_indices.tolist()

    new_feature_names = np.array(feature_names)[selected_features_indices]
    new_feature_names = new_feature_names.tolist()
    new_feature_indices = selected_features_indices

    print(new_feature_names)
    print(original_target_correlation[selected_features_indices])

    loop_features = np.take(loop_features, selected_features_indices, axis=1)

    print(target_correlation[selected_features_indices].tolist())

    to_file = {
        "use_threshold": use_threshold,
        "mode": mode,
        "target_correlation_threshold": target_correlation_threshold,
        "top_x_correlation": top_x_correlation,
        "selected_features": new_feature_names,
        "selected_features_indices": selected_features_indices,
        "selected_features_correlation": target_correlation[selected_features_indices].tolist()
    }

    with open("./results/fs_target_correlation.json", "w") as outfile:
        json.dump(to_file, outfile, indent=4)

    return loop_features, new_feature_names, new_feature_indices


def fs_prediction(loop_features, selected_features_indices):
    loop_features = np.take(loop_features, selected_features_indices, axis=1)
    return loop_features


def smote_oversampling_regression(loop_features, loop_targets, new_feature_names):
    print("+++ SMOTE Oversampling Regression")

    loop_features_np = np.array(loop_features)
    loop_targets_np = np.array(loop_targets)
    loop_targets_np = np.reshape(loop_targets_np, (np.shape(loop_targets_np)[0], 1))
    combined_data = np.append(loop_features_np, loop_targets_np, axis=1)
    combined_data_list = combined_data.tolist()
    dataframe = pd.DataFrame(combined_data_list)

    column_names = new_feature_names.copy()
    column_names.append("target")
    dataframe.columns = column_names

    smogn.smoter(data=dataframe, y="target")

    upsampled_data = np.array(dataframe.values.tolist())
    new_loop_targets = upsampled_data[:, -1].tolist()
    new_loop_features = np.delete(upsampled_data, -1, axis=1).tolist()

    return new_loop_features, new_loop_targets
