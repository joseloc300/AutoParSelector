import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt


def analyze_data(loop_features, loop_targets, feature_names, data_analysis_params, pre_data_processing, classification):
    if (pre_data_processing and not (data_analysis_params["pre_data_processing"])) or \
            (not pre_data_processing and not (data_analysis_params["post_data_processing"])):
        return

    print("+ Data Analysis")
    if data_analysis_params["data_correlation"]["enabled"]:
        data_correlation(loop_features, loop_targets, feature_names, data_analysis_params["data_correlation"],
                         pre_data_processing, classification)


def data_correlation(loop_features, loop_targets, feature_names, correlation_params, pre_data_processing,
                     classification):
    print("++ Data correlation")

    corr_matrix = get_correlation_matrix(loop_features, loop_targets, feature_names)
    if correlation_params["graphical_results"]:
        sn.heatmap(corr_matrix, annot=True)
        plt.show()

    if pre_data_processing:
        file_path = "./results/corr_matrix_pre.csv"
    else:
        if classification:
            file_path = "./results/corr_matrix_classification.csv"
        else:
            file_path = "./results/corr_matrix_regression.csv"

    corr_matrix.to_csv(file_path)


def get_correlation_matrix(loop_features, loop_targets, feature_names):
    converted_data = convert_data_frame_format(loop_features, loop_targets)
    df = pd.DataFrame(data=converted_data)

    col_names = feature_names.copy()
    col_names.append("target")
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.fillna(0)
    corr_matrix.index = col_names
    corr_matrix.columns = col_names

    return corr_matrix


def convert_data_frame_format(loop_features, loop_targets):
    numpy_array = np.array(loop_features)
    transpose = numpy_array.T
    transpose = np.append(transpose, [loop_targets], axis=0)
    transpose_array = transpose.tolist()

    converted_data = {}
    columns = []

    for i in range(len(transpose_array)):
        converted_data[str(i)] = transpose_array[i]
        columns.append(str(i))

    return converted_data
