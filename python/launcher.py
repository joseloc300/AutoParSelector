from data_loading import load_data
from data_loading import load_main_params
from data_analysis import analyze_data
from data_processing import process_data
from sk_learn_models import run_sk_models
from tf_models import run_tf_models
import utils


def main():
    main_params = load_main_params()
    loop_features, loop_targets, origin_files = load_data(main_params["read_data_prefixes"])

    analyze_data(loop_features, loop_targets, main_params["data_analysis"], True)
    loop_features, loop_targets = process_data(loop_features, loop_targets, main_params["data_processing"])
    analyze_data(loop_features, loop_targets, main_params["data_analysis"], False)

    x_train, x_test, y_train, y_test = utils.stratified_regressions_sampling(loop_features, loop_targets, origin_files,
                                                                             main_params["sampling_params"])

    run_sk_models(x_train, x_test, y_train, y_test)
    run_tf_models(x_train, x_test, y_train, y_test)

    exit(0)


main()
