import json
import os
import numpy as np


def load_main_params():
    print("+ Load Main Params")
    main_params_path = "./params/main_params.json"
    with open(main_params_path) as f:
        main_params = json.load(f)

    return main_params


def load_data(read_data_mode, is_prediction):
    print("+ Load data")

    prefix_filters = read_data_mode
    found_json_files = []

    base_path = './data/'
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if os.path.isfile(entry_path) and entry.endswith(".json"):
            if len(prefix_filters) == 0:
                entry_info = {
                    "path": entry_path,
                    "filename": entry.lower()
                }
                found_json_files.append(entry_info)
            else:
                for prefix_filter in prefix_filters:
                    if entry.lower().startswith(prefix_filter.lower()):
                        entry_info = {
                            "path": entry_path,
                            "filename": entry.lower()
                        }
                        found_json_files.append(entry_info)
                        break

    loop_features = []
    loop_targets = []
    loop_info = []
    feature_names = []

    for json_file_info in found_json_files:
        with open(json_file_info["path"]) as f:
            data = json.load(f)

        for i in range(data["totalBenchmarkVersions"]):
            new_loop_features, new_loop_targets, new_loop_info, new_feature_names = parse_benchmark_simple(
                data["benchmarks"][i], is_prediction)

            loop_features.extend(new_loop_features)
            loop_targets.extend(new_loop_targets)
            loop_info.extend(new_loop_info)
            if len(feature_names) == 0:
                feature_names = new_feature_names

    print("++ Rows Loaded: " + str(len(loop_features)))

    return loop_features, loop_targets, loop_info, feature_names


def parse_benchmark_simple(benchmark, is_prediction):
    n_versions = benchmark["par"]["nVersions"]
    problem_size_flag = benchmark["problemSizeFlag"]

    if not is_prediction and n_versions == 0:
        return [], []

    loops_info_orig = benchmark["loops"]
    loops_features, loops_info, feature_names = get_loops_features(loops_info_orig, problem_size_flag)
    avg_seq_times = get_seq_avg_times(benchmark["seq"])

    bench_loop_features = []
    bench_loop_targets = []

    if n_versions > 0:
        for version in benchmark["par"]["versions"]:
            main_loop_index = version["mainLoop"]

            measure = version["measures"][0]
            loop_index = measure["loopIndex"]

            if loop_index != main_loop_index:
                continue

            avg_time = 0
            n_runs = 0
            for run in measure["runs"]:
                if run["value"] is not None:
                    avg_time += run["value"]
                    n_runs += 1

            if n_runs == 0:
                continue
            else:
                avg_time /= n_runs

            speedup = avg_seq_times[loop_index] / avg_time

            loops_features_copy = loops_features[loop_index].copy()

            bench_loop_features.append(loops_features_copy)
            bench_loop_targets.append(speedup)
    else:
        for loop_features_instance in loops_features:
            loops_features_copy = loop_features_instance.copy()
            bench_loop_features.append(loops_features_copy)

    return bench_loop_features, bench_loop_targets, loops_info, feature_names


def get_loops_features(loops, problem_size_flag):
    loops_features = []
    loops_info = np.zeros(shape=(0, 3))
    feature_names = []

    first_iteration = True

    for loop_key in loops:
        current_loop = loops[loop_key]
        new_loop = np.zeros(0, np.int)

        new_loop_info = np.asarray(list([current_loop["id"], current_loop["ompPragma"], problem_size_flag]))

        if first_iteration:
            feature_names = get_feature_names(current_loop["features"])
            first_iteration = False

        for loop_feature_type in current_loop["features"]:
            for loop_feature_name in current_loop["features"][loop_feature_type]:
                if loop_feature_name == "instructionInfo":
                    for instruction_feature_group in current_loop["features"][loop_feature_type][loop_feature_name]:
                        new_loop = np.append(new_loop, list(
                            current_loop["features"][loop_feature_type][loop_feature_name][
                                instruction_feature_group].values()))
                else:
                    new_loop = np.append(new_loop, current_loop["features"][loop_feature_type][loop_feature_name])

        loops_features.append(new_loop)
        loops_info = np.vstack((loops_info, new_loop_info))

    return loops_features, loops_info, feature_names


def get_seq_avg_times(seq):
    avg_times = {}

    for loop_data in seq:
        avg_time = 0
        loop_index = loop_data["loopIndex"]

        n_runs = 0
        for run in loop_data["runs"]:
            if run["value"] is not None:
                avg_time += run["value"]
                n_runs += 1

        if n_runs == 0:
            avg_times[loop_index] = 0
        else:
            avg_time /= n_runs
            avg_times[loop_index] = avg_time

    return avg_times


def get_feature_names(features):
    feature_names = []

    values_to_check = list(features.values())
    keys_to_check = list(features.keys())

    while len(values_to_check) > 0:
        if isinstance(values_to_check[0], dict):
            for key in values_to_check[0].keys():
                values_to_check.append(values_to_check[0][key])
                keys_to_check.append(keys_to_check[0] + "/" + key)
        else:
            feature_names.append(keys_to_check[0])

        values_to_check.pop(0)
        keys_to_check.pop(0)

    return feature_names
