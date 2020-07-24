import json
import re
import os
import numpy as np


def load_main_params():
    print("+ Load Main Params")
    main_params_path = "./params/main_params.json"
    with open(main_params_path) as f:
        main_params = json.load(f)

    return main_params


def load_data(read_data_mode):
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

    origin_files = []
    loop_features = []
    loop_targets = []

    for json_file_info in found_json_files:
        with open(json_file_info["path"]) as f:
            data = json.load(f)

        for i in range(data["totalBenchmarkVersions"]):
            new_loop_features, new_loop_targets = parse_benchmark_simple(data["benchmarks"][i])
            loop_features.extend(new_loop_features)
            loop_targets.extend(new_loop_targets)
            for j in range(len(new_loop_features)):
                origin_files.append(json_file_info["filename"])

    print("++ Rows Loaded: " + str(len(loop_features)))

    return loop_features, loop_targets, origin_files


def parse_benchmark_simple(benchmark):
    n_versions = benchmark["par"]["nVersions"]

    if n_versions == 0:
        return [], []

    loops_info_orig = benchmark["loops"]
    loops_info = get_loops_info(loops_info_orig)
    avg_seq_times = get_seq_avg_times(benchmark["seq"])

    bench_loop_features = []
    bench_loop_targets = []

    for version in benchmark["par"]["versions"]:
        par_loops = version["parLoops"]
        # loops_family_info = get_loops_family_info(par_loops, loops_info_orig)
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

        # TODO improve this
        if n_runs == 0:
            # avg_time = 1
            print("count")
            continue
        else:
            avg_time /= n_runs

        speedup = avg_seq_times[loop_index] / avg_time

        loop_info_copy = loops_info[loop_index].copy()
        loop_info_copy = np.append(loop_info_copy, len(par_loops))

        bench_loop_features.append(loop_info_copy)
        bench_loop_targets.append(speedup)

    return bench_loop_features, bench_loop_targets


def get_loops_family_info(par_loops, loop_info_orig):
    loops_family_info = {}
    for loop_key in loop_info_orig:
        if int(loop_key) not in par_loops:
            continue

        loop = loop_info_orig[loop_key]

        loop_family_info = {
            "ancestors": 0,
            "descendants": 0
        }

        curr_loop_id = loop["id"]

        for loop_2_key in loop_info_orig:

            # TODO é suposto existir? nao estava ca, mas penso ser necesario para garantir que so contamos
            # ancestors/descendants que estao a ser paralelizado em simultaneo com o loop que estamos a analisar
            if int(loop_2_key) not in par_loops:
                continue

            loop_2 = loop_info_orig[loop_2_key]
            if loop_2["id"] == curr_loop_id:
                continue

            relation = compare_rank(loop["rank"], loop_2["rank"])
            if relation == 0:
                loop_family_info["ancestors"] += 1
            elif relation == 1:
                loop_family_info["descendants"] += 1

        loops_family_info[loop_key] = loop_family_info

    return loops_family_info


# returns loop_rank_1 relation to loop_rank_0
# return 0 if ancestor, 1 if descendant and 2 otherwise
def compare_rank(loop_rank_0, loop_rank_1):
    if check_rank_ancestor(loop_rank_0, loop_rank_1):
        return 0
    elif check_rank_descendant(loop_rank_0, loop_rank_1):
        return 1
    else:
        return 2


def check_rank_ancestor(loop_rank_0, loop_rank_1):
    if len(loop_rank_0) >= len(loop_rank_1):
        return False

    for i in range(len(loop_rank_0)):
        if loop_rank_1[i] != loop_rank_0[i]:
            return False

    return True


def check_rank_descendant(loop_rank_0, loop_rank_1):
    if len(loop_rank_0) <= len(loop_rank_1):
        return False

    for i in range(len(loop_rank_1)):
        if loop_rank_1[i] != loop_rank_0[i]:
            return False

    return True


def get_loops_info(loops):
    loops_info = []

    IGNORED_FEATURES = ["id", "parentLoopId", "origLine", "rank", "index"]

    for loop_key in loops:
        current_loop = loops[loop_key]
        new_loop = np.zeros(0, np.int)

        for loop_feature in current_loop:
            if loop_feature in IGNORED_FEATURES:
                continue

            if loop_feature == "ompPragma":
                pragma_info = get_pragma_info(loops[loop_key]["ompPragma"])
                new_loop = np.append(new_loop, list(pragma_info.values()))
            elif loop_feature == "instructionInfo":
                for instruction_feature in current_loop[loop_feature]:
                    if instruction_feature == "joinpoints" or instruction_feature == "recursiveJoinpoints":
                        new_loop = np.append(new_loop, list(current_loop[loop_feature][instruction_feature].values()))
                    else:
                        new_loop = np.append(new_loop, current_loop[loop_feature][instruction_feature])
            else:
                new_loop = np.append(new_loop, current_loop[loop_feature])

        loops_info.append(new_loop)

    return loops_info


# TODO need to test reductions
def get_pragma_info(pragma):
    pragma_info = {
        "n_privates": 0,
        "n_first_privates": 0,
        "n_reductions": 0,
        "n_scalar_reductions": 0,
        "n_array_reductions": 0
    }

    private_result = re.search(" private([^)]+)", pragma)
    if private_result:
        private = private_result.group()[1:]
        n_privates = private.split(",")
        pragma_info["n_privates"] = len(n_privates)

    first_private_result = re.search("firstprivate([^)]+)", pragma)
    if first_private_result:
        n_first_privates = first_private_result.group().split(",")
        pragma_info["n_first_privates"] = len(n_first_privates)

    reduction_results = re.findall("reduction ([^)]+)", pragma)
    if len(reduction_results) > 0:
        pragma_info["n_reductions"] = len(reduction_results)
        for reduction in reduction_results:
            if "[" in reduction:
                pragma_info["n_array_reductions"] += 1
            else:
                pragma_info["n_scalar_reductions"] += 1

    return pragma_info


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

        # TODO improve this
        if n_runs == 0:
            avg_times[loop_index] = 1
        else:
            avg_time /= n_runs
            avg_times[loop_index] = avg_time

    return avg_times
