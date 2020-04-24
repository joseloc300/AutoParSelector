import json
import numpy as np
import re

from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.isotonic import IsotonicRegression

N_FEATURES = 7

NESTED_LEVEL_INDEX = 0
IS_MAIN_LOOP_INDEX = 1
N_ANCESTORS_INDEX = 2
N_DESCENDANTS_INDEX = 3
N_PRIVATES_INDEX = 4
N_FIRST_PRIVATES_INDEX = 5
N_REDUCTIONS_INDEX = 6


def main():
    loop_features, loop_targets = load_data()

    linear_models(loop_features, loop_targets)
    decision_tree_regressor(loop_features, loop_targets)
    # isotonic_regression(loop_features, loop_targets)

    exit(0)


def split_train_test_data(loop_features, loop_targets, train_ratio):
    x = np.asarray(loop_features)
    y = np.asarray(loop_targets)

    split_point = int(len(loop_features) * train_ratio)

    x_train, x_test = np.split(x, [split_point])
    y_train, y_test = np.split(y, [split_point])

    return x_train, x_test, y_train, y_test


def linear_models(loop_features, loop_targets):
    ordinary_least_squares(loop_features, loop_targets)
    ridge_regression(loop_features, loop_targets)
    lasso(loop_features, loop_targets)
    elastic_net(loop_features, loop_targets)


def isotonic_regression(loop_features, loop_targets):
    train_ratio = 0.8
    x_train, x_test, y_train, y_test = split_train_test_data(loop_features, loop_targets, train_ratio)

    reg = IsotonicRegression()
    # reg.fit(x_train, y_train)
    x_new = reg.fit_transform(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Isotonic Regression")
    print(score)


# 1.1.1
def ordinary_least_squares(loop_features, loop_targets):
    train_ratio = 0.8
    x_train, x_test, y_train, y_test = split_train_test_data(loop_features, loop_targets, train_ratio)

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Ordinary Least Squares")
    print(score)


# 1.1.2
def ridge_regression(loop_features, loop_targets):
    train_ratio = 0.8
    x_train, x_test, y_train, y_test = split_train_test_data(loop_features, loop_targets, train_ratio)

    reg = linear_model.Ridge(alpha=0.5)
    reg.fit(x_train, y_train)
    score = reg.score(x_test, y_test)

    print("Ridge Regression")
    print(score)

    reg_2 = linear_model.RidgeCV(alphas=np.arange(0.01, 1.0, 0.01), cv=None)
    reg_2.fit(x_train, y_train)
    score_2 = reg_2.score(x_test, y_test)

    print("Ridge Regression CV")
    print(score_2)


# 1.1.3
def lasso(loop_features, loop_targets):
    train_ratio = 0.8
    x_train, x_test, y_train, y_test = split_train_test_data(loop_features, loop_targets, train_ratio)

    reg = linear_model.Lasso(alpha=0.5)
    reg.fit(x_train, y_train)

    score = reg.score(x_test, y_test)

    print("Lasso")
    print(score)

    reg_2 = linear_model.LassoCV(eps=0.001, n_alphas=100, cv=None)
    reg_2.fit(x_train, y_train)

    score_2 = reg_2.score(x_test, y_test)

    print("Lasso CV")
    print(score_2)


# 1.1.5
def elastic_net(loop_features, loop_targets):
    pass


def decision_tree_regressor(loop_features, loop_targets):
    train_ratio = 0.8
    x_train, x_test, y_train, y_test = split_train_test_data(loop_features, loop_targets, train_ratio)

    reg_1 = DecisionTreeRegressor(max_depth=2)
    reg_2 = DecisionTreeRegressor(max_depth=5)

    reg_1.fit(x_train, y_train)
    reg_2.fit(x_train, y_train)

    score_1 = reg_1.score(x_test, y_test)
    score_2 = reg_2.score(x_test, y_test)

    print("Decision Tree Regressor")
    print(score_1)
    print(score_2)


def load_data():
    with open('/home/josecunha/Desktop/Clava/results/polybench_2020-04-13T15:47:30.735Z.json') as f:
        data = json.load(f)

    n_loops = 0
    N_FEATURES = 0

    loop_features = []
    loop_targets = []

    runs_per_version = int(data["runsPerVersion"])

    for i in range(data["totalBenchmarks"]):
        new_loop_features, new_loop_targets = parse_benchmark_simple(data["benchmarks"][i], runs_per_version)
        loop_features.extend(new_loop_features)
        loop_targets.extend(new_loop_targets)

    return loop_features, loop_targets


def parse_benchmark_simple(benchmark, runs_per_version):
    n_loops = benchmark["nLoops"]
    if n_loops == 0:
        return [], []

    if benchmark["benchmark"] == "/home/josecunha/Desktop/Clava/sources/polybench-c-3.2/stencils/fdtd-2d":
        return [], []

    loops_info_orig = benchmark["loops"]
    loops_info = get_loops_info(loops_info_orig)
    avg_seq_times = get_seq_avg_times(benchmark["seq"], runs_per_version)

    loop_features = []
    loop_targets = []

    for version in benchmark["par"]["versions"]:
        par_loops = version["parLoops"]
        loops_family_info = get_loops_family_info(par_loops, loops_info_orig)
        main_loop = version["mainLoop"]

        for measure in version["measures"]:
            loop_index = measure["loopIndex"]
            if loop_index not in par_loops:
                continue

            avg_time = 0
            for run in measure["runs"]:
                avg_time += run["value"]

            avg_time /= runs_per_version

            speedup = avg_seq_times[loop_index] / avg_time

            loop_info_copy = loops_info[loop_index].copy()

            loop_index_str = str(loop_index)
            n_ancestors = loops_family_info[loop_index_str]["ancestors"]
            n_descendants = loops_family_info[loop_index_str]["descendants"]
            loop_info_copy[N_ANCESTORS_INDEX] = n_ancestors
            loop_info_copy[N_DESCENDANTS_INDEX] = n_descendants

            loop_features.append(loop_info_copy)
            loop_targets.append(speedup)

    return loop_features, loop_targets


def get_loops_family_info(loops, loop_info_orig):
    loops_family_info = {}
    for loop_key in loop_info_orig:
        if int(loop_key) not in loops:
            continue

        loop = loop_info_orig[loop_key]

        loop_family_info = {}
        loop_family_info["ancestors"] = 0
        loop_family_info["descendants"] = 0

        curr_loop_id = loop["id"]

        for loop_2_key in loop_info_orig:
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

    for loop_key in loops:
        new_loop = np.zeros(N_FEATURES, np.int)
        new_loop[NESTED_LEVEL_INDEX] = loops[loop_key]["nestedLevel"]
        new_loop[IS_MAIN_LOOP_INDEX] = int(loops[loop_key]["isMainLoop"])

        pragma_info = get_pragma_info(loops[loop_key]["pragmas"])

        new_loop[N_PRIVATES_INDEX] = pragma_info["n_privates"]
        new_loop[N_FIRST_PRIVATES_INDEX] = pragma_info["n_first_privates"]
        new_loop[N_REDUCTIONS_INDEX] = pragma_info["n_reductions"]

        loops_info.append(new_loop)

    return loops_info


def get_pragma_info(pragmas):
    pragma_info = {
        "n_privates": 0,
        "n_first_privates": 0,
        "n_reductions": 0
    }

    for pragma in pragmas:
        if pragma.startswith("#pragma scop"):
            continue

        private_result = re.search(" private\([^)]+\)", pragma)
        if private_result:
            private = private_result.group()[1:]
            n_privates = private.split(",")
            pragma_info["n_privates"] = len(n_privates)

        first_private_result = re.search("firstprivate\([^)]+\)", pragma)
        if first_private_result:
            n_first_privates = first_private_result.group().split(",")
            pragma_info["n_first_privates"] = len(n_first_privates)

        reduction_result = re.search("reduction\([^)]+\)", pragma)
        if reduction_result:
            pragma_info["n_reductions"] = 1

    return pragma_info


def get_seq_avg_times(seq, runs_per_version):
    avg_times = []

    for loop_data in seq:
        avg_time = 0

        for run in loop_data["runs"]:
            avg_time += run["value"]

        avg_time /= runs_per_version
        avg_times.append(avg_time)

    return avg_times


main()
