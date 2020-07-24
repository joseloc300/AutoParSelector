import random
import math
import numpy as np


def stratified_regressions_sampling(loop_features, loop_targets, origin_files, sampling_params):
    train_prefixes = sampling_params["train_prefixes"]
    test_prefixes = sampling_params["test_prefixes"]
    train_ratio = sampling_params["train_ratio"]
    block_size = sampling_params["block_size"]
    stratify = sampling_params["stratify"]
    randomize = sampling_params["randomize"]
    r_seed = sampling_params["r_seed"]
    r_shuffles = sampling_params["r_shuffles"]

    # new_loop_features = np.asarray(loop_features[:])
    # new_loop_targets = np.asarray(loop_targets[:])

    new_x_train = np.zeros((0, len(loop_features[0])))
    new_y_train = np.zeros(0)
    new_x_test = np.zeros((0, len(loop_features[0])))
    new_y_test = np.zeros(0)

    new_loop_features = np.zeros((0, len(loop_features[0])))
    new_loop_targets = np.zeros(0)

    for i in range(len(origin_files)):
        train_file = False
        test_file = False
        if len(train_prefixes) == 0:
            train_file = True
        else:
            for train_prefix in train_prefixes:
                if origin_files[i].startswith(train_prefix.lower()):
                    train_file = True
                    break

        if len(test_prefixes) == 0:
            test_file = True
        else:
            for test_prefix in test_prefixes:
                if origin_files[i].startswith(test_prefix.lower()):
                    test_file = True
                    break

        if train_file and not test_file:
            new_x_train = np.append(new_x_train, [loop_features[i]], axis=0)
            new_y_train = np.append(new_y_train, loop_targets[i])
        elif test_file and not train_file:
            new_x_test = np.append(new_x_test, [loop_features[i]], axis=0)
            new_y_test = np.append(new_y_test, loop_targets[i])
        elif train_file and test_file:
            new_loop_features = np.append(new_loop_features, [loop_features[i]], axis=0)
            new_loop_targets = np.append(new_loop_targets, loop_targets[i])

    print(len(new_x_train))
    print(len(new_x_test))
    print(len(new_loop_features))

    ARRAY_SIZE = len(new_loop_targets)
    N_BLOCKS = math.ceil(ARRAY_SIZE / block_size)

    if stratify:
        # order by regression target, bubble sort
        for i in range(ARRAY_SIZE):
            for j in range(0, ARRAY_SIZE - i - 1):
                if new_loop_targets[j] > new_loop_targets[j + 1]:
                    new_loop_targets[j], new_loop_targets[j + 1] = new_loop_targets[j + 1], new_loop_targets[j]
                    new_loop_features[j], new_loop_features[j + 1] = new_loop_features[j + 1], new_loop_features[j]

    for i in range(N_BLOCKS):
        BLOCK_IDX = i * block_size

        # block length is equal to block size except when it's the last block where we need to check against array size
        BLOCK_LEN = ARRAY_SIZE - BLOCK_IDX - 1 if i == N_BLOCKS - 1 else block_size

        if randomize:
            random.seed(r_seed)
            if r_shuffles is None:
                r_shuffles = BLOCK_LEN * 2

            for j in range(r_shuffles):
                idx_0 = random.randrange(BLOCK_IDX, BLOCK_IDX + BLOCK_LEN)
                idx_1 = random.randrange(BLOCK_IDX, BLOCK_IDX + BLOCK_LEN)

                new_loop_targets[idx_0], new_loop_targets[idx_1] = new_loop_targets[idx_1], new_loop_targets[idx_0]
                new_loop_features[idx_0], new_loop_features[idx_1] = new_loop_features[idx_1], new_loop_features[idx_0]

        TRAIN_CUT_INDEX = BLOCK_IDX + (BLOCK_LEN * train_ratio)

        for j in range(BLOCK_IDX, BLOCK_IDX + BLOCK_LEN):
            loop_features_row_to_add = np.asarray(new_loop_features[j])
            loop_target_to_add = np.asarray(new_loop_targets[j])

            if j < TRAIN_CUT_INDEX:
                new_x_train = np.append(new_x_train, [loop_features_row_to_add], axis=0)
                new_y_train = np.append(new_y_train, loop_target_to_add)
            else:
                new_x_test = np.append(new_x_test, [loop_features_row_to_add], axis=0)
                new_y_test = np.append(new_y_test, loop_target_to_add)

    return new_x_train, new_x_test, new_y_train, new_y_test
