import numpy as np


import logging
logger = logging.getLogger("default_log")


def getdata(file_path,files):
    data_x,data_y = [],[]
    for i in files:
        file = file_path+'/'+i
        f = np.load(file)
        x = f['x']
        y = f['y']
        x = np.squeeze(x)
        x = x[:, :, np.newaxis, np.newaxis]
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        # print(x.shape,y.shape)
        data_x.append(x)
        data_y.append(y)
    return data_x,data_y

def save_seq_ids(fname, ids):
    """Save sequence of IDs into txt file."""
    with open(fname, "w") as f:
        for _id in ids:
            f.write(str(_id) + "\n")


def load_seq_ids(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    with open(fname, "r") as f:
        for line in f:
            ids.append(line.strip())
    ids = np.asarray(ids)
    return ids


def print_n_samples_each_class(labels):
    """Print the number of samples in each class."""

    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        logger.info("{}: {}".format(class_dict[c], n_samples))


def compute_portion_each_class(labels):
    """Determine the portion of each class."""

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    class_portions = np.zeros(len(unique_labels), dtype=np.float32)
    for c in unique_labels:
        n_class_samples = len(np.where(labels == c)[0])
        class_portions[c] = n_class_samples/float(n_samples)
    return class_portions


def get_balance_class_oversample(x, y):
    """Balance the number of samples of all classes by (oversampling).

    The process is as follows:
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """

    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def get_balance_class_sample(x, y):
    """Balance the number of samples of all classes by sampling.

    The process is as follows:
        1. Find the class that has the smallest number of samples
        2. Randomly select samples in each class equal to that smallest number
    """

    class_labels = np.unique(y)
    n_min_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_min_classes == -1:
            n_min_classes = n_samples
        elif n_min_classes > n_samples:
            n_min_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        sample_idx = np.random.choice(idx, size=n_min_classes, replace=False)
        balance_x.append(x[sample_idx])
        balance_y.append(y[sample_idx])
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y

