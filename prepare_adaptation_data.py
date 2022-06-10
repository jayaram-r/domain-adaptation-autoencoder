"""
Preparing the target domain dataset for adaptation and testing.
Data from the target distribution will be split up equally (50/50) for adaptation and testing.
The first data split is used for adaptation or finetuning or retraining (depending on the method) of the MDN channel and autoencoder.
The second data split will be used solely for testing/evaluation of performance on the target distribution.
Different random subsets of the required target size (e.g 10 samples per symbol) are sub-sampled from the first data split, and used
by the adaptation and fine-tuning methods.

Usage example:
n_adapt_per_class='5,10,20,30,40,50'
python prepare_adaptation_data.py -d <data directory> -b <base directory for saving the data> --n-trials <number of random trials> --nad $n_adapt_per_class

"""
import sys
import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
import scipy.io as sio

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from helpers.utils import convert_to_complex_ndarray
from helpers.constants import *


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', required=True, help='Data directory with the target domain data files')
    parser.add_argument('-b', '--base-dir', default='.', help='Directory for saving the adaptation data files.')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of random trials')
    parser.add_argument('--n-adapt-per-symbol', '--nad', default='5,10,20,30,40,50',
                        help='Number of target domain adaptation samples per constellation symbol. Specified as a '
                             'comma-separated string of values')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(args.base_dir):
        os.makedirs(args.base_dir)

    return args


def create_directories(data_dir, base_dir, n_trials):
    # Sub-directories for the training and test data split
    data_dir_train = "{}_train".format(data_dir)
    if not os.path.isdir(data_dir_train):
        os.makedirs(data_dir_train)

    data_dir_test = "{}_test".format(data_dir)
    if not os.path.isdir(data_dir_test):
        os.makedirs(data_dir_test)

    # Sub-directories for each adaptation trial
    trial_direcs = []
    for t in range(n_trials):
        d = os.path.join(base_dir, 'trial{:d}'.format(t))
        trial_direcs.append(d)
        if not os.path.isdir(d):
            os.makedirs(d)

    return data_dir_train, data_dir_test, trial_direcs


def load_data_files(tx_file, rx_file, labels_file, shuffle=False):
    labels = loadmat_helper(labels_file)  # the one-hot-coded labels for the transmitted message
    x_data = loadmat_helper(tx_file)      # transmitted symbol data
    y_data = loadmat_helper(rx_file)      # received symbol data

    n_samp, n_symb = labels.shape
    assert x_data.shape[0] == n_samp, "Error in the size of data array 'x_data'"
    assert y_data.shape[0] == n_samp, "Error in the size of data array 'y_data'"
    # Complex array to 2d real array
    x_data1 = np.zeros((n_samp, 2))
    y_data1 = np.zeros((n_samp, 2))
    x_data1[:, 0] = np.reshape(x_data.real, n_samp)
    x_data1[:, 1] = np.reshape(x_data.imag, n_samp)
    y_data1[:, 0] = np.reshape(y_data.real, n_samp)
    y_data1[:, 1] = np.reshape(y_data.imag, n_samp)
    if shuffle:
        ind = np.random.permutation(n_samp)
        x_data1 = x_data1[ind, :]
        y_data1 = y_data1[ind, :]
        labels = labels[ind, :]

    data_dict = {'x_data': x_data1.astype(np.float32),
                 'y_data': y_data1.astype(np.float32),
                 'labels': labels.astype(np.float32),
                 'labels_int': np.argmax(labels, axis=1)}
    return data_dict


def save_data_files(data_dict, data_dir):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # Save the data arrays into mat files in the required complex number format
    mdict = {'tx_send': convert_to_complex_ndarray(data_dict['x_data'])}
    sio.savemat(os.path.join(data_dir, TX_DATA_BASENAME), mdict)

    mdict = {'rx_channel': convert_to_complex_ndarray(data_dict['y_data'])}
    sio.savemat(os.path.join(data_dir, RX_DATA_BASENAME), mdict)

    mdict = {'labels': data_dict['labels']}
    sio.savemat(os.path.join(data_dir, LABELS_BASENAME), mdict)


def create_train_test_split(data_dir, data_dir_train, data_dir_test):
    x_data_file = os.path.join(data_dir, TX_DATA_BASENAME)
    y_data_file = os.path.join(data_dir, RX_DATA_BASENAME)
    labels_file = os.path.join(data_dir, LABELS_BASENAME)
    data_all = load_data_files(x_data_file, y_data_file, labels_file)

    # 50/50 train-test split. This should be stratified by class
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    ind_tr, ind_te = next(sss.split(data_all['y_data'], data_all['labels_int']))
    data_train = {'x_data': data_all['x_data'][ind_tr, :],
                  'y_data': data_all['y_data'][ind_tr, :],
                  'labels': data_all['labels'][ind_tr, :],
                  'labels_int': data_all['labels_int'][ind_tr]}
    data_test = {'x_data': data_all['x_data'][ind_te, :],
                 'y_data': data_all['y_data'][ind_te, :],
                 'labels': data_all['labels'][ind_te, :],
                 'labels_int': data_all['labels_int'][ind_te]}

    # Save the training and test data files to the appropriate directories
    save_data_files(data_train, data_dir_train)
    save_data_files(data_test, data_dir_test)

    return data_train, data_test


def create_adaptation_subsets(data_train, trial_direcs, n_adapt_per_symbol):
    n_trials = len(trial_direcs)
    n_symb = data_train['labels'].shape[1]
    n_samp = data_train['y_data'].shape[0]
    for t in range(n_trials):
        ind_shuff = np.random.permutation(n_samp)
        x_data = data_train['x_data'][ind_shuff, :]
        y_data = data_train['y_data'][ind_shuff, :]
        labels = data_train['labels'][ind_shuff, :]
        labels_int = data_train['labels_int'][ind_shuff]
        for m in n_adapt_per_symbol:
            x_data1 = []
            y_data1 = []
            labels1 = []
            for i in range(n_symb):
                # m samples from symbol (class) i
                ind_curr = np.where(labels_int == i)[0]
                ind_curr = ind_curr[:m]
                x_data1.append(x_data[ind_curr, :])
                y_data1.append(y_data[ind_curr, :])
                labels1.append(labels[ind_curr, :])

            x_data1 = np.concatenate(x_data1, axis=0)
            y_data1 = np.concatenate(y_data1, axis=0)
            labels1 = np.concatenate(labels1, axis=0)
            data_temp = {'x_data': x_data1,
                         'y_data': y_data1,
                         'labels': labels1,
                         'labels_int': np.argmax(labels1, axis=1)}
            # Save the training and test data files to the appropriate directories
            direc = os.path.join(trial_direcs[t], 'data_adapt_{:d}'.format(m))
            save_data_files(data_temp, direc)


def main():
    args = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # List of number of adaptation samples
    n_adapt_per_symbol = [int(a.strip(' ')) for a in args.n_adapt_per_symbol.split(',')]

    # Prepare the data directories
    data_dir_train, data_dir_test, trial_direcs = create_directories(args.data_dir, args.base_dir, args.n_trials)

    # Load the data from the mat files, create a 50/50 stratified train-test split, and save the data files
    data_train, data_test = create_train_test_split(args.data_dir, data_dir_train, data_dir_test)

    # Create random data subsets of different target sizes for adaptation
    create_adaptation_subsets(data_train, trial_direcs, n_adapt_per_symbol)


if __name__ == '__main__':
    main()
