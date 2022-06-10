"""
Main script for decoding samples from FPGA using either a standard autoncoder or an adapted autoencoder.
For a description of the command line inputs, type:
`python adapt_autoencoder_decode.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--type-autoencoder or --ta: type of autoencoder model ('standard', 'symbol_estimation_mmse', 'symbol_estimation_map'), (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--autoencoder-model-file or --amf: path to the autoencoder model file or directory (REQUIRED).
--adaptation-params-file or --apf: To evaluate the adapted autoencoder, specify the path to the adaptation parameters
                                   file using this option. By default, the script `adapt_autoencoder_measure.py` saves
                                   the adaptation parameters to a numpy file named `adaptation_params.npy`.
                                   Omit this option to evaluate the autoencoder without adaptation.
--data-dir or -d: path to the data directory where the test data files are read from.
--output-dir or -o: output directory path. The BER of the autoencoder on the decoded samples will be written to a .mat file here.

Most of the other options can be left at their default values unless there is a specific need to change them.
Any non-default options that were used for training the autoencoder using `train_autoencoder.py` should also be set in
this script (if applicable). For example, if the dimension of the encoding is 4 and the number of components in the
channel MDN is 3, then the following options should be specified in both the training and adaptation scripts:
`--de 4 --nc 3`

DATA FILES:
Before running, ensure that the data files for decoding are available in the data directory specified by the
command line option `--data-dir`. By default this is a directory `./data_test` under the current working directory.
Specifically, the following two mat files are required:
- `rx_symbols.mat`: File containing the received symbols.
- `labels.mat`: File containing the transmitted one-hot-coded message labels.

USAGE EXAMPLES:
    Assume that the scripts `train_autoencoder.py` and `adapt_autoencoder_measure.py` have been run on data from FPGA,
    and that the channel model, autoencoder model, and adaptation parameters are saved at the following locations:
    cmf="${model_dir}/channel_model/channel"
    amf="${model_dir}/autoencoder_symbol_estimation_map/autoencoder"
    apf="${model_dir}/adaptation_params.npy"
    data_dir="${PWD}/data_test"
    output_dir=${PWD}/outputs

    python adapt_autoencoder_decode.py --n-bits 4 --cmf $cmf --amf $amf --apf $apf --ta symbol_estimation_map -d $data_dir -o $output_dir

    To evaluate the autoencoder without adaptation, simply omit the option `--apf`.
    After running, inspect the performance metric (BER) file saved to the output directory.
"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    load_data_transceiver,
    get_autoencoder_data_filename,
    save_perf_metrics,
    load_perf_metrics,
    loadmat_helper
)
from helpers.MDN_classes import initialize_MDN_model
from helpers.autoencoder_classes import (
    initialize_autoencoder,
    initialize_autoencoder_adapted
)
from helpers.metrics import calculate_metric_autoencoder
from helpers.standard_modulations import StandardQAM
from helpers.constants import *
# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_inputs():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--n-bits', '--nb', type=int, required=True, help='number of bits per symbol')
    parser.add_argument('--channel-model-file', '--cmf', required=True,
                        help='Path to the file or directory containing the trained channel model weights.')
    parser.add_argument('--autoencoder-model-file', '--amf', required=True,
                        help='Path to the file or directory containing the trained autoencoder model weights.')
    parser.add_argument('--adaptation-params-file', '--apf', default='',
                        help='Path to the numpy file with the saved adaptation parameters. If not specified, no '
                             'the autoencoder model without any adaptation is evaluated.')
    parser.add_argument('--type-autoencoder', '--ta', required=True,
                        choices=['standard', 'symbol_estimation_mmse', 'symbol_estimation_map'],
                        help='Type of autoencoder model trained.')
    # Data and output directory
    parser.add_argument('--data-dir', '-d', default='./data_test', help='Directory for the saved data files')
    parser.add_argument('--output-dir', '-o', default='./outputs_decode',
                        help='Directory for saving the output files, plots etc')
    # Model hyper-parameters
    parser.add_argument('--dim-encoding', '--de', type=int, default=2,
                        help='Dimension of the encoded symbols. Should be an even integer')
    parser.add_argument('--n-components', '--nc', type=int, default=5,
                        help='Number of components in the Gaussian mixture density network')
    parser.add_argument('--n-hidden', '--nh', type=int, default=100, help='Size of the hidden fully-connected layer')
    # Optimizer configuration
    parser.add_argument('--batch-size', '--bs', type=int, default=128, help='Batch size for optimization')
    parser.add_argument('--n-epochs', '--ne', type=int, default=100, help='Number of optimization epochs')
    parser.add_argument('--optim-method', '--om', choices=['adam', 'sgd'], default='sgd',
                        help="Optimization method to use: 'adam' or 'sgd'")
    parser.add_argument('--learning-rate', '--lr', type=float, default=-1.,
                        help='Learning rate (or initial learning rate) of the stochastic gradient-based optimizer')
    parser.add_argument('--use-fixed-lr', '--ufl', action='store_true', default=False,
                        help="Option that disables the use of exponential learning rate schedule")
    # General constants
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help="Option that randomly shuffles the order of the test data. This is useful since the "
                             "decoder applies average power normalization, which is sensitive to the order of symbols.")
    parser.add_argument('--avg-power-symbols', '--aps', type=float, default=1.,
                        help='Average power of the symbols in the constellation (maximum value)')
    parser.add_argument('--l2-reg-strength', '--reg', type=float, default=0.,
                        help='Regularization strength (or coefficient) multiplying the regularization term for the '
                             'encoder layer activations in the loss function. Set to 0 and disabled by default.')
    parser.add_argument('--disable-scale-outputs', '--dso', action='store_true', default=False,
                        help="By default the channel outputs will be scaled by their average power before decoding. "
                             "Use this option to disable this scaling.")
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of parallel jobs/CPU cores available to use. Can be an upper bound.')
    args = parser.parse_args()

    args.channel_model_file = os.path.abspath(args.channel_model_file)
    args.autoencoder_model_file = os.path.abspath(args.autoencoder_model_file)
    if args.adaptation_params_file:
        args.adaptation_params_file = os.path.abspath(args.adaptation_params_file)

    args.output_dir = os.path.abspath(args.output_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dim_encoding % 2 == 1:
        raise ValueError("Encoding dimension {:d} not valid. Must be an even integer.".format(args.dim_encoding))

    config_basic = {
        'n_bits': args.n_bits,
        'dim_encoding': args.dim_encoding,
        'n_components': args.n_components,
        'n_hidden': args.n_hidden,
        'mod_order': 2 ** args.n_bits,
        'rate_comm': args.n_bits / args.dim_encoding,
        'type_autoencoder': args.type_autoencoder,
        # 'data_file_autoenc': os.path.join(args.data_dir, get_autoencoder_data_filename(args.n_bits)),
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs),
        'n_jobs': args.n_jobs
    }
    lr_adam = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_ADAM
    lr_sgd = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_SGD
    config_optimizer = {
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'optim_method': args.optim_method,
        'learning_rate_adam': lr_adam,
        'learning_rate_sgd': lr_sgd,
        'use_lr_schedule': (not args.use_fixed_lr)
    }
    return args, config_basic, config_optimizer


def wrapper_autoencoder_decode(autoencoder, y_test, labels_test):
    # Note that the function `helpers.calculate_metric_autoencoder` can be used decoding and BER calculation as well
    # Decode using the autoencoder given the samples and compare with labels.
    n_samples = len(y_test)
    if len(labels_test) != n_samples:
        raise ValueError("Invalid input length: len(y_test): %d, len(labels_test): %d" % (n_samples, len(labels_test)))

    # Not using a large batch size here since the scale factor at the decoder depends on the batch size
    batch_size = max(BATCH_SIZE_PRED, 2 * autoencoder.inputs_unique.shape[0])
    y_hat = autoencoder.decoder_predict(y_test, batch_size=batch_size)
    true = np.argmax(labels_test.numpy(), axis=1)
    pred = np.argmax(y_hat, axis=1)
    ber = np.mean(np.not_equal(true, pred))

    return ber


def qam_decode(y_test, labels_test, n_bits, x_unique):
    n_unique = x_unique.shape[0]
    p_avg_auto = tf.reduce_sum(x_unique ** 2) / n_unique
    # Set average power of the QAM constellation to the same as that of the autoencoder
    qam = StandardQAM(n_bits, avg_power=p_avg_auto)

    p_avg_qam = tf.reduce_sum(qam.constellation ** 2) / n_unique
    print("Average power of {:d}-QAM: {:.4f}. Average power of the autoencoder: {:.4f}".
          format(n_unique, p_avg_qam, p_avg_auto))
    # Prediction and BER
    batch_size = max(BATCH_SIZE_PRED, 2 * n_unique)
    pred = qam.decode(y_test, batch_size=batch_size)
    true = np.argmax(labels_test.numpy(), axis=1)
    ber = np.mean(np.not_equal(true, pred))

    # Set BER of exactly 0 to a small non-zero value (avoids taking log(0) in the BER plots)
    return max(ber, 0.1 / y_test.shape[0])


def load_test_data(sample_file, label_file, shuffle=False):
    labels = loadmat_helper(label_file)     # the sent one-hot-coded labels
    y_data = loadmat_helper(sample_file)    # the received modulated symbols (channel outputs)

    # turn data into real and imag.
    n_samp = len(y_data)
    y_data1 = np.zeros((n_samp, 2))
    y_data1[:, 0] = np.reshape(y_data.real, n_samp)
    y_data1[:, 1] = np.reshape(y_data.imag, n_samp)
    if shuffle:
        ind = np.random.permutation(n_samp)
        y_data1 = y_data1[ind, :]
        labels = labels[ind, :]

    return tf.convert_to_tensor(y_data1, dtype=DTYPE_TF), tf.convert_to_tensor(labels, dtype=DTYPE_TF)


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Create a new channel model and load the saved weights file
    mdn_model = initialize_MDN_model(
        config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
    )
    mdn_model.load_weights(args.channel_model_file).expect_partial()

    # Load the saved autoencoder model weights into a newly initialized autoencoder model
    n_train = config_basic['mod_order'] * 1000    # value is not important
    autoencoder = initialize_autoencoder(config_basic['type_autoencoder'], mdn_model, config_basic, config_optimizer,
                                         n_train, temperature=CONFIG_ANNEAL['temp_final'])
    autoencoder.load_weights(args.autoencoder_model_file).expect_partial()

    if os.path.isfile(args.adaptation_params_file):
        # Load the adaptation parameters if provided
        psi_values = tf.convert_to_tensor(np.load(args.adaptation_params_file), dtype=DTYPE_TF)
        # print("\nInitializing the adapted autoencoder from the saved adaptation parameters.")
        autoencoder = initialize_autoencoder_adapted(
            config_basic['type_autoencoder'], autoencoder.channel, autoencoder.encoder, autoencoder.decoder,
            psi_values, config_basic, temperature=CONFIG_ANNEAL['temp_final']
        )
    else:
        psi_values = None
        # print("\nEvaluating the autoencoder model without adaptation.")

    # Load the data to be decoded
    y_test, labels_test = load_test_data(os.path.join(args.data_dir, RX_DATA_BASENAME),
                                         os.path.join(args.data_dir, LABELS_BASENAME), shuffle=args.shuffle)
    print("\nNumber of samples for decoding: {:d}".format(y_test.shape[0]))

    # BER of standard M-QAM modulation is calculated as a baseline
    # x_unique = autoencoder.encoder(autoencoder.inputs_unique)
    # ber_qam = qam_decode(y_test, labels_test, config_basic['n_bits'], x_unique)

    # Decode the channel outputs and calculate BER
    t1 = time.time()
    ber = calculate_metric_autoencoder(autoencoder, labels_test, y_test, metric='error_rate')
    t_decode = time.time() - t1
    print("BER of autoencoder = {:.8f}".format(ber))
    print("Decoding finished in %.2f seconds" % t_decode)
    # Save the BER values
    # sio.savemat(os.path.join(args.output_dir, 'ber_qam.mat'), {'BER': ber_qam})
    if psi_values is None:
        sio.savemat(os.path.join(args.output_dir, 'ber.mat'), {'BER': ber})
    else:
        sio.savemat(os.path.join(args.output_dir, 'ber_adapted.mat'), {'BER': ber})


if __name__ == '__main__':
    main()
