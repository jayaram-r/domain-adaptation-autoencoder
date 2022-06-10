"""
Main script for training a mixture density network (MDN) channel model.
For a description of the command line inputs, type:
`python train_mdn_channel.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED)
--model-dir or -m: path to the directory where models files and checkpoints will be saved.
--output-dir or -o: path to the directory where output files, plots, and performance metrics will be saved.
--dim-encoding or --de: the encoding dimension - 2, 4, etc.
--simulated-channel or --sim: specifies the use of simulated channel data.
--type-channel or --tc: specifies the type of simulated channel model to use.
--SNR-channel: specifies the SNR of the simulated channel data in dB.
--channel-model-file or --cmf: Path to the directory containing the trained channel model weights. Can be optionally
                               specified to initialize the current channel model training.
--constellation-file or --cof: Path to the numpy file with the initial constellation symbols. If not specified, by
                               default, an M-QAM constellation is used.

Usage Examples:
    1. Train on simulated data from an AWGN channel with the default SNR of 14dB
    python train_mdn_channel.py --n-bits 4 --de 2 --sim --tc AWGN

    2. Train on simulated data from Ricean fading channel with an SNR of 18dB
    python train_mdn_channel.py --n-bits 4 --de 2 --sim --tc fading_ricean --SNR-channel 18

    3. Same as (2), but the options --cmf and --cof are used to initialize the channel model and the initial
    constellation from saved files.
    cmf='models_train/channel_model/channel'
    cof='models_train/constellation_autoencoder.npy'
    python train_mdn_channel.py --n-bits 4 --de 2 --sim --tc fading_ricean --SNR-channel 18 --cmf $cmf --cof $cof

    4. Train on channel data loaded from files. Specify directories to save the output and model files.
    python train_mdn_channel.py --n-bits 4 --de 2 --tx-data-file <path to tx data file>
    --rx-data-file <path to rx data file> -o <output directory path> -m <models directory path>

    Remember to specify different output and model directories for each experiment so that the results are not overwritten.
"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    sample_batch_MDN,
    check_weights_equality,
    generate_channel_data_real,
    generate_channel_data_simulated,
    plot_channel_data
)
from helpers.MDN_classes import (
    MDN_model,
    MDN_model_disc,
    initialize_MDN_model,
    initialize_MDN_model_disc,
    load_channel_model_from_file
)
from helpers.standard_modulations import StandardQAM
from helpers.constants import *
# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_inputs():
    parser = argparse.ArgumentParser()
    # Required argument
    parser.add_argument('--n-bits', '--nb', type=int, required=True, help='number of bits per symbol')
    # Channel data files - required if not using a simulated channel model
    parser.add_argument('--tx-data-file', '--tdf', default='',
                        help='Path to the file containing transmitted channel data. Not required for simulated '
                             'channel variations.')
    parser.add_argument('--rx-data-file', '--rdf', default='',
                        help='Path to the file containing received channel data. Not required for simulated '
                             'channel variations.')
    # Saved files of the channel model and the constellation can be optionally specified to initialize
    # the current training.
    parser.add_argument('--channel-model-file', '--cmf', default='',
                        help='Path to the directory containing the trained channel model weights. Can be optionally '
                             'specified to initialize the current channel model training.')
    parser.add_argument('--constellation-file', '--cof', default='',
                        help='Path to the numpy file with the initial constellation symbols. If not specified, by '
                             'default, an M-QAM constellation is used.')
    parser.add_argument('--discrim-training-mdn', '--disc', action='store_true', default=False,
                        help="Use option to enable discriminative training of the MDN channel model. Deprecated - "
                             "does not have good empirical performance.")
    # Options to use a simulated channel model
    parser.add_argument('--simulate-channel', '--sim', action='store_true', default=False,
                        help="Use option to simulate the channel variations according to a standard model, e.g. AWGN "
                             "or Ricean fading. Option '--type-channel' specifies the type of channel model.")
    parser.add_argument('--type-channel', '--tc', choices=['AWGN', 'fading', 'fading_ricean', 'fading_rayleigh'],
                        default='fading_ricean', help='Type of simulated channel model to use')
    parser.add_argument('--SNR-channel', type=float, default=14.0, help='SNR of the simulated channel model in dB.')
    parser.add_argument('--sigma-noise-measurement', '--sigma', type=float, default=0.09,
                        help='Noise standard deviation of the channel. This is estimated from the channel data in '
                             'case the data files are provided.')
    parser.add_argument('--n-train-channel', '--ntrc', type=int, default=25000,
                        help='Number of samples for training the channel model')
    # Directories for the output, and model files
    parser.add_argument('--output-dir', '-o', default='./outputs_mdn_train',
                        help='Directory for saving the output files, plots etc')
    parser.add_argument('--model-dir', '-m', default='./models_mdn', help='Directory for saving the model files')
    # Model hyper-parameters
    parser.add_argument('--dim-encoding', '--de', type=int, default=2,
                        help='Dimension of the encoded symbols. Should be an even integer')
    parser.add_argument('--n-components', '--nc', type=int, default=5,
                        help='Number of components in the Gaussian mixture density network')
    parser.add_argument('--n-hidden', '--nh', type=int, default=100, help='Size of the hidden fully-connected layer')
    # Optimizer configuration
    parser.add_argument('--batch-size', '--bs', type=int, default=128, help='Batch size for optimization')
    parser.add_argument('--n-epochs-ch', '--nec', type=int, default=100,
                        help='Number of optimization epochs for training the channel model')
    parser.add_argument('--optim-method', '--om', choices=['adam', 'sgd'], default='sgd',
                        help="Optimization method to use: 'adam' or 'sgd'")
    parser.add_argument('--learning-rate', '--lr', type=float, default=-1.,
                        help='Learning rate (or initial learning rate) of the stochastic gradient-based optimizer')
    parser.add_argument('--use-fixed-lr', '--ufl', action='store_true', default=False,
                        help="Option that disables the use of exponential learning rate schedule")
    # General constants
    parser.add_argument('--avg-power-symbols', '--aps', type=float, default=1.,
                        help='Average power of the symbols in the constellation (maximum value)')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of parallel jobs/CPU cores available to use. Can be an upper bound.')
    parser.add_argument('--skip-summary', action='store_true', default=False,
                        help='Use this option to skip the summary step (plots and output files) after training.')
    args = parser.parse_args()

    if not args.simulate_channel:
        if (not os.path.isfile(args.tx_data_file)) or (not os.path.isfile(args.rx_data_file)):
            raise ValueError("Did not receive valid channel data files.")

        args.tx_data_file = os.path.abspath(args.tx_data_file)
        args.rx_data_file = os.path.abspath(args.rx_data_file)
    else:
        print("\nUsing simulated channel data of type '{}'".format(args.type_channel))

    if args.channel_model_file:
        args.channel_model_file = os.path.abspath(args.channel_model_file)
        print("Initializing the channel model from the file: {}".format(args.channel_model_file))
    if args.constellation_file:
        args.constellation_file = os.path.abspath(args.constellation_file)
        print("Initializing the constellation from the file: {}".format(args.constellation_file))

    args.output_dir = os.path.abspath(args.output_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    if args.dim_encoding % 2 == 1:
        raise ValueError("Encoding dimension {:d} not valid. Must be an even integer.".format(args.dim_encoding))

    config_basic = {
        'n_bits': args.n_bits,
        'dim_encoding': args.dim_encoding,
        'n_components': args.n_components,
        'n_hidden': args.n_hidden,
        'mod_order': 2 ** args.n_bits,
        'rate_comm': args.n_bits / args.dim_encoding,
        'discrim_training_mdn': args.discrim_training_mdn,
        'type_channel': args.type_channel,
        'SNR_channel_dB': args.SNR_channel,
        'sigma_noise_measurement': args.sigma_noise_measurement,
        'avg_power_symbols': args.avg_power_symbols,
        'EbNodB_range': SNR_RANGE_DEF,
        'EbNodB_min': SNR_MIN_RICEAN,
        'EbNo_min': 10. ** (SNR_MIN_RICEAN / 10.),
        'n_jobs': args.n_jobs
    }
    lr_adam = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_ADAM
    lr_sgd = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_SGD
    config_optimizer = {
        'batch_size': args.batch_size,
        'n_epochs_ch': args.n_epochs_ch,
        'optim_method': args.optim_method,
        'learning_rate_adam': lr_adam,
        'learning_rate_sgd': lr_sgd,
        'use_lr_schedule': (not args.use_fixed_lr)
    }
    return args, config_basic, config_optimizer


def wrapper_train_channel(x_data, y_data, config, model_dir, modelfile_init='', batch_size=128,
                          n_epochs=100, verbose=False):
    if verbose:
        if config['discrim_training_mdn']:
            print("\nTraining the MDN channel model. Loss function: posterior log-likelihood:")
        else:
            print("\nTraining the MDN channel model. Loss function: conditional log-likelihood:")

    # Initialize, compile, and train the MDN channel model
    if config['discrim_training_mdn']:
        mdn_model = initialize_MDN_model_disc(
            config['n_components'], config['dim_encoding'], config['n_hidden'], config['x_unique_init']
        )
    else:
        mdn_model = initialize_MDN_model(
            config['n_components'], config['dim_encoding'], config['n_hidden']
        )

    if modelfile_init:
        # Initialize the channel model weights from the initialization file
        mdn_model.load_weights(modelfile_init).expect_partial()

    batch_size = min(int(y_data.shape[0] / 2.), batch_size)
    hist = mdn_model.fit(x=x_data, y=y_data, batch_size=batch_size, epochs=n_epochs, validation_split=0.1)
    if verbose:
        print('')
        mdn_model.summary()

    # Save the channel model to a file
    fname = os.path.join(model_dir, 'channel_model', 'channel')
    mdn_model.save_weights(fname)   # save the model weights

    '''
    # Use the channel model from the saved file instead of the original model
    wts_orig = mdn_model.get_weights()
    # Load the saved model weights into a newly initialized model
    if config['discrim_training_mdn']:
        mdn_model = initialize_MDN_model_disc(
            config['n_components'], config['dim_encoding'], config['n_hidden'], config['x_unique_init']
        )
    else:
        mdn_model = initialize_MDN_model(
            config['n_components'], config['dim_encoding'], config['n_hidden']
        )

    stat = mdn_model.load_weights(fname).expect_partial()
    # Sanity check to ensure that the loaded model has the same weights as the original model
    if not check_weights_equality(mdn_model.get_weights(), wts_orig):
        print("ERROR: weights of the channel model loaded from the file are not the same.")
    '''
    return mdn_model, np.array(hist.history['loss']), np.array(hist.history['val_loss'])


def summarize_channel_training(mdn_model, train_loss, val_loss, x_data, y_data, config, output_dir):
    # Plot the learning curves
    fig = plt.figure()
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color='r', marker='.', label='training loss')
    plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color='b', marker='.', label='validation loss')
    # plt.title('Learning curves', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='upper right')
    plot_filename = os.path.join(output_dir, 'learning_curves_channel_training.png')
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)

    # Scatter plot of the sample data generated by the channel model
    if config['dim_encoding'] > 2:
        return

    params = mdn_model.predict(x_data)
    y_samples = sample_batch_MDN(params, config['n_components'], config['dim_encoding'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_data[:, 0], y_data[:, 1], alpha=0.4, c='grey')    # actual channel outputs
    ax.scatter(y_samples[:, 0], y_samples[:, 1], alpha=0.4, c='lightseagreen')  # synthetic channel outputs
    ax.scatter(x_data[:, 0], x_data[:, 1], alpha=0.4, c='k')    # channel inputs
    plt.title("Simulated channel outputs using the MDN in sea-green.\n"
              "Actual channel outputs in gray. Channel inputs in black", fontsize=9)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plot_filename = os.path.join(output_dir, 'plot_channel_model.png')
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Data for training the channel model
    if args.simulate_channel:
        if args.constellation_file:
            constellation_init = np.load(args.constellation_file)
        else:
            # Constellation of standard M-QAM
            constellation_init = StandardQAM(config_basic['n_bits'],
                                             avg_power=config_basic['avg_power_symbols']).constellation
        x_data, y_data, x_unique = generate_channel_data_simulated(
            args.type_channel, config_basic['SNR_channel_dB'], args.n_train_channel, config_basic, constellation_init
        )
    else:
        x_data, y_data, x_unique = generate_channel_data_real(args.tx_data_file, args.rx_data_file, config_basic)
        args.sigma_noise_measurement = config_basic['sigma_noise_measurement']

    # Save the constellation to a numpy file
    fname = os.path.join(args.model_dir, 'constellation_init.npy')
    with open(fname, 'wb') as fp:
        np.save(fp, x_unique.numpy())

    config_basic['x_unique_init'] = x_unique
    print("Measurement noise standard deviation: {:.6f}".format(config_basic['sigma_noise_measurement']))
    # Train the MDN model
    t1 = time.time()
    mdn_model, train_loss_ch, val_loss_ch = wrapper_train_channel(
        x_data, y_data, config_basic, args.model_dir, modelfile_init=args.channel_model_file,
        batch_size=config_optimizer['batch_size'], n_epochs=config_optimizer['n_epochs_ch'], verbose=True
    )
    t2 = time.time()
    print("\nTime taken for training the MDN: {:g} seconds".format(t2 - t1))
    if not args.skip_summary:
        # Plot the channel data
        if config_basic['dim_encoding'] == 2:
            plot_filename = os.path.join(args.output_dir, 'plot_channel_data.png')
            plot_channel_data(x_data, y_data, plot_filename)

        # Generate plots to summarize the MDN training
        summarize_channel_training(mdn_model, train_loss_ch, val_loss_ch, x_data, y_data, config_basic, args.output_dir)


if __name__ == '__main__':
    main()
