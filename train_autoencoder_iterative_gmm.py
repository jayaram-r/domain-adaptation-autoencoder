"""
Main script for training the autoencoder and mixture density network (MDN) channel model. Similar to the script
`train_autoencoder.py`, but this one retrains the channel iteratively after every epoch of training the autoencoder.
New training data for the channel is generated based on the learned constellations of the autoencoder at the
end of every epoch, and this script supports a conditional Gaussian mixture (simulated) channel model.
The parameters of the Gaussian mixture channel model are specified as input via a Pickle file.
This script supports only the standard type of autoencoder training, and not the other variants such as MAP/MMSE symbol estimation autoencoder.

Main command line options:
--n-bits or --nb : number of bits per symbol (REQUIRED).
--params-gmm : path to the Pickle file with the Gaussian mixture parameters per symbol (REQUIRED).

--max-phase-shift or --mps : Maximum phase shift magnitude 's' in degrees. If 's > 0', then the phase shift is uniformly
                             distributed in [-s, s].
--output-dir or -o : path to the directory where output files, plots, and performance metrics will be saved.
--model-dir or -m : path to the directory where models files and checkpoints will be saved.
--data-dir or -d : path to the directory where the data files are saved and reloaded.
--dim-encoding or --de : the encoding dimension - 2, 4, etc.
--n-epochs or --ne : number of optimization epochs for training the autoencoder.
--channel-model-file or --cmf : path to the directory containing the trained channel model weights. Can be optionally
specified to initialize the current channel model training.
--autoencoder-model-file or --amf : path to the directory containing the trained autoencoder weights. Can be optionally
specified to initialize the current autoencoder model training.
--constellation-file or --cof : Path to the numpy file with the initial constellation symbols. If not specified, by
default, an M-QAM constellation is used.

Usage Example:
# Initialization for the MDN channel, autoencoder, and the constellation can be specified.
cmf='models_train/channel_model/channel'
amf='models_train/autoencoder/autoencoder'
cof='models_train/constellation_autoencoder.npy'

python train_autoencoder_iterative_gmm.py --n-bits 4 --params-gmm <path to GMM file> --de 2 --mps 10 --cmf $cmf --amf $amf --cof $cof

"""
import sys
import argparse
import os
import csv
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    load_data_transceiver,
    sample_batch_MDN,
    get_autoencoder_data_filename,
    generate_data_autoencoder,
    generate_data_gmm,
    check_weights_equality,
    get_noise_stddev,
    get_ber_channel_variations,
    save_perf_metrics,
    load_perf_metrics,
    plot_channel_data
)
from helpers.MDN_classes import (
    MDN_model,
    MDN_model_disc,
    initialize_MDN_model,
    initialize_MDN_model_disc,
    load_channel_model_from_file
)
from helpers.autoencoder_classes import (
    get_autoencoder_name,
    AutoencoderInverseAffine,
    initialize_autoencoder,
    load_autoencoder_from_file
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
    parser.add_argument('--params-gmm', required=True,
                        help='Path to the pickle file with the GMM parameters defining the channel model')
    # Saved files of the channel model, autoencoder, and the constellation can be optionally specified to initialize
    # the current training.
    parser.add_argument('--channel-model-file', '--cmf', default='',
                        help='Path to the directory containing the trained channel model weights. Can be optionally '
                             'specified to initialize the current channel model training.')
    parser.add_argument('--autoencoder-model-file', '--amf', default='',
                        help='Path to the directory containing the trained autoencoder model weights. Can be '
                             'optionally specified to initialize the current autoencoder training.')
    parser.add_argument('--constellation-file', '--cof', default='',
                        help='Path to the numpy file with the initial constellation symbols. If not specified, by '
                             'default, an M-QAM constellation is used.')
    parser.add_argument('--max-phase-shift', '--mps', type=float, default=0.,
                        help="Maximum phase shift magnitude 's' in degrees. If 's > 0', then the phase shift is "
                             "uniformly distributed in [-s, s]")
    parser.add_argument('--discrim-training-mdn', '--disc', action='store_true', default=False,
                        help="Use option to enable discriminative training of the MDN channel model. Deprecated - "
                             "does not have good empirical performance.")
    # Options specifying the simulated channel model
    parser.add_argument('--n-train-channel', '--ntrc', type=int, default=25000,
                        help='Number of samples for training the channel model')
    # Directories for the data, output, and model files
    parser.add_argument('--data-dir', '-d', default='./Data', help='Directory for the saved data files')
    parser.add_argument('--output-dir', '-o', default='./outputs_train',
                        help='Directory for saving the output files, plots etc')
    parser.add_argument('--model-dir', '-m', default='./models_train', help='Directory for saving the model files')
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
    parser.add_argument('--n-epochs', '--ne', type=int, default=100,
                        help='Number of optimization epochs for training the autoencoder')
    parser.add_argument('--optim-method', '--om', choices=['adam', 'sgd'], default='sgd',
                        help="Optimization method to use: 'adam' or 'sgd'")
    parser.add_argument('--learning-rate', '--lr', type=float, default=-1.,
                        help='Learning rate (or initial learning rate) of the stochastic gradient-based optimizer')
    parser.add_argument('--use-fixed-lr', '--ufl', action='store_true', default=False,
                        help="Option that disables the use of exponential learning rate schedule")
    # Number of training, validation, and test samples, specified per symbol
    parser.add_argument('--n-train-per-symbol', '--ntr', type=int, default=20000,
                        help='Number of training samples per constellation symbol')
    parser.add_argument('--n-test-per-symbol', '--nte', type=int, default=20000,
                        help='Number of test samples per constellation symbol')
    # General constants
    parser.add_argument('--avg-power-symbols', '--aps', type=float, default=1.,
                        help='Average power of the symbols in the constellation (maximum value)')
    parser.add_argument('--l2-reg-strength', '--reg', type=float, default=0.,
                        help='Regularization strength (or coefficient) multiplying the regularization term for the '
                             'encoder layer activations in the loss function. Set to 0 and disabled by default.')
    parser.add_argument('--disable-scale-outputs', '--dso', action='store_true', default=False,
                        help="By default the channel outputs will be scaled by their average power before decoding. "
                             "Use this option to disable this scaling.")
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    parser.add_argument('--skip-summary', action='store_true', default=False,
                        help='Use this option to skip the summary step (plots and output files) after training.')
    args = parser.parse_args()

    if args.channel_model_file:
        args.channel_model_file = os.path.abspath(args.channel_model_file)
        print("Initializing the channel model from the file: {}".format(args.channel_model_file))
    if args.autoencoder_model_file:
        args.autoencoder_model_file = os.path.abspath(args.autoencoder_model_file)
        print("Initializing the autoencoder model from the file: {}".format(args.autoencoder_model_file))
    if args.constellation_file:
        args.constellation_file = os.path.abspath(args.constellation_file)
        print("Initializing the constellation from the file: {}".format(args.constellation_file))

    args.output_dir = os.path.abspath(args.output_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)

    if args.dim_encoding % 2 == 1:
        raise ValueError("Encoding dimension {:d} not valid. Must be an even integer.".format(args.dim_encoding))

    config_basic = {
        'n_bits': args.n_bits,
        'dim_encoding': args.dim_encoding,
        'n_components': args.n_components,
        'n_hidden': args.n_hidden,
        'mod_order': 2 ** args.n_bits,
        'rate_comm': args.n_bits / args.dim_encoding,
        'type_autoencoder': 'standard',
        'discrim_training_mdn': args.discrim_training_mdn,
        'data_file_autoenc': os.path.join(args.data_dir, get_autoencoder_data_filename(args.n_bits)),
        'n_train_channel': args.n_train_channel,
        'max_phase_shift': args.max_phase_shift,
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs)
    }
    lr_adam = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_ADAM
    lr_sgd = args.learning_rate if (args.learning_rate > 0.) else LEARNING_RATE_SGD
    config_optimizer = {
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'n_epochs_ch': args.n_epochs_ch,
        'optim_method': args.optim_method,
        'learning_rate_adam': lr_adam,
        'learning_rate_sgd': lr_sgd,
        'use_lr_schedule': (not args.use_fixed_lr),
        'anneal': True
    }
    return args, config_basic, config_optimizer


def wrapper_train_autoencoder_and_channel(params_gmm, x_train, x_val, mdn_model, config, config_optimizer, model_dir,
                                          modelfile_init=''):
    print("\nTraining the autoencoder and channel models iteratively:")
    type_autoencoder = 'standard'
    n_train = x_train.shape[0]
    # Directory to save the model checkpoints
    model_ckpt_path = os.path.join(model_dir, 'model_ckpts')
    model_ckpt_filename = os.path.join(model_ckpt_path, 'autoencoder_{}'.format(type_autoencoder))
    if not os.path.isdir(model_ckpt_path):
        os.makedirs(model_ckpt_path)

    # Initialize and compile the autoencoder
    autoencoder = initialize_autoencoder(type_autoencoder, mdn_model, config, config_optimizer, n_train)
    if modelfile_init:
        # Initialize the autoencoder weights from the initialization file
        autoencoder.load_weights(modelfile_init).expect_partial()

    batch_size_ch = min(int(config['n_train_channel'] / 2.), config_optimizer['batch_size'])
    train_loss = []
    val_loss = []
    min_val_loss = sys.float_info.max
    best_epoch = 0
    for n in range(1, config_optimizer['n_epochs'] + 1):
        # Train the autoencoder with the channel model fixed
        print("\nAutoencoder training, epoch {:d}:".format(n))
        hist = autoencoder.fit(
            x=x_train, y=x_train, epochs=1, batch_size=config_optimizer['batch_size'], validation_data=(x_val, x_val)
        )
        train_loss.append(hist.history['loss'][0])
        val_loss.append(hist.history['val_loss'][0])
        if val_loss[-1] < min_val_loss:
            min_val_loss = val_loss[-1]
            best_epoch = n
            autoencoder.save_weights(model_ckpt_filename)  # save the weights of the current best solution

        if n == config_optimizer['n_epochs']:
            break

        # Generate data for training the channel model with the updated autoencoder constellation
        print("\nChannel training with the updated constellation:")
        constellation = autoencoder.encoder(autoencoder.inputs_unique).numpy()
        x_data_ch, y_data_ch = generate_data_wrapper(params_gmm, constellation, config['x_unique_init'],
                                                     config['n_train_channel'], config['max_phase_shift'])
        # Train the channel model
        mdn_model = initialize_mdn_between_epochs(mdn_model, constellation, x_data_ch, y_data_ch,
                                                  batch_size_ch, config)
        _ = mdn_model.fit(x=x_data_ch, y=y_data_ch, batch_size=batch_size_ch,
                          epochs=config_optimizer['n_epochs_ch'], validation_split=0.1)
        # Update the channel model of the autoencoder
        autoencoder.channel.set_weights(mdn_model.get_weights())

    print("\nUsing the saved autoencoder weights from epoch {:d} (corresponds to minimum validation loss).".
          format(best_epoch))
    autoencoder.load_weights(model_ckpt_filename).expect_partial()

    # Save the autoencoder constellation to a file
    constellation = autoencoder.encoder(autoencoder.inputs_unique).numpy()
    fname = os.path.join(model_dir, CONSTELLATION_BASENAME)
    with open(fname, 'wb') as fp:
        np.save(fp, constellation)

    # Save the autoencoder constellation and the corresponding one-hot-coded labels
    symbol_dic = {"SYMBOLS": constellation}
    sio.savemat(os.path.join(model_dir, 'symbols.mat'), symbol_dic)
    label_dic = {"Labels": autoencoder.inputs_unique.numpy()}
    sio.savemat(os.path.join(model_dir, 'labels.mat'), label_dic)

    # Save the channel model corresponding to the best epoch (from the autoencoder training) to a file
    mdn_model = autoencoder.channel
    fname = os.path.join(model_dir, 'channel_model', 'channel')
    mdn_model.save_weights(fname)   # save only the model weights
    # mdn_model.save(fname, include_optimizer=True)   # save the entire model

    # Save the final autoencoder model to a file
    fname = os.path.join(model_dir, get_autoencoder_name(autoencoder), 'autoencoder')
    autoencoder.save_weights(fname)     # save only the model weights
    # autoencoder.save(fname, include_optimizer=True)     # save the entire model

    return autoencoder, constellation, mdn_model, np.array(train_loss), np.array(val_loss)


def initialize_mdn_between_epochs(mdn_model_curr, constellation, x_data_ch, y_data_ch, batch_size, config):
    '''
    Initialize the MDN model suitably between autoencoder training epochs. Evaluate the loss function of the MDN
    with the current best weights and randomly initialized weights, and select the one corresponding to a lower loss
    function.
    '''
    # Initialize a new MDN model with the same weights as the current MDN model
    if config['discrim_training_mdn']:
        mdn_model1 = initialize_MDN_model_disc(
            config['n_components'], config['dim_encoding'], config['n_hidden'], constellation
        )
    else:
        mdn_model1 = initialize_MDN_model(config['n_components'], config['dim_encoding'], config['n_hidden'])

    mdn_model1.set_weights(mdn_model_curr.get_weights())
    loss1 = mdn_model1.evaluate(x=x_data_ch, y=y_data_ch, batch_size=batch_size, verbose=0)

    # Initialize a new MDN model with random initial weights
    if config['discrim_training_mdn']:
        mdn_model2 = initialize_MDN_model_disc(
            config['n_components'], config['dim_encoding'], config['n_hidden'], constellation
        )
    else:
        mdn_model2 = initialize_MDN_model(config['n_components'], config['dim_encoding'], config['n_hidden'])

    loss2 = mdn_model2.evaluate(x=x_data_ch, y=y_data_ch, batch_size=batch_size, verbose=0)
    print("Initializing the MDN between autoencoder training epochs: loss function with current weights = {:.4f}, "
          "loss function with random initialization = {:.4f}".format(loss1, loss2))
    # Choose the model with smaller loss function
    if loss1 <= loss2:
        return mdn_model1
    else:
        return mdn_model2


def summarize_autoencoder_training(autoencoder, constellation, params_gmm, train_loss, val_loss, config, output_dir,
                                   n_test=300000):
    basename = 'autoencoder'
    # basename = get_autoencoder_name(autoencoder)
    # Plot the learning curves
    fig = plt.figure()
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, color='r', marker='.', label='training loss')
    plt.plot(np.arange(1, len(val_loss) + 1), val_loss, color='b', marker='.', label='validation loss')
    # plt.title('Autoencoder learning', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(loc='upper right')
    plot_filename = os.path.join(output_dir, 'learning_curves_{}.png'.format(basename))
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)

    # Distinct symbols learned by the autoencoder, their average power, and prior probabilities
    config['x_unique_auto'] = constellation
    config['prior_prob_x'] = autoencoder.prior_prob_inputs
    config['p_avg_auto'] = np.sum(constellation ** 2) / constellation.shape[0]
    print("Average power of the symbols learned by the autoencoder: {:.4f}".format(config['p_avg_auto']))

    # Generate a large test set with the final autoencoder constellation and calculate the BER
    constellation_init = config['x_unique_init']
    n_symb = constellation_init.shape[0]
    params_gmm_new = []
    for i in range(n_symb):
        tmp_dict = {k: v for k, v in params_gmm[i].items()}
        tmp_dict['means'] = params_gmm[i]['means'] - constellation_init[i, :] + constellation[i, :]
        params_gmm_new.append(tmp_dict)

    x_data, y_data, labels, _ = generate_data_gmm(params_gmm_new, constellation, n_test,
                                                  max_phase_shift=config['max_phase_shift'])
    labels_one_hot = tf.one_hot(labels.ravel(), n_symb).numpy()
    ber = calculate_metric_autoencoder(autoencoder, labels_one_hot, y_data, metric='error_rate')
    print("\nBER of the autoencoder = {:.6f}".format(ber))
    sio.savemat(os.path.join(output_dir, 'ber.mat'), {'BER': ber})

    if config['dim_encoding'] == 2:
        # Plot the constellation learned by the autoencoder
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(config['x_unique_auto'][:, 0], config['x_unique_auto'][:, 1], alpha=1, c='k', marker='^')
        ax.scatter(config['x_unique_init'][:, 0], config['x_unique_init'][:, 1], alpha=1, c='r', marker='o')
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.title("Final learned constellation: black triangles.\n"
                  "Initial constellation: red circles.", fontsize=9)
        plot_filename = os.path.join(output_dir, 'constellation_{}.png'.format(basename))
        fig.tight_layout()
        fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
        plt.close(fig)


def wrapper_train_channel(x_data, y_data, config, modelfile_init='', batch_size=128, n_epochs=100, verbose=False):
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
        mdn_model = initialize_MDN_model(config['n_components'], config['dim_encoding'], config['n_hidden'])

    if modelfile_init:
        # Initialize the channel model weights from the initialization file
        mdn_model.load_weights(modelfile_init).expect_partial()

    batch_size = min(int(y_data.shape[0] / 2.), batch_size)
    hist = mdn_model.fit(x=x_data, y=y_data, batch_size=batch_size, epochs=n_epochs, validation_split=0.1)
    if verbose:
        print('')
        mdn_model.summary()

    return mdn_model, np.array(hist.history['loss']), np.array(hist.history['val_loss'])


def summarize_channel_training(mdn_model, train_loss, val_loss, constellation, params_gmm, config, output_dir):
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

    # Generate data from the Gaussian mixture channel
    x_data, y_data = generate_data_wrapper(params_gmm, constellation, config['x_unique_init'],
                                           config['n_train_channel'], config['max_phase_shift'])
    # Samples generated from the channel model
    params = mdn_model.predict(x_data)
    y_samples = sample_batch_MDN(params, config['n_components'], config['dim_encoding'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_data[:, 0], y_data[:, 1], alpha=0.4, c='grey')    # actual channel outputs
    ax.scatter(y_samples[:, 0], y_samples[:, 1], alpha=0.4, c='lightseagreen')  # synthetic channel outputs
    ax.scatter(x_data[:, 0], x_data[:, 1], alpha=0.4, c='k')  # channel inputs
    plt.title("Simulated channel outputs from the MDN in sea-green.\n"
              "Actual channel outputs in gray. Channel inputs in black", fontsize=9)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plot_filename = os.path.join(output_dir, 'plot_channel_model.png')
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def generate_data_wrapper(params_gmm, constellation, constellation_init, n_samp, max_phase_shift):
    # For the Gaussian mixture corresponding to each symbol in the constellation, subtract off the old symbol and
    # add the new symbol to the component means. The component priors and covariances are kept the same.
    params_gmm_new = []
    for i in range(constellation_init.shape[0]):
        tmp_dict = {k: v for k, v in params_gmm[i].items()}
        tmp_dict['means'] = params_gmm[i]['means'] - constellation_init[i, :] + constellation[i, :]
        params_gmm_new.append(tmp_dict)

    x_data, y_data, _, _ = generate_data_gmm(params_gmm_new, constellation, n_samp, max_phase_shift=max_phase_shift)

    return tf.convert_to_tensor(x_data, dtype=DTYPE_TF), tf.convert_to_tensor(y_data, dtype=DTYPE_TF)


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Data for training the autoencoder
    x_train, x_val, x_test, labels_test = generate_data_autoencoder(
        config_basic['data_file_autoenc'], config_basic['mod_order'], args.n_train_per_symbol, args.n_test_per_symbol
    )
    # Data for training the initial channel model
    if args.constellation_file:
        constellation_init = np.load(args.constellation_file)
    else:
        # Constellation of standard M-QAM
        constellation_init = StandardQAM(config_basic['n_bits'],
                                         avg_power=config_basic['avg_power_symbols']).constellation

    config_basic['x_unique_init'] = constellation_init
    # Load the GMM parameters file
    with open(args.params_gmm, 'rb') as fp:
        params_gmm = pickle.load(fp)

    # Generate channel data from the GMM model based on the initial constellation
    x_data, y_data, _, _ = generate_data_gmm(
        params_gmm, constellation_init, args.n_train_channel, max_phase_shift=args.max_phase_shift
    )
    x_data = tf.convert_to_tensor(x_data, dtype=DTYPE_TF)
    y_data = tf.convert_to_tensor(y_data, dtype=DTYPE_TF)

    # Plot the initial channel data
    if config_basic['dim_encoding'] == 2:
        plot_filename = os.path.join(args.output_dir, 'plot_channel_data_init.png')
        plot_channel_data(x_data, y_data, plot_filename)

    time_log = dict()
    # Train the initial channel MDN model
    t1 = time.time()
    mdn_model, train_loss_ch, val_loss_ch = wrapper_train_channel(
        x_data, y_data, config_basic, modelfile_init=args.channel_model_file,
        batch_size=config_optimizer['batch_size'], n_epochs=config_optimizer['n_epochs_ch'], verbose=True
    )
    t2 = time.time()
    time_log['channel_train'] = t2 - t1
    print("\nTime taken for training the initial channel model: {:g} seconds".format(t2 - t1))

    # Train the autoencoder and channel models iteratively
    t1 = time.time()
    autoencoder, constellation, mdn_model, train_loss, val_loss = wrapper_train_autoencoder_and_channel(
        params_gmm, x_train, x_val, mdn_model, config_basic, config_optimizer, args.model_dir,
        modelfile_init=args.autoencoder_model_file
    )
    t2 = time.time()
    time_log['autoencoder_train'] = t2 - t1
    print("\nTime taken for iteratively training the autoencoder and channel models: {:g} minutes".
          format((t2 - t1) / 60.))
    if not args.skip_summary:
        # Summarize the channel model training
        summarize_channel_training(mdn_model, train_loss_ch, val_loss_ch, constellation, params_gmm, config_basic,
                                   args.output_dir)
        # Calculate performance and summarize the autoencoder training
        summarize_autoencoder_training(autoencoder, constellation, params_gmm, train_loss, val_loss,
                                       config_basic, args.output_dir)

    # Log the training times to a file
    fname = os.path.join(args.output_dir, 'time_training.csv')
    with open(fname, 'w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(['channel_train', 'autoencoder_train'])
        cw.writerow(['{:.4f}'.format(time_log['channel_train']), '{:.4f}'.format(time_log['autoencoder_train'])])


if __name__ == '__main__':
    main()
