"""
Main script for the baseline method of fine-tuning the channel model and autoencoder on real samples measured by FPGA.
For a description of the command line inputs, type:
`python finetune_autoencoder.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--autoencoder-model-file or --amf: path to the autoencoder model file or directory (REQUIRED).
--last :  use this  option to specify that only  the last layer of MDN should be finetuned.
--model-dir or -m: path to the model directory where the adaptation parameters file should be saved.
--data-dir or -d: path to the data directory where the adaptation data files are read from.

Most of the other options can be left at their default values unless there is a specific need to change them.
Any non-default options that were used for training the autoencoder using `train_autoencoder.py` should also be set in
this script (if applicable). For example, if the dimension of the encoding is 4 and the number of components in the
channel MDN is 3, then the following options should be specified in both the training and adaptation scripts:
`--de 4 --nc 3`

DATA FILES:
Before running, ensure that the data files for adaptation are available in the data directory specified by the
command line option `--data-dir`. By default this is a directory `./data_adapt` under the current working directory.
Specifically, the following two mat files are required:
- `tx_symbols.mat`: File containing the transmitted (modulated) symbols.
- `rx_symbols.mat`: File containing the received symbols.

USAGE EXAMPLES:
    Assume that the script `train_autoencoder.py` has been run on data from FPGA, and that the channel model
    and autoencoder models are saved in the following directories:
    model_dir="${PWD}/models_train"
    cmf="${model_dir}/channel_model/channel"
    amf="${model_dir}/autoencoder/autoencoder"
    data_dir="${PWD}/data_adapt"
    # Directory for saving the adapted model can be the same as `model_dir`

    python finetune_autoencoder.py --n-bits 4 --cmf $cmf --amf $amf -m $model_dir -d $data_dir

    After running, check for the adapted MDN model and autoencoder under $model_dir.
"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scipy.spatial.distance import cdist
import scipy.io as sio
import multiprocessing
from functools import partial

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    load_data_transceiver,
    generate_data_autoencoder,
    get_autoencoder_data_filename,
    check_weights_equality,
    save_perf_metrics,
    load_perf_metrics,
    get_num_jobs,
    get_labels_from_symbols,
    sample_batch_MDN
)
from helpers.MDN_classes import (
    MDN_model,
    initialize_MDN_model
)
from helpers.autoencoder_classes import (
    get_autoencoder_name,
    initialize_autoencoder,
    initialize_autoencoder_adapted
)
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
    parser.add_argument('--last', action='store_true', default=False,
                        help="Use option to specify that only the last layer of the MDN should be finetuned.")
    # Data and model directory
    parser.add_argument('--data-dir', '-d', default='./data_adapt', help='Directory for the adaptation data files')
    parser.add_argument('--model-dir', '-m', default='./models_adapt', help='Directory for saving the model files')
    # Model hyper-parameters
    parser.add_argument('--dim-encoding', '--de', type=int, default=2,
                        help='Dimension of the encoded symbols. Should be an even integer')
    parser.add_argument('--n-components', '--nc', type=int, default=5,
                        help='Number of components in the Gaussian mixture density network')
    parser.add_argument('--n-hidden', '--nh', type=int, default=100, help='Size of the hidden fully-connected layer')
    # Optimizer configuration
    parser.add_argument('--batch-size', '--bs', type=int, default=128, help='Batch size for optimization')
    parser.add_argument('--n-epochs', '--ne', type=int, default=20, help='Number of optimization epochs')
    parser.add_argument('--optim-method', '--om', choices=['adam', 'sgd'], default='sgd',
                        help="Optimization method to use: 'adam' or 'sgd'")
    parser.add_argument('--learning-rate', '--lr', type=float, default=-1.,
                        help='Learning rate (or initial learning rate) of the stochastic gradient-based optimizer')
    parser.add_argument('--use-fixed-lr', '--ufl', action='store_true', default=False,
                        help="Option that disables the use of exponential learning rate schedule")
    # Number of training and test samples, specified per symbol
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
    args = parser.parse_args()

    args.channel_model_file = os.path.abspath(args.channel_model_file)
    args.autoencoder_model_file = os.path.abspath(args.autoencoder_model_file)
    args.model_dir = os.path.abspath(args.model_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    if args.dim_encoding % 2 == 1:
        raise ValueError("Encoding dimension {:d} not valid. Must be an even integer.".format(args.dim_encoding))

    data_dir_temp = os.path.abspath('./Data')
    if not os.path.isdir(data_dir_temp):
        os.makedirs(data_dir_temp)

    config_basic = {
        'n_bits': args.n_bits,
        'dim_encoding': args.dim_encoding,
        'n_components': args.n_components,
        'n_hidden': args.n_hidden,
        'mod_order': 2 ** args.n_bits,
        'rate_comm': args.n_bits / args.dim_encoding,
        'type_autoencoder': 'standard',
        'data_file_autoenc': os.path.join(data_dir_temp, get_autoencoder_data_filename(args.n_bits)),
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs),
        'n_jobs': 1
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


def wrapper_train_autoencoder(autoencoder, x_train, x_val, config_optimizer, model_dir, verbose=0):
    print("\nFine-tuning the autoencoder model with fixed MDN and encoder:")
    print("Size of the training set: {:d}".format(x_train.shape[0]))
    type_autoencoder = 'standard'
    # Directory to save the model checkpoints
    model_ckpt_path = os.path.join(model_dir, 'model_ckpts')
    model_ckpt_filename = os.path.join(model_ckpt_path, 'autoencoder_{}'.format(type_autoencoder))
    if not os.path.isdir(model_ckpt_path):
        os.makedirs(model_ckpt_path)

    wts_encoder = autoencoder.encoder.get_weights()
    wts_mdn = autoencoder.channel.get_weights()

    # Define callbacks
    # monitor='val_loss', mode='min'
    # monitor='val_categorical_accuracy', mode='max'
    mc = ModelCheckpoint(filepath=model_ckpt_filename, save_weights_only=True, monitor='val_loss',
                         mode='min', verbose=verbose, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=20)
    # Train the autoencoder
    hist = autoencoder.fit(
        x=x_train, y=x_train, epochs=config_optimizer['n_epochs'], batch_size=config_optimizer['batch_size'],
        validation_data=(x_val, x_val), callbacks=[mc]
    )
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    # Load the weights corresponding to the best epoch from the saved checkpoint
    autoencoder.load_weights(model_ckpt_filename).expect_partial()

    # Save the autoencoder constellation to a file
    unique_x = autoencoder.encoder(autoencoder.inputs_unique).numpy()
    fname = os.path.join(model_dir, CONSTELLATION_BASENAME)
    with open(fname, 'wb') as fp:
        np.save(fp, unique_x)

    # Save the autoencoder constellation and the corresponding one-hot-coded labels
    symbol_dic = {"SYMBOLS": unique_x}
    sio.savemat(os.path.join(model_dir, 'symbols.mat'), symbol_dic)
    label_dic = {"Labels": autoencoder.inputs_unique.numpy()}
    sio.savemat(os.path.join(model_dir, 'labels.mat'), label_dic)

    # Save the autoencoder model to a file
    fname = os.path.join(model_dir, get_autoencoder_name(autoencoder), 'autoencoder')
    autoencoder.save_weights(fname)     # save only the model weights
    # autoencoder.save(fname, include_optimizer=True)     # save the entire model

    # Sanity check
    if not check_weights_equality(autoencoder.encoder.get_weights(), wts_encoder):
        print("\nERROR: weights of the encoder have changed.")
    if not check_weights_equality(autoencoder.channel.get_weights(), wts_mdn):
        print("\nERROR: weights of the MDN (channel) have changed.")

    return autoencoder, np.array(train_loss), np.array(val_loss)


def finetune_mdn_baseline(mdn_model, target_domain_x, target_domain_y, model_dir, n_epochs=200):
    # Adaptation of the MDN using fine-tuning with a warm start based on the source domain parameters
    print("\nFine-tuning the MDN model on target domain data:")
    t_init = time.time()
    # Weights are copied from the trained MDN, but a new optimizer is initialized
    mdn_model_new = initialize_MDN_model(mdn_model.n_mixes, mdn_model.n_dims, mdn_model.n_hidden)
    mdn_model_new.set_weights(mdn_model.get_weights())
    mdn_model_new.summary()

    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    batch_size = max(10, int(np.ceil(0.01 * target_domain_x.shape[0])))
    hist = mdn_model_new.fit(x=target_domain_x, y=target_domain_y, batch_size=batch_size, epochs=n_epochs)

    t_adapt = time.time() - t_init
    # Generate samples from the adapted Gaussian mixture. `y_samples` should have the same shape as `target_domain_y`
    pred_params = mdn_model_new.predict(target_domain_x)
    y_samples = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)

    # Save the MDN model to a file
    fname = os.path.join(model_dir, 'channel_model', 'channel')
    mdn_model_new.save_weights(fname)  # save only the model weights
    # mdn_model_new.save(fname, include_optimizer=True)   # save the entire model

    return mdn_model_new, y_samples, t_adapt


def finetune_mdn_baseline_last_layer(mdn_model, target_domain_x, target_domain_y, model_dir, n_epochs=200):
    # Adaptation of the MDN's last layer using fine-tuning with a warm start based on the source domain parameters
    print("\nFine-tuning the last layer of the MDN model on target domain data:")
    t_init = time.time()
    # Create a new model and freeze the weights of all but its final layer
    mdn_model_new = MDN_model(mdn_model.n_hidden, mdn_model.n_dims, mdn_model.n_mixes)
    n_layers = len(mdn_model_new.layers)
    for i in range(n_layers - 1):
        mdn_model_new.layers[i].trainable = False

    weights_mdn = mdn_model.get_weights()
    # Compile and build the new model
    loss_mdn = MDN_base.get_mixture_loss_func(mdn_model.n_dims, mdn_model.n_mixes)
    optim_obj = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ADAM, epsilon=EPSILON_ADAM)
    mdn_model_new.compile(loss=loss_mdn, optimizer=optim_obj)
    _ = mdn_model_new(tf.keras.Input(shape=(mdn_model.n_dims,)))
    # Initialize using the weights of the trained MDN
    mdn_model_new.set_weights(weights_mdn)
    mdn_model_new.summary()
    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    batch_size = max(10, int(np.ceil(0.01 * target_domain_x.shape[0])))
    hist = mdn_model_new.fit(x=target_domain_x, y=target_domain_y, batch_size=batch_size, epochs=n_epochs)

    t_adapt = time.time() - t_init
    # Verify that the weights of all but the final layer have not changed
    flag_eq = check_weights_equality(mdn_model_new.get_weights(), weights_mdn, n_arrays=(2 * (n_layers - 1)))
    if not flag_eq:
        print("ERROR: weights of the adapted MDN layers are different from expected.")

    # Generate samples from the adapted Gaussian mixture. `y_samples` should have the same shape as `target_domain_y`
    pred_params = mdn_model_new.predict(target_domain_x)
    y_samples = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)

    # Save the MDN model to a file
    fname = os.path.join(model_dir, 'channel_model', 'channel')
    mdn_model_new.save_weights(fname)  # save only the model weights
    # mdn_model_new.save(fname, include_optimizer=True)   # save the entire model

    return mdn_model_new, y_samples, t_adapt


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Load the channel input/output samples for adaptation
    target_domain_x, target_domain_y = load_data_transceiver(
        os.path.join(args.data_dir, TX_DATA_BASENAME), os.path.join(args.data_dir, RX_DATA_BASENAME), shuffle=True
    )
    target_domain_x = tf.convert_to_tensor(target_domain_x, dtype=DTYPE_TF)
    target_domain_y = tf.convert_to_tensor(target_domain_y, dtype=DTYPE_TF)
    print("\nSize of the adaptation data: {:d}".format(target_domain_x.shape[0]))

    # Load the saved channel model weights into a newly initialized channel model
    mdn_model = initialize_MDN_model(
        config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
    )
    mdn_model.load_weights(args.channel_model_file).expect_partial()

    # Fine-tune the MDN model on the target domain data
    if args.last:
        mdn_model_new, _, _ = finetune_mdn_baseline_last_layer(mdn_model, target_domain_x, target_domain_y,
                                                               args.model_dir)
    else:
        mdn_model_new, _, _ = finetune_mdn_baseline(mdn_model, target_domain_x, target_domain_y, args.model_dir)

    weights_mdn_new = mdn_model_new.get_weights()
    # Load the saved autoencoder model weights into a newly initialized autoencoder model.
    # The parameters of the encoder and MDN (channel) networks are frozen
    n_train = config_basic['mod_order'] * 1000    # value is not important
    autoencoder = initialize_autoencoder(config_basic['type_autoencoder'], mdn_model_new, config_basic, config_optimizer,
                                         n_train, temperature=CONFIG_ANNEAL['temp_final'], freeze_encoder=True)
    autoencoder.load_weights(args.autoencoder_model_file).expect_partial()

    # Update the channel model of the autoencoder with the weights of the adapted MDN
    autoencoder.channel.set_weights(weights_mdn_new)

    # Data for training the autoencoder
    x_train, x_val, x_test, labels_test = generate_data_autoencoder(
        config_basic['data_file_autoenc'], config_basic['mod_order'], args.n_train_per_symbol, args.n_test_per_symbol
    )
    # Train the autoencoder based on the fine-tuned MDN
    autoencoder, train_loss, val_loss = wrapper_train_autoencoder(autoencoder, x_train, x_val, config_optimizer,
                                                                  args.model_dir)


if __name__ == '__main__':
    # This is needed to prevent `multiprocessing` calls from getting hung on MacOS
    multiprocessing.set_start_method("spawn")
    main()
