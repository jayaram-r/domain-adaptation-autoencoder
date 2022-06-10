"""
Main script for adapting the channel model and autoencoder on real samples measured by FPGA.
For a description of the command line inputs, type:
`python adapt_autoencoder_measure.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--type-autoencoder or --ta: type of autoencoder model ('standard', 'symbol_estimation_mmse', 'symbol_estimation_map'), (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--autoencoder-model-file or --amf: path to the autoencoder model file or directory (REQUIRED).
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

    python adapt_autoencoder_measure.py --n-bits 4 --cmf $cmf --amf $amf --ta standard -m $model_dir -d $data_dir

    After running, check for the adaptation parameters file `adaptation_params.npy` saved under $model_dir.
"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
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
    get_labels_from_symbols
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
from helpers.optimize import (
    get_loss_func_proposed,
    minimize_channel_adaptation_bfgs,
    minimize_channel_adaptation_bfgs_parallel
)
from helpers.metrics import calculate_metric_autoencoder
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
    parser.add_argument('--type-autoencoder', '--ta', required=True,
                        choices=['standard', 'symbol_estimation_mmse', 'symbol_estimation_map'],
                        help='Type of autoencoder model trained.')
    parser.add_argument('--adaptation-objective', '--ao', choices=['log_posterior', 'log_likelihood'],
                        default='log_posterior', help='Type of objective function for adapting the channel model.')
    parser.add_argument('--lambda-fixed', type=float, default=-1.0,
                        help='Regularization hyper-parameter value. It is set automatically if this option is not used')
    # Data and model directory
    parser.add_argument('--data-dir', '-d', default='./data_adapt', help='Directory for the saved data files')
    parser.add_argument('--model-dir', '-m', default='./models_adapt', help='Directory for saving the model files')
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
    args.model_dir = os.path.abspath(args.model_dir)
    args.data_dir = os.path.abspath(args.data_dir)
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
        'type_autoencoder': args.type_autoencoder,
        # 'data_file_autoenc': os.path.join(args.data_dir, get_autoencoder_data_filename(args.n_bits)),
        'adaptation_objective': args.adaptation_objective,
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs),
        'n_jobs': get_num_jobs(args.n_jobs)
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


def wrapper_autoencoder_adaptation(autoencoder_orig, mdn_model, target_domain_input, target_domain_x, target_domain_y,
                                   config, model_dir, type_metric='log_loss', num_init=1, lambda_fixed=-1.0):
    # Valid options for `type_metric` are ['log_loss', 'error_rate', 'inverse_trans_log_like', 'inverse_trans_log_post']
    # Adapt the channel model and the autoencoder using real samples measured by FPGA.
    # Distinct constellation symbols
    unique_x = autoencoder_orig.encoder(autoencoder_orig.inputs_unique)
    # Probability of each unique symbol estimated from the training data
    prior_prob_x = autoencoder_orig.prior_prob_inputs

    if lambda_fixed < 0.:
        # Logarithmically-spaced lambda values between 10^-5 and 100
        lambda_values = np.logspace(-5, 2, num=8, base=10.)
    else:
        lambda_values = np.array([lambda_fixed])

    num_lambda = lambda_values.shape[0]
    metric_train = np.zeros(num_lambda)
    autoencoders_adapted = []
    print("Number of target domain adaptation samples: {:d}".format(len(target_domain_x)))
    t_adapt = 0.
    # Minimize the channel adaptation loss function for different lambda values and evaluate the solution
    weights_mdn = mdn_model.get_weights()
    results_minimize = None
    if config['n_jobs'] > 1:
        t1 = time.time()
        # Call the function that creates the loss function and its gradient once. This performs the graph compilation
        # using `tf.function` for the first time, which can be slower than subsequent calls in parallel
        loss_func_proposed, loss_func_val_and_grad = get_loss_func_proposed(
            mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
            config['dim_encoding'], lambda_value=tf.constant(1.0, dtype=tf.float32), type_objec=config['adaptation_objective']
        )
        # Use `multiprocessing` to parallelize the minimization over lambda values
        helper_minimize = partial(
            minimize_channel_adaptation_bfgs_parallel, weights_mdn, target_domain_x, target_domain_y,
            unique_x, prior_prob_x, mdn_model.n_mixes, config['dim_encoding'], mdn_model.n_hidden,
            config['adaptation_objective'], num_init
        )
        pool_obj = multiprocessing.Pool(processes=config['n_jobs'])
        results_minimize = []
        mpr = pool_obj.map_async(helper_minimize, lambda_values, callback=results_minimize.extend)
        # mpr.get()
        pool_obj.close()
        pool_obj.join()
        # results_minimize = [helper_minimize(v) for v in lambda_values]
        t_adapt += (time.time() - t1)

    for j in range(num_lambda):
        t1 = time.time()
        print("lambda = {:g}".format(lambda_values[j]))
        if results_minimize is None:
            psi_values, loss_final = minimize_channel_adaptation_bfgs(
                mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
                config['dim_encoding'], lambda_values[j], type_objec=config['adaptation_objective'], num_init=num_init
            )
        else:
            # Precomputed result from the parallel run
            psi_values, loss_final = results_minimize[j]

        # Create a new autoencoder model with the adapted channel model and the pre-trained encoder and
        # decoder networks
        autoencoder = initialize_autoencoder_adapted(
            config['type_autoencoder'], mdn_model, autoencoder_orig.encoder, autoencoder_orig.decoder, psi_values,
            config, temperature=CONFIG_ANNEAL['temp_final']
        )
        # Calculate the performance metric (log-loss or BER) on the target domain training set
        metric_train[j] = calculate_metric_autoencoder(
            autoencoder, target_domain_input, target_domain_y, metric=type_metric
        )
        autoencoders_adapted.append(autoencoder)
        t_adapt += (time.time() - t1)

    # Find the `lambda` value that has the minimum metric on the target domain training set
    v_min = np.min(metric_train)
    ind_cand = np.where(metric_train <= v_min)[0]
    if ind_cand.shape[0] == 1:
        j_star = ind_cand[0]
    else:
        # Break ties by choosing the solution corresponding to largest `lambda`
        lambda_cand = lambda_values[ind_cand]
        j_star = ind_cand[np.argmax(lambda_cand)]

    lambda_best = lambda_values[j_star]
    metric_min_train = metric_train[j_star]
    print("\nSelected lambda: {:.4e}. Minimum metric: {:.8f}".format(lambda_best, metric_min_train))
    print("Time taken for adaptation: {:.2f} seconds".format(t_adapt))
    # Adapted autoencoder corresponding to the best lambda value
    autoencoder = autoencoders_adapted[j_star]

    # Saving this model to a file is not required. We will only save the best adaptation parameters.
    # fname_auto = os.path.join(model_dir, get_autoencoder_name(autoencoder) + '_adapted', 'autoencoder')
    # autoencoder.save_weights(fname_auto)     # save only the model weights

    # Save the best adaptation parameters in the numpy format
    fname_params = os.path.join(model_dir, ADAPTATION_PARAMS_BASENAME)
    with open(fname_params, 'wb') as fp:
        np.save(fp, autoencoder.psi_values.numpy())

    '''
    # Sanity check: comment out after testing.
    # Load the adaptation parameters from the saved file and create a new autoencoder with the adaptation parameters.
    # Compare the weights and predictions of this model with the adapted autoencoder already in memory.
    psi_values = tf.convert_to_tensor(np.load(fname_params), dtype=DTYPE_TF)
    autoencoder_new = initialize_autoencoder_adapted(
        config['type_autoencoder'], autoencoder_orig.channel, autoencoder_orig.encoder, autoencoder_orig.decoder,
        psi_values, config, temperature=CONFIG_ANNEAL['temp_final']
    )
    stat1 = check_weights_equality(autoencoder_new.get_weights(), autoencoder.get_weights())
    stat2 = check_weights_equality(autoencoder_new.encoder.get_weights(), autoencoder.encoder.get_weights())
    stat3 = check_weights_equality(autoencoder_new.decoder.get_weights(), autoencoder.decoder.get_weights())
    stat4 = check_weights_equality(autoencoder_new.channel.get_weights(), autoencoder.channel.get_weights())
    preds1 = autoencoder.predict(target_domain_input)
    preds2 = autoencoder_new.predict(target_domain_input)
    stat5 = np.allclose(preds1, preds2, rtol=0., atol=1e-6)
    preds1 = autoencoder.decoder_predict(target_domain_y)
    preds2 = autoencoder_new.decoder_predict(target_domain_y)
    stat6 = np.allclose(preds1, preds2, rtol=0., atol=1e-6)
    import pdb; pdb.set_trace()
    '''
    return autoencoder, lambda_best, metric_min_train


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Load the saved channel model weights into a newly initialized channel model
    mdn_model = initialize_MDN_model(
        config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
    )
    mdn_model.load_weights(args.channel_model_file).expect_partial()

    # Load the saved autoencoder model weights into a newly initialized autoencoder model
    n_train = config_basic['mod_order'] * 1000    # value is not important
    autoencoder = initialize_autoencoder(config_basic['type_autoencoder'], mdn_model, config_basic, config_optimizer,
                                         n_train, temperature=CONFIG_ANNEAL['temp_final'])
    autoencoder.load_weights(args.autoencoder_model_file).expect_partial()

    # Load the channel input/output samples for adaptation
    target_domain_x, target_domain_y = load_data_transceiver(
        os.path.join(args.data_dir, TX_DATA_BASENAME), os.path.join(args.data_dir, RX_DATA_BASENAME), shuffle=True
    )
    # Numpy array to TF tensor
    target_domain_x = tf.convert_to_tensor(target_domain_x, dtype=DTYPE_TF)
    target_domain_y = tf.convert_to_tensor(target_domain_y, dtype=DTYPE_TF)

    # Get the one-hot-coded labels for the adaptation samples
    unique_x = autoencoder.encoder(autoencoder.inputs_unique).numpy()
    target_domain_input = get_labels_from_symbols(target_domain_x.numpy(), unique_x)

    # Adapt the channel model and autoencoder
    _ = wrapper_autoencoder_adaptation(autoencoder, mdn_model, target_domain_input, target_domain_x, target_domain_y,
                                       config_basic, args.model_dir, lambda_fixed=args.lambda_fixed)


if __name__ == '__main__':
    # This is needed to prevent `multiprocessing` calls from getting hung on MacOS
    multiprocessing.set_start_method("spawn")
    main()
