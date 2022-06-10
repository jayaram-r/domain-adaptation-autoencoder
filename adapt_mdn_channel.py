"""
Main script for adapting the MDN channel model on both simulated and real data.
For a description of the command line inputs, type:
`python adapt_mdn_channel.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--constellation-file or --cof: Path to the numpy file with the initial constellation symbols. If not specified, by
                               default, an M-QAM constellation is used.
--adaptation-objective or --ao: Type of objective to use for MDN adaptation. Default value is 'log_likelihood', but it
                                can be changed to 'log_posterior'.
--data-dir-adapt or -da: path to the data directory where the adaptation data files are saved. Not needed if using simulated channel.
--data-dir-test or -dt: path to the data directory where the test data files are saved. Not needed if using simulated channel.
--model-dir or -m: path to the model directory where the adaptation parameters file should be saved.
--output-dir or -o: path to the directory where output files should be saved.
--plot: Use this option to enable plots to the generated in the output directory.
# Options below are for using simulated channel data
--sim: use this option for simulated channel data.
--type-channel or --tc: Type of simulated channel. Options are ['AWGN', 'fading', 'fading_ricean', 'fading_rayleigh'].
--SNR-channel or --snr: SNR in dB of the simulated channel. Default is 14dB.
--n-adapt-per-symbol or --nad: Number of adaptation samples per symbol

Most of the other options can be left at their default values unless there is a specific need to change them.
Any non-default options that were used for training the MDN using `train_mdn_channel.py` should also be set in
this script (if applicable). For example, if the dimension of the encoding is 4 and the number of components in the
channel MDN is 3, then the following options should be specified in both the training and adaptation scripts: `--de 4 --nc 3`

DATA FILES (not needed if using `--sim`):
Before running, ensure that the data files for adaptation are available in the data directory specified by the
command line option `--data-dir-adapt`. By default this is a directory `./data_adapt` under the current working directory.
Also ensure that the data files for testing are available in the directory specified by the command line option
`--data-dir-test`. By default this is a directory `./data_test` under the current working directory.

Specifically, the following two mat files should be present in each of the above data directories:
- `tx_symbols.mat`: File containing the transmitted (modulated) symbols.
- `rx_symbols.mat`: File containing the received symbols.

USAGE EXAMPLES:
    Assume that the script `train_mdn_channel.py` has been run on data from FPGA, and that the MDN model
    and constellation file are saved in the following directories:
    model_dir="${PWD}/models_mdn"
    cmf="${model_dir}/channel_model/channel"
    cof="${model_dir}/constellation_init.npy"
    data_adapt="${PWD}/data_adapt"
    data_test="${PWD}/data_test"
    output_dir="${PWD}/outputs_adapt"

    python adapt_mdn_channel.py --n-bits 4 --cmf $cmf --cof $cof --da $data_adapt --dt $data_test -m $model_dir -o $output_dir --plot

    After running, check for the adaptation parameters file `adaptation_params.npy` saved under $model_dir.
    Check the output directory for performance metrics and plots.
"""
import sys
import argparse
import os
import csv
import time
import math
import numpy as np
import tensorflow as tf
import multiprocessing
from functools import partial
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    load_data_transceiver,
    check_weights_equality,
    get_num_jobs,
    get_labels_from_symbols,
    affine_transform_gmm_params,
    split_adaptation_params,
    get_mixture_params,
    sample_batch_MDN,
    generate_channel_data_simulated
)
from helpers.MDN_classes import (
    MDN_model,
    initialize_MDN_model,
    load_channel_model_from_file
)
from helpers.optimize import (
    get_loss_func_proposed,
    minimize_channel_adaptation_bfgs,
    minimize_channel_adaptation_bfgs_parallel,
    minimize_channel_adaptation_adam,
    minimize_channel_adaptation_adam_parallel
)
from helpers.metrics import validation_metric_mdn
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
    parser.add_argument('--constellation-file', '--cof', default='',
                        help='Path to the numpy file with the initial constellation symbols. If not specified, by '
                             'default, an M-QAM constellation is used.')
    parser.add_argument('--target-channel-model-file', '--tcmf', default='',
                        help='Path to the file or directory containing the target channel model weights. '
                             'If specified, this is used to evaluate the best-case performance of a fully trained '
                             'channel model on the target domain.')
    parser.add_argument('--adaptation-objective', '--ao', choices=['log_posterior', 'log_likelihood'],
                        default='log_likelihood', help='Type of objective function for adapting the channel model.')
    # Data directories
    parser.add_argument('--data-dir-adapt', '--da', default='./data_adapt',
                        help='Directory with the data files for adaptation.')
    parser.add_argument('--data-dir-test', '--dt', default='./data_test',
                        help='Directory with the data files for testing performance.')
    # Options to use a simulated channel model
    parser.add_argument('--simulate-channel', '--sim', action='store_true', default=False,
                        help="Use option to simulate the adaptation data according to a standard channel model, "
                             "e.g. AWGN or Ricean fading. Option '--type-channel' specifies the type of channel model.")
    parser.add_argument('--type-channel', '--tc', choices=['AWGN', 'fading', 'fading_ricean', 'fading_rayleigh'],
                        default='fading_ricean', help='Type of simulated channel model to use')
    parser.add_argument('--SNR-channel', '--snr', type=float, default=14.0, help='SNR of the simulated channel in dB.')
    parser.add_argument('--sigma-noise-measurement', '--sigma', type=float, default=0.09,
                        help='Noise standard deviation of the channel. This is estimated from the channel data in '
                             'case the data files are provided.')
    parser.add_argument('--n-adapt-per-symbol', '--nad', type=int, default=20,
                        help='Number of target domain adaptation samples per constellation symbol')
    parser.add_argument('--n-test-per-symbol', '--nte', type=int, default=1500,
                        help='Number of test samples per constellation symbol')
    # Model and output directories
    parser.add_argument('--model-dir', '-m', default='./models_mdn', help='Directory for saving the model files')
    parser.add_argument('--output-dir', '-o', default='./outputs_mdn_adapt',
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
    parser.add_argument('--append-outputs', action='store_true', default=False,
                        help="Use this option to append the output files instead of creating a new one. Useful when "
                             "recording the outputs over multiple trials for averaging.")
    parser.add_argument('--plot', action='store_true', default=False,
                        help="Use this option to generate plots of the adapted MDN solution")
    parser.add_argument('--avg-power-symbols', '--aps', type=float, default=1.,
                        help='Average power of the symbols in the constellation (maximum value)')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    parser.add_argument('--n-jobs', type=int, default=8,
                        help='Number of parallel jobs/CPU cores available to use. Can be an upper bound.')
    args = parser.parse_args()

    if args.simulate_channel:
        print("\nUsing simulated channel data of type '{}' and SNR {:g}dB for adaptation".format(args.type_channel,
                                                                                                 args.SNR_channel))
    args.channel_model_file = os.path.abspath(args.channel_model_file)
    if args.constellation_file:
        args.constellation_file = os.path.abspath(args.constellation_file)
        print("Initializing the MDN constellation from the file: {}".format(args.constellation_file))

    if args.target_channel_model_file:
        args.target_channel_model_file = os.path.abspath(args.target_channel_model_file)

    args.data_dir_adapt = os.path.abspath(args.data_dir_adapt)
    args.data_dir_test = os.path.abspath(args.data_dir_test)
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
        'adaptation_objective': args.adaptation_objective,
        'type_channel': args.type_channel,
        'SNR_channel_dB': args.SNR_channel,
        'sigma_noise_measurement': args.sigma_noise_measurement,
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': 0.0,
        'scale_outputs': True,
        'EbNodB_range': SNR_RANGE_DEF,
        'EbNodB_min': SNR_MIN_RICEAN,
        'EbNo_min': 10. ** (SNR_MIN_RICEAN / 10.),
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


def helper_param_transform(mdn_model, x_test, params_affine):
    pred_params_orig = mdn_model.predict(x_test)
    mus, sigmas, pi_logits = get_mixture_params(pred_params_orig, mdn_model.n_mixes, mdn_model.n_dims, logits=True)
    new_mus, new_sigmas, new_pi_logits = affine_transform_gmm_params(mus, sigmas, pi_logits, params_affine,
                                                                     mdn_model.n_mixes, mdn_model.n_dims)
    pred_params_new = tf.concat([new_mus, new_sigmas, new_pi_logits], 1)

    return pred_params_orig, pred_params_new


def wrapper_adaptation_proposed(mdn_model, target_domain_input, target_domain_x, target_domain_y, unique_x,
                                x_test, y_test, config, model_dir, type_metric='inverse_trans_log_like',
                                optim_method='bfgs', n_epochs=200, num_init=1):
    # Options for `type_metric` are 'inverse_trans_log_like' and 'inverse_trans_log_post'.
    optim_method = optim_method.lower()
    assert optim_method in ('adam', 'bfgs'), "Invalid value '{}' for the input 'optim_method'".format(optim_method)

    t_init = time.time()
    # Uniform prior probability for the unique symbols
    prior_prob_x = (1. / unique_x.shape[0]) * tf.ones([unique_x.shape[0]])
    unique_x = tf.convert_to_tensor(unique_x, dtype=DTYPE_TF)
    # Logarithmically-spaced lambda values between 10^-5 and 1
    lambda_values = np.logspace(-5, 0, num=8, base=10.)
    num_lambda = lambda_values.shape[0]
    metric_val = np.zeros(num_lambda)
    psi_values_adapted = []
    # Minimize the channel adaptation loss function for different lambda values and validate the solution
    weights_mdn = mdn_model.get_weights()
    results_minimize = None
    if config['n_jobs'] > 1:
        # Use `multiprocessing` to parallelize the minimization over lambda values
        # Call the function that creates the loss function and its gradient once. This performs the graph compilation
        # using `tf.function` for the first time, which can be slower than subsequent calls in parallel
        loss_func_proposed, loss_func_val_and_grad = get_loss_func_proposed(
            mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes, config['dim_encoding'],
            lambda_value=tf.constant(1.0, dtype=tf.float32), type_objec=config['adaptation_objective']
        )
        if optim_method == 'bfgs':
            helper_minimize = partial(
                minimize_channel_adaptation_bfgs_parallel, weights_mdn, target_domain_x, target_domain_y,
                unique_x, prior_prob_x, mdn_model.n_mixes, config['dim_encoding'], mdn_model.n_hidden,
                config['adaptation_objective'], num_init
            )
        else:
            helper_minimize = partial(
                minimize_channel_adaptation_adam_parallel, weights_mdn, target_domain_x, target_domain_y,
                unique_x, prior_prob_x, mdn_model.n_mixes, config['dim_encoding'], mdn_model.n_hidden,
                config['adaptation_objective'], n_epochs
            )

        pool_obj = multiprocessing.Pool(processes=config['n_jobs'])
        results_minimize = []
        mpr = pool_obj.map_async(helper_minimize, lambda_values, callback=results_minimize.extend)
        # mpr.get()
        pool_obj.close()
        pool_obj.join()
        # results_minimize = [helper_minimize(v) for v in lambda_values]

    for j in range(num_lambda):
        print("lambda = {:g}".format(lambda_values[j]))
        if results_minimize is None:
            if optim_method == 'bfgs':
                psi_values, loss_final = minimize_channel_adaptation_bfgs(
                    mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
                    config['dim_encoding'], lambda_values[j], type_objec=config['adaptation_objective'], num_init=num_init
                )
            else:
                psi_values, loss_final = minimize_channel_adaptation_adam(
                    mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
                    config['dim_encoding'], lambda_values[j], type_objec=config['adaptation_objective'], n_epochs=n_epochs
                )

        else:
            # Precomputed result from the parallel run
            psi_values, loss_final = results_minimize[j]

        # Calculate the validation metric on the target domain training set
        metric_val[j] = validation_metric_mdn(mdn_model, psi_values, unique_x, prior_prob_x, target_domain_input,
                                              target_domain_x, target_domain_y, metric=type_metric)
        psi_values_adapted.append(psi_values)

    # Find the `lambda` value that has the minimum metric on the target domain training set
    v_min = np.min(metric_val)
    ind_cand = np.where(metric_val <= v_min)[0]
    if ind_cand.shape[0] == 1:
        j_star = ind_cand[0]
    else:
        # Break ties by choosing the solution corresponding to smallest `lambda`
        lambda_cand = lambda_values[ind_cand]
        j_star = ind_cand[np.argmin(lambda_cand)]

    lambda_best = lambda_values[j_star]
    min_metric_val = metric_val[j_star]
    psi_values = psi_values_adapted[j_star]
    t_adapt = time.time() - t_init
    print("\nSelected lambda: {:.4e}. Minimum validation metric: {:.8f}".format(lambda_best, min_metric_val))
    # Save the adaptation parameters corresponding to the best lambda value
    fname_params = os.path.join(model_dir, ADAPTATION_PARAMS_BASENAME)
    with open(fname_params, 'wb') as fp:
        np.save(fp, psi_values.numpy())

    # Evaluate the adapted MDN on the test data
    params_affine = split_adaptation_params(psi_values, mdn_model.n_mixes, mdn_model.n_dims)
    pred_params_orig, pred_params_new = helper_param_transform(mdn_model, x_test, params_affine)
    # Log-likelihood of the original and adapted MDN on the test channel outputs
    loss_mdn = MDN_base.get_mixture_loss_func(mdn_model.n_dims, mdn_model.n_mixes)
    loglike_orig = -1. * loss_mdn(y_test, pred_params_orig)
    loglike_adap = -1. * loss_mdn(y_test, pred_params_new)
    print("Log-likelihood of the original and adapted MDN on the test data: {:.6f}, {:.6f}".format(loglike_orig,
                                                                                                   loglike_adap))
    change = (loglike_adap - loglike_orig) / np.abs(loglike_orig)
    print("Relative change in log-likelihood(%): {:.6f}".format(100. * change))
    # Generate samples from the adapted Gaussian mixture. `y_samples` should have the same shape as `target_domain_y`
    _, pred_params = helper_param_transform(mdn_model, target_domain_x, params_affine)
    y_samples = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)

    return loglike_adap, loglike_orig, y_samples, t_adapt


def transfer_mdn_baseline(mdn_model, target_domain_x, target_domain_y, x_test, y_test, n_epochs=200):
    # Adaptation of the MDN using a baseline method where the MDN is initialized with the current weights and it is
    # trained on the target domain data.
    t_init = time.time()
    # Weights are copied from the trained MDN, but a new optimizer is initialized
    mdn_model_new = initialize_MDN_model(mdn_model.n_mixes, mdn_model.n_dims, mdn_model.n_hidden)
    mdn_model_new.set_weights(mdn_model.get_weights())
    mdn_model_new.summary()

    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    batch_size = max(10, int(np.ceil(0.01 * target_domain_x.shape[0])))
    hist = mdn_model_new.fit(x=target_domain_x, y=target_domain_y, batch_size=batch_size, epochs=n_epochs)

    t_adapt = time.time() - t_init
    # Parameters predicted by the original and adapted MDN on the test inputs
    pred_params_orig = mdn_model.predict(x_test)
    pred_params_new = mdn_model_new.predict(x_test)
    # Log-likelihood of the original and adapted MDN on the test channel outputs
    loss_mdn = MDN_base.get_mixture_loss_func(mdn_model.n_dims, mdn_model.n_mixes)
    loglike_orig = -1. * loss_mdn(y_test, pred_params_orig)
    loglike_adap = -1. * loss_mdn(y_test, pred_params_new)
    print("\nLog-likelihood of the original and adapted MDN on the test data: {:.6f}, {:.6f}".format(loglike_orig,
                                                                                                     loglike_adap))
    change = (loglike_adap - loglike_orig) / np.abs(loglike_orig)
    print("Relative change in log-likelihood(%): {:.6f}".format(100. * change))
    # Generate samples from the adapted Gaussian mixture. `y_samples` should have the same shape as `target_domain_y`
    pred_params = mdn_model_new.predict(target_domain_x)
    y_samples = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)

    return loglike_adap, loglike_orig, y_samples, t_adapt


def transfer_mdn_baseline_last_layer(mdn_model, target_domain_x, target_domain_y, x_test, y_test, n_epochs=200):
    # Adaptation of the MDN using a baseline method where the MDN is initialized with the current weights and only
    # the final layer is adapted using the target domain data.
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

    # Parameters predicted by the original and adapted MDN on the test inputs
    pred_params_orig = mdn_model.predict(x_test)
    pred_params_new = mdn_model_new.predict(x_test)
    # Log-likelihood of the original and adapted MDN on the test channel outputs
    loglike_orig = -1. * loss_mdn(y_test, pred_params_orig)
    loglike_adap = -1. * loss_mdn(y_test, pred_params_new)
    print("\nLog-likelihood of the original and adapted MDN on the test data: {:.6f}, {:.6f}".format(loglike_orig,
                                                                                                     loglike_adap))
    change = (loglike_adap - loglike_orig) / np.abs(loglike_orig)
    print("Relative change in log-likelihood(%): {:.6f}".format(100. * change))
    # Generate samples from the adapted Gaussian mixture. `y_samples` should have the same shape as `target_domain_y`
    pred_params = mdn_model_new.predict(target_domain_x)
    y_samples = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)

    return loglike_adap, loglike_orig, y_samples, t_adapt


def write_results(loglike_orig, loglike_adap, time_adapt, methods, output_dir, append_outputs):
    # Write the log-likelihood values and running times to a file
    if append_outputs:
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True

    fname = os.path.join(output_dir, 'loglike_comparison.csv')
    with open(fname, mode=mode) as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        if header:
            cw.writerow(['method', 'loglike', 'rel_loglike'])

        cw.writerow(['no_adaptation', '{:.6f}'.format(loglike_orig), '0.0'])
        for m in methods:
            change = (loglike_adap[m] - loglike_orig) / np.abs(loglike_orig)
            cw.writerow([m, '{:.6f}'.format(loglike_adap[m]), '{:.6f}'.format(change)])

    fname = os.path.join(output_dir, 'time_comparison.csv')
    with open(fname, mode=mode) as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        if header:
            cw.writerow(methods)

        cw.writerow(['{:.2f}'.format(time_adapt[m]) for m in methods])


def plot_mdn_samples(y_samples_orig, y_samples, target_domain_x, target_domain_y, methods, output_dir):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    axes[0, 0].scatter(target_domain_y[:, 0], target_domain_y[:, 1], alpha=0.4, c='grey')
    axes[0, 0].scatter(y_samples_orig[:, 0], y_samples_orig[:, 1], alpha=0.4, c='lightseagreen')
    axes[0, 0].scatter(target_domain_x[:, 0], target_domain_x[:, 1], alpha=0.4, c='k', marker='^')
    axes[0, 0].set_title("no_adaptation", fontsize=12)

    for i in (1, 2, 3):
        m = methods[i - 1]
        j = k = 1
        if i == 1:
            j = 0
            k = 1
        if i == 2:
            j = 1
            k = 0

        axes[j, k].scatter(target_domain_y[:, 0], target_domain_y[:, 1], alpha=0.4, c='grey')
        axes[j, k].scatter(y_samples[m][:, 0], y_samples[m][:, 1], alpha=0.4, c='lightseagreen')
        axes[j, k].scatter(target_domain_x[:, 0], target_domain_x[:, 1], alpha=0.4, c='k', marker='^')
        axes[j, k].set_title(m, fontsize=12)

    # remove the x and y ticks
    plt.setp(axes, xticks=[], yticks=[])
    fname = os.path.join(output_dir, 'plot_mdn_samples.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)


def load_adaptation_and_test_data(data_dir_adapt, data_dir_test):
    # Load the channel data for adaptation
    target_domain_x, target_domain_y = load_data_transceiver(
        os.path.join(data_dir_adapt, TX_DATA_BASENAME), os.path.join(data_dir_adapt, RX_DATA_BASENAME), shuffle=True
    )
    target_domain_x = tf.convert_to_tensor(target_domain_x, dtype=DTYPE_TF)
    target_domain_y = tf.convert_to_tensor(target_domain_y, dtype=DTYPE_TF)
    # Remove this after testing
    # target_domain_x = target_domain_x[:320, :]
    # target_domain_y = target_domain_y[:320, :]
    print("\nNumber of adaptation samples: {:d}".format(target_domain_x.shape[0]))
    # Load the channel data for testing
    x_test, y_test = load_data_transceiver(
        os.path.join(data_dir_test, TX_DATA_BASENAME), os.path.join(data_dir_test, RX_DATA_BASENAME), shuffle=True
    )
    x_test = tf.convert_to_tensor(x_test, dtype=DTYPE_TF)
    y_test = tf.convert_to_tensor(y_test, dtype=DTYPE_TF)
    print("Number of test samples: {:d}".format(x_test.shape[0]))

    return target_domain_x, target_domain_y, x_test, y_test


def main():
    # Read the command line inputs
    args, config_basic, config_optimizer = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    if args.constellation_file:
        constellation_init = np.load(args.constellation_file)
    else:
        # Constellation of standard M-QAM
        constellation_init = StandardQAM(config_basic['n_bits'],
                                         avg_power=config_basic['avg_power_symbols']).constellation

    # Load the saved MDN model weights into a newly initialized MDN model
    mdn_model = initialize_MDN_model(
        config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
    )
    mdn_model.load_weights(args.channel_model_file).expect_partial()

    # Load the MDN model trained on the target domain if specified
    mdn_model_target = None
    if args.target_channel_model_file:
        mdn_model_target = initialize_MDN_model(
            config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
        )
        mdn_model_target.load_weights(args.target_channel_model_file).expect_partial()

    if not args.simulate_channel:
        # Load the channel data for adaptation from the specified data directories
        target_domain_x, target_domain_y, x_test, y_test = load_adaptation_and_test_data(args.data_dir_adapt,
                                                                                         args.data_dir_test)
    else:
        # Generate data for adaptation and testing from a standard channel model
        n_adapt = args.n_adapt_per_symbol * config_basic['mod_order']
        target_domain_x, target_domain_y, _ = generate_channel_data_simulated(
            args.type_channel, config_basic['SNR_channel_dB'], n_adapt, config_basic, constellation_init
        )
        print("\nNumber of adaptation samples: {:d}".format(n_adapt))
        n_test = args.n_test_per_symbol * config_basic['mod_order']
        x_test, y_test, _ = generate_channel_data_simulated(
            args.type_channel, config_basic['SNR_channel_dB'], n_test, config_basic, constellation_init
        )
        print("Number of test samples: {:d}".format(n_test))

    # Get the one-hot-coded labels for the adaptation samples
    target_domain_input = get_labels_from_symbols(target_domain_x.numpy(), constellation_init)
    time_adapt = dict()     # running time
    loglike_adap = dict()   # adapted log-likelihood
    loglike_orig = None     # original log-likelihood
    y_samples = dict()      # samples generated from the adapted MDN
    methods = []
    print("\nRunning MDN adaptation using the proposed method:")
    method = 'proposed'
    methods.append(method)
    loglike_adap[method], loglike_orig, y_samples[method], time_adapt[method] = wrapper_adaptation_proposed(
        mdn_model, target_domain_input, target_domain_x, target_domain_y, constellation_init, x_test, y_test,
        config_basic, args.model_dir
    )
    print("Time taken for adaptation: {:.2f} seconds".format(time_adapt[method]))

    print("\nRunning MDN adaptation using the baseline method 'transfer':")
    method = 'transfer'
    methods.append(method)
    loglike_adap[method], _, y_samples[method], time_adapt[method] = transfer_mdn_baseline(
        mdn_model, target_domain_x, target_domain_y, x_test, y_test
    )
    print("Time taken for adaptation: {:.2f} seconds".format(time_adapt[method]))

    print("\nRunning MDN adaptation using the baseline method 'transfer_last_layer':")
    method = 'transfer_last_layer'
    methods.append(method)
    loglike_adap[method], _, y_samples[method], time_adapt[method] = transfer_mdn_baseline_last_layer(
        mdn_model, target_domain_x, target_domain_y, x_test, y_test
    )
    print("Time taken for adaptation: {:.2f} seconds".format(time_adapt[method]))

    if mdn_model_target:
        method = 'retrained_target'
        methods.append(method)
        # Log-likelihood on the test set
        loss_mdn = MDN_base.get_mixture_loss_func(mdn_model_target.n_dims, mdn_model_target.n_mixes)
        pred_params_new = mdn_model_target.predict(x_test)
        loglike_adap[method] = -1. * loss_mdn(y_test, pred_params_new)
        # Samples
        pred_params_new = mdn_model_target.predict(target_domain_x)
        y_samples[method] = sample_batch_MDN(pred_params_new, mdn_model_target.n_mixes, mdn_model_target.n_dims)
        time_adapt[method] = -1.0

    # Write the log-likelihood values and running times to a file
    write_results(loglike_orig, loglike_adap, time_adapt, methods, args.output_dir, args.append_outputs)

    # Plot the generated samples by the different methods
    if args.plot:
        # Generate samples from the original MDN Gaussian mixture
        pred_params = mdn_model.predict(target_domain_x)
        y_samples_orig = sample_batch_MDN(pred_params, mdn_model.n_mixes, mdn_model.n_dims)
        plot_mdn_samples(y_samples_orig, y_samples, target_domain_x, target_domain_y, methods, args.output_dir)


if __name__ == '__main__':
    # This is needed to prevent `multiprocessing` calls from getting hung on MacOS
    multiprocessing.set_start_method("spawn")
    main()
