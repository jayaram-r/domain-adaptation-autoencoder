"""
Main script for adapting the channel model and autoencoder on simulated channel variations.
For a description of the command line inputs, type:
`python adapt_autoencoder_sim.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--autoencoder-model-file or --amf: path to the autoencoder model file or directory (REQUIRED).
--type-autoencoder or --ta: type of autoencoder model ('standard', 'symbol_estimation_mmse', 'symbol_estimation_map', 'adapt_generative'), (REQUIRED).
--type-channel or --tc: type of simulated channel to adapt to.
--n-adapt-per-symbol or --nad: number of adaptation samples per symbol.
--snr-min: minimum SNR (in dB) over which the target domain channel is varied. Default is 4.
--snr-max: maximum SNR (in dB) over which the target domain channel is varied. Default is 20.
--output-dir or -o: output directory path.

Most of the other options can be left at their default values unless there is a specific need to change them.
Any non-default options that were used for training the autoencoder using `train_autoencoder.py` should also be set in
this script (if applicable). For example, if the dimension of the encoding is 4 and the number of components in the
channel MDN is 3, then the following options should be specified in both the training and adaptation scripts:
`--de 4 --nc 3`

Usage Examples:
    Assume that the script `train_autoencoder.py` has been run on data from an AWGN channel, and that the channel model
    and autoencoder models are saved in the following directories:
    model_dir=${PWD}/models_awgn
    channel_modelfile="${model_dir}/channel_model/channel"
    auto_modelfile="${model_dir}/autoencoder/autoencoder"
    output_dir=${PWD}/outputs_awgn_to_ricean

    1. Adapt the channel and autoencoder models to a Ricean fading channel using the inverse-affine-transformation
    method. The number of adaptation samples per symbol is set to 20.
    python adapt_autoencoder_sim.py --n-bits 4 --cmf $channel_modelfile --amf $auto_modelfile --ta standard
    --tc fading_ricean -o $output_dir --nad 20

    2. Adapt the channel and autoencoder models to a Ricean fading channel using the MMSE symbol estimation
    method. The number of adaptation samples per symbol is set to 20.
    python adapt_autoencoder_sim.py --n-bits 4 --cmf $channel_modelfile --amf $auto_modelfile
    --ta symbol_estimation_mmse --tc fading_ricean -o $output_dir --nad 20

    After running, inspect the performance metrics and plots that are saved to the output directory.
"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
import multiprocessing
from functools import partial
from scipy.stats import norm
from scipy.stats import t as tdist
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import MDN_base
from helpers.utils import (
    load_data_transceiver,
    simulate_channel_variations_gaussian,
    simulate_channel_variations_fading,
    simulate_channel_variations_ricean_fading,
    get_autoencoder_data_filename,
    generate_data_autoencoder,
    check_weights_equality,
    get_noise_stddev,
    estimate_stddev_awgn,
    calculate_fading_factor,
    calculate_ricean_fading_params,
    get_ber_channel_variations,
    save_perf_metrics,
    load_perf_metrics,
    configure_plot_axes,
    get_num_jobs
)
from helpers.MDN_classes import initialize_MDN_model
from helpers.autoencoder_classes import (
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
                        choices=['standard', 'symbol_estimation_mmse', 'symbol_estimation_map', 'adapt_generative'],
                        help='Type of autoencoder model trained.')
    parser.add_argument('--adaptation-objective', '--ao', choices=['log_posterior', 'log_likelihood'],
                        default='log_posterior', help='Type of objective function for adapting the channel model.')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials to repeat the adaptation runs. Averaged results are reported.')
    # Options for the simulated channel model from the target domain
    parser.add_argument('--type-channel', '--tc', choices=['AWGN', 'fading', 'fading_ricean', 'fading_rayleigh'],
                        default='fading_ricean', help='Type of simulated channel model to use')
    parser.add_argument('--snr-min', type=float, default=4.0, help='Minimum SNR of the channel in dB.')
    parser.add_argument('--snr-max', type=float, default=20.0, help='Maximum SNR of the channel in dB.')
    parser.add_argument('--snr-step', type=float, default=2.0, help='SNR is incremented by this step size.')
    parser.add_argument('--sigma-noise-measurement', '--sigma', type=float, default=0.09,
                        help='Noise standard deviation of the channel.')
    parser.add_argument('--lambda-fixed', type=float, default=-1.0,
                        help='Regularization hyper-parameter value. It is set automatically if this option is not used')
    # Data and output directory
    parser.add_argument('--data-dir', '-d', default='./Data', help='Directory for the saved data files')
    parser.add_argument('--output-dir', '-o', default='./outputs_adapt',
                        help='Directory for saving the output files, plots etc')
    # Model hyper-parameters
    parser.add_argument('--dim-encoding', '--de', type=int, default=2,
                        help='Dimension of the encoded symbols. Should be an even integer')
    parser.add_argument('--n-components', '--nc', type=int, default=5,
                        help='Number of components in the Gaussian mixture density network')
    parser.add_argument('--n-hidden', '--nh', type=int, default=100, help='Size of the hidden fully-connected layer')
    # Number of adaptation and test samples - specified per symbol
    parser.add_argument('--n-adapt-per-symbol', '--nad', type=int, default=20,
                        help='Number of target domain adaptation samples per constellation symbol')
    parser.add_argument('--n-test-per-symbol', '--nte', type=int, default=20000,
                        help='Number of test samples per constellation symbol')
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
    args.output_dir = os.path.abspath(args.output_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    # print("\nOutput directory: {}".format(args.output_dir))
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
        'data_file_autoenc': os.path.join(args.data_dir, get_autoencoder_data_filename(args.n_bits)),
        'adaptation_objective': args.adaptation_objective,
        'type_channel': args.type_channel,
        'sigma_noise_measurement': args.sigma_noise_measurement,
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs),
        'EbNodB_range': np.arange(args.snr_min, 1.001 * args.snr_max, args.snr_step),
        'EbNodB_min': SNR_MIN_RICEAN,
        'EbNo_min': 10. ** (SNR_MIN_RICEAN / 10.),
        'n_jobs': get_num_jobs(args.n_jobs),
        'n_trials': args.n_trials
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


def generate_data_adaptation(n_target_train, variation, EbNodB, unique_x, p_avg, config):
    # Genrate channel adaptation data of a specified type and SNR.
    variation = variation.lower()
    if variation in ('gaussian_noise', 'awgn'):
        sigma_noise = get_noise_stddev(EbNodB, rate=config['rate_comm'], E_avg_symbol=p_avg)
        print("\nSNR = {:g}dB. Noise standard deviation = {:.6f}".format(EbNodB, sigma_noise))
        stddev_target = [sigma_noise] * config['dim_encoding']
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_gaussian(
            None, unique_x, n_target_train, n_target_train, stddev_target, use_channel_output=False
        )
    elif variation == 'fading':
        EbNo = 10. ** (EbNodB / 10.)  # dB to ratio
        scale_fading = calculate_fading_factor(EbNo, config['sigma_noise_measurement'], config['rate_comm'], p_avg)
        print("\nSNR = {:g}dB. scale_fading = {:.4f}".format(EbNodB, scale_fading))
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_fading(
            unique_x, n_target_train, n_target_train, scale_fading, config['sigma_noise_measurement']
        )
    elif variation in ('fading_ricean', 'fading_rayleigh'):
        EbNo = 10. ** (EbNodB / 10.)  # dB to ratio
        EbNo_min = config['EbNo_min'] if (variation == 'fading_ricean') else EbNo
        nu, sigma_a, K = calculate_ricean_fading_params(
            EbNo, EbNo_min, config['sigma_noise_measurement'], config['rate_comm'], p_avg
        )
        print("\nSNR = {:g}dB. Ricean parameters: nu = {:.6f}, sigma_a = {:.6f}, K = {:.4f}dB".
              format(EbNodB, nu, sigma_a, K))
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_ricean_fading(
            unique_x, n_target_train, n_target_train, nu, sigma_a, config['sigma_noise_measurement']
        )
    else:
        raise ValueError("Invalid value '{}' for the input 'variation'".format(variation))

    target_domain_input = inputs_target_list[0]  # one-hot-coded inputs
    target_domain_x = x_target_list[0]  # encoded symbols
    target_domain_y = y_target_list[0]  # channel outputs
    return target_domain_input, target_domain_x, target_domain_y


def wrapper_autoencoder_adaptation(autoencoder_orig, mdn_model, n_adapt_per_symbol, x_test, labels_test, config,
                                   type_metric='log_loss', num_init=1, lambda_fixed=-1.0):
    # Valid options for `type_metric` are ['log_loss', 'error_rate', 'inverse_trans_log_like', 'inverse_trans_log_post']
    # Adapt the channel model and the autoencoder to the target channel data of varying SNR.
    # Distinct constellation symbols and their average power
    unique_x = autoencoder_orig.encoder(autoencoder_orig.inputs_unique)
    p_avg = tf.reduce_sum(unique_x ** 2) / unique_x.shape[0]
    # Probability of each unique symbol estimated from the training data
    prior_prob_x = autoencoder_orig.prior_prob_inputs

    # n_symbols = config['dim_encoding'] // 2
    EbNodB_range = config['EbNodB_range']
    if lambda_fixed < 0.:
        # Logarithmically-spaced lambda values between 10^-5 and 100
        lambda_values = np.logspace(-5, 2, num=8, base=10.)
    else:
        lambda_values = np.array([lambda_fixed])

    num_lambda = lambda_values.shape[0]
    num_snr_vals = EbNodB_range.shape[0]
    metric_train = np.zeros((num_lambda, num_snr_vals))
    lambda_best = np.zeros(num_snr_vals)
    metric_min_train = np.zeros(num_snr_vals)
    ber_min_test = np.zeros(num_snr_vals)
    n_target_train = n_adapt_per_symbol * config['mod_order']
    print("Number of target domain adaptation samples: {:d}".format(n_target_train))
    # Total wall-clock time taken for adaptation over all the SNR values
    t_adapt_cum = 0.
    weights_mdn = mdn_model.get_weights()
    for i in range(num_snr_vals):
        # Generate data for adapting the channel model from the specified type of variation and target SNR
        target_domain_input, target_domain_x, target_domain_y = generate_data_adaptation(
            n_target_train, config['type_channel'], EbNodB_range[i], unique_x, p_avg, config
        )
        t_adapt = 0.
        # Minimize the channel adaptation loss function for different lambda values, and evaluate the solution
        results_minimize = None
        if config['n_jobs'] > 1:
            t1 = time.time()
            # Call the function that creates the loss function and its gradient once. This performs the graph
            # compilation using `tf.function` for the first time, which can be slower than subsequent calls in parallel
            loss_func_proposed, loss_func_val_and_grad = get_loss_func_proposed(
                mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
                config['dim_encoding'], lambda_value=tf.constant(1.0, dtype=tf.float32),
                type_objec=config['adaptation_objective']
            )
            # Use `multiprocessing` to parallelize the minimization over lambda values.
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

        autoencoders_adapted = []
        for j in range(num_lambda):
            t1 = time.time()
            print("lambda = {:g}".format(lambda_values[j]))
            if results_minimize is None:
                psi_values, loss_final = minimize_channel_adaptation_bfgs(
                    mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, mdn_model.n_mixes,
                    config['dim_encoding'], lambda_values[j], type_objec=config['adaptation_objective'],
                    num_init=num_init
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
            metric_train[j, i] = calculate_metric_autoencoder(
                autoencoder, target_domain_input, target_domain_y, metric=type_metric
            )
            autoencoders_adapted.append(autoencoder)
            t_adapt += (time.time() - t1)

        t_adapt_cum += t_adapt
        # For the current SNR, find the `lambda` value that has the minimum metric on the target domain training set
        v_min = np.min(metric_train[:, i])
        ind_cand = np.where(metric_train[:, i] <= v_min)[0]
        if ind_cand.shape[0] == 1:
            j_star = ind_cand[0]
        else:
            # Break ties by choosing the solution corresponding to largest `lambda`
            lambda_cand = lambda_values[ind_cand]
            j_star = ind_cand[np.argmax(lambda_cand)]

        lambda_best[i] = lambda_values[j_star]
        metric_min_train[i] = metric_train[j_star, i]
        # Calculate BER of the selected best model on the test set
        ber_min_test[i] = get_ber_channel_variations(
            autoencoders_adapted[j_star], autoencoder_orig.channel, x_test, labels_test, config['rate_comm'],
            config['sigma_noise_measurement'], variation=config['type_channel'], EbNodB_range=EbNodB_range[i],
            EbNodB_min=config['EbNodB_min'], batch_size=10000
        )

    return ber_min_test, metric_min_train, lambda_best, t_adapt_cum / num_snr_vals


def summarize_results(EbNodB_range, ber_min_test, std_err_ber_min_test, ber_original, metric_min_train,
                      lambda_best, output_dir):
    # Write the performance metrics to a file and stdout
    fname = os.path.join(output_dir, 'ber_vs_snr_adaptation.csv')
    print("\nSelected 'lambda' value and corresponding BER vs. SNR:")
    line = ['Eb_No(dB)', 'lambda', 'metric_train', 'BER_no_adaptation', 'BER_with_adaptation', 'BER_with_adaptation_std_err']
    print('\t'.join(line))
    with open(fname, mode='w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(line)
        for i in range(ber_min_test.shape[0]):
            line = ['{:.2f}'.format(EbNodB_range[i]), '{:.4e}'.format(lambda_best[i]),
                    '{:.8e}'.format(metric_min_train[i]), '{:.8e}'.format(ber_original[i]),
                    '{:.8e}'.format(ber_min_test[i]), '{:.12e}'.format(std_err_ber_min_test[i])]
            print('\t'.join(line))
            cw.writerow(line)

    if len(EbNodB_range) <= 2:
        # skip the plot
        return

    # Plot BER vs. SNR of the original and the adapted autoencoder
    fname = os.path.join(output_dir, 'plot_ber_vs_snr_adaptation.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_vals = []
    ax.plot(EbNodB_range, ber_original, color=COLORS[0], marker=POINT_STYLES[0],
            linestyle='-', linewidth=0.75, label='Autoencoder original')
    y_vals.extend(ber_original)
    ax.plot(EbNodB_range, ber_min_test, color=COLORS[2], marker=POINT_STYLES[2],
            linestyle='-', linewidth=0.75, label='Autoencoder adapted')
    y_vals.extend(ber_min_test)
    configure_plot_axes(ax, y_vals)
    plt.xlabel('Signal-to-Noise ratio (dB)', fontsize=12)
    plt.ylabel('Block Error Rate', fontsize=12)
    # plt.title('BER vs. SNR with adaption', fontsize=13)
    plt.grid(True, axis='both', linestyle='dashed', linewidth=0.75, alpha=1.)
    plt.legend(loc='best', ncol=1)
    fig.tight_layout()
    fig.savefig(fname, dpi=600, bbox_inches='tight')
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

    # Load the saved channel model weights into a newly initialized channel model
    mdn_model = initialize_MDN_model(
        config_basic['n_components'], config_basic['dim_encoding'], config_basic['n_hidden']
    )
    mdn_model.load_weights(args.channel_model_file).expect_partial()

    # Load the saved autoencoder model weights into a newly initialized autoencoder model
    n_train = config_basic['mod_order'] * args.n_test_per_symbol    # value is not important
    autoencoder = initialize_autoencoder(config_basic['type_autoencoder'], mdn_model, config_basic, config_optimizer,
                                         n_train, temperature=CONFIG_ANNEAL['temp_final'])
    autoencoder.load_weights(args.autoencoder_model_file).expect_partial()

    # Data for testing the autoencoder performance
    _, _, x_test, labels_test = generate_data_autoencoder(
        config_basic['data_file_autoenc'], config_basic['mod_order'], 10, args.n_test_per_symbol
    )
    # Calculate BER vs SNR of the autoencoder without adaptation on the test data
    ber_original = get_ber_channel_variations(
        autoencoder, autoencoder.channel, x_test, labels_test, config_basic['rate_comm'],
        config_basic['sigma_noise_measurement'], variation=config_basic['type_channel'],
        EbNodB_range=config_basic['EbNodB_range'], EbNodB_min=config_basic['EbNodB_min'], batch_size=10000
    )

    # Adapt the channel model and the autoencoder to the target channel data of varying SNR.
    # Repeat over a specified number of trials and report the averaged results
    ber_min_test_list = []
    avg_ber_min_test = None
    avg_metric_min_train = None
    avg_lambda_best = None
    avg_time_avg_adapt = 0.
    for t in range(config_basic['n_trials']):
        print("\n\nRunning trial {:d}".format(t + 1))
        ber_min_test, metric_min_train, lambda_best, time_avg_adapt = wrapper_autoencoder_adaptation(
            autoencoder, mdn_model, args.n_adapt_per_symbol, x_test, labels_test, config_basic,
            lambda_fixed=args.lambda_fixed
        )
        ber_min_test_list.append(ber_min_test)
        avg_time_avg_adapt += time_avg_adapt
        if t > 0:
            avg_ber_min_test += ber_min_test
            avg_metric_min_train += metric_min_train
            avg_lambda_best += lambda_best
        else:
            avg_ber_min_test = ber_min_test
            avg_metric_min_train = metric_min_train
            avg_lambda_best = lambda_best

    avg_time_avg_adapt /= config_basic['n_trials']
    if config_basic['n_trials'] > 1:
        avg_ber_min_test /= config_basic['n_trials']
        avg_metric_min_train /= config_basic['n_trials']
        avg_lambda_best /= config_basic['n_trials']

    # Standard error of the mean for reporting the confidence interval
    conf = 0.95
    z_val = norm.ppf((1 + conf) / 2.)
    t_val = tdist.ppf((1 + conf) / 2., df=(config_basic['n_trials'] - 1))
    if config_basic['n_trials'] >= 100:
        c = z_val / np.sqrt(config_basic['n_trials'])
    else:
        # use the t-distribution for small `n_trials`
        c = t_val / np.sqrt(config_basic['n_trials'])

    std_err_ber_min_test = c * np.std(np.array(ber_min_test_list), axis=0, ddof=1)

    print("\nAverage time taken for adapting the autoencoder: {:g} seconds".format(avg_time_avg_adapt))
    fname = os.path.join(args.output_dir, 'time_adaptation.csv')
    with open(fname, 'w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(['n_adapt_samples', 'time_avg_adaptation'])
        cw.writerow(['{:d}'.format(args.n_adapt_per_symbol), '{:.4f}'.format(avg_time_avg_adapt)])

    # Summarize and plot the results of adaptation
    summarize_results(config_basic['EbNodB_range'], avg_ber_min_test, std_err_ber_min_test, ber_original,
                      avg_metric_min_train, avg_lambda_best, args.output_dir)


if __name__ == '__main__':
    # This is needed to prevent `multiprocessing` calls from getting hung on MacOS
    multiprocessing.set_start_method("spawn")
    main()
