"""
Main script for the baseline method of fine-tuning the MDN (channel) model and the autoencoder on simulated channel
variations. Supports fine-tuning the entire MDN or only the last layer.
For a description of the command line inputs, type:
`python finetune_autoencoder_sim.py -h`

Main command line options:
--n-bits or --nb: number of bits per symbol (REQUIRED).
--channel-model-file or --cmf: path to the channel model file or directory (REQUIRED).
--autoencoder-model-file or --amf: path to the autoencoder model file or directory (REQUIRED).
--last :  use this  option to specify that only  the last layer of MDN should be finetuned.
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

"""
import sys
import argparse
import os
import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
from helpers.MDN_classes import (
    MDN_model,
    initialize_MDN_model
)
from helpers.autoencoder_classes import (
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
    parser.add_argument('--n-train-per-symbol', '--ntr', type=int, default=10000,
                        help='Number of training samples per constellation symbol')
    parser.add_argument('--n-test-per-symbol', '--nte', type=int, default=20000,
                        help='Number of test samples per constellation symbol')
    # Optimizer configuration
    parser.add_argument('--batch-size', '--bs', type=int, default=128, help='Batch size for optimization')
    parser.add_argument('--n-epochs', '--ne', type=int, default=20, help='Number of optimization epochs')
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
    args = parser.parse_args()

    args.channel_model_file = os.path.abspath(args.channel_model_file)
    args.autoencoder_model_file = os.path.abspath(args.autoencoder_model_file)
    args.output_dir = os.path.abspath(args.output_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
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
        'type_autoencoder': 'standard',
        'data_file_autoenc': os.path.join(args.data_dir, get_autoencoder_data_filename(args.n_bits)),
        'type_channel': args.type_channel,
        'sigma_noise_measurement': args.sigma_noise_measurement,
        'avg_power_symbols': args.avg_power_symbols,
        'l2_reg_strength': args.l2_reg_strength,
        'scale_outputs': (not args.disable_scale_outputs),
        'EbNodB_range': np.arange(args.snr_min, 1.001 * args.snr_max, args.snr_step),
        'EbNodB_min': SNR_MIN_RICEAN,
        'EbNo_min': 10. ** (SNR_MIN_RICEAN / 10.),
        'n_jobs': 1,
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

    # Sanity check
    if not check_weights_equality(autoencoder.encoder.get_weights(), wts_encoder):
        print("\nERROR: weights of the encoder have changed.")
    if not check_weights_equality(autoencoder.channel.get_weights(), wts_mdn):
        print("\nERROR: weights of the MDN (channel) have changed.")

    return autoencoder, np.array(train_loss), np.array(val_loss)


def finetune_mdn_baseline(mdn_model, target_domain_x, target_domain_y, n_epochs=200):
    # Adaptation of the MDN using fine-tuning with a warm start based on the source domain parameters
    print("\nFine-tuning the MDN model on target domain data:")
    # Weights are copied from the trained MDN, but a new optimizer is initialized
    mdn_model_new = initialize_MDN_model(mdn_model.n_mixes, mdn_model.n_dims, mdn_model.n_hidden)
    mdn_model_new.set_weights(mdn_model.get_weights())
    mdn_model_new.summary()

    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    batch_size = max(10, int(np.ceil(0.01 * target_domain_x.shape[0])))
    hist = mdn_model_new.fit(x=target_domain_x, y=target_domain_y, batch_size=batch_size, epochs=n_epochs)

    return mdn_model_new


def finetune_mdn_baseline_last_layer(mdn_model, target_domain_x, target_domain_y, n_epochs=200):
    # Adaptation of the MDN's last layer using fine-tuning with a warm start based on the source domain parameters
    print("\nFine-tuning the last layer of the MDN model on target domain data:")
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

    # Verify that the weights of all but the final layer have not changed
    flag_eq = check_weights_equality(mdn_model_new.get_weights(), weights_mdn, n_arrays=(2 * (n_layers - 1)))
    if not flag_eq:
        print("ERROR: weights of the adapted MDN layers are different from expected.")

    return mdn_model_new


def wrapper_finetune_autoencoder(mdn_model, autoencoder_model_file, unique_x, x_train, x_val, x_test, labels_test,
                                 n_adapt_per_symbol, config, config_optimizer, model_dir, last_layer=False):
    # Adapt the channel model and the autoencoder to the target channel data of varying SNR
    p_avg = tf.reduce_sum(unique_x ** 2) / unique_x.shape[0]
    EbNodB_range = config['EbNodB_range']
    num_snr_vals = EbNodB_range.shape[0]
    ber_min_test = np.zeros(num_snr_vals)
    n_target_train = n_adapt_per_symbol * config['mod_order']
    print("Number of target domain adaptation samples: {:d}".format(n_target_train))
    t_adapt = 0.
    for i in range(num_snr_vals):
        # Generate data for adapting the channel model from the specified type of variation and target SNR
        target_domain_input, target_domain_x, target_domain_y = generate_data_adaptation(
            n_target_train, config['type_channel'], EbNodB_range[i], unique_x, p_avg, config
        )
        t1 = time.time()
        # Fine-tune the MDN model on the target domain data
        if last_layer:
            mdn_model_new = finetune_mdn_baseline_last_layer(mdn_model, target_domain_x, target_domain_y)
        else:
            mdn_model_new = finetune_mdn_baseline(mdn_model, target_domain_x, target_domain_y)

        weights_mdn_new = mdn_model_new.get_weights()
        # Load the saved autoencoder model weights into a newly initialized autoencoder model.
        # The parameters of the encoder and MDN (channel) networks are frozen
        n_train = config['mod_order'] * 1000  # value is not important
        autoencoder = initialize_autoencoder(config['type_autoencoder'], mdn_model_new, config, config_optimizer,
                                             n_train, temperature=CONFIG_ANNEAL['temp_final'], freeze_encoder=True)
        autoencoder.load_weights(autoencoder_model_file).expect_partial()

        # Update the channel model of the autoencoder with the weights of the adapted MDN
        autoencoder.channel.set_weights(weights_mdn_new)
        # Train the autoencoder based on the fine-tuned MDN
        autoencoder, train_loss, val_loss = wrapper_train_autoencoder(autoencoder, x_train, x_val,
                                                                      config_optimizer, model_dir)
        t_adapt += (time.time() - t1)
        # Calculate BER of the fine-tuned autoencoder on the test set
        ber_min_test[i] = get_ber_channel_variations(
            autoencoder, mdn_model, x_test, labels_test, config['rate_comm'],
            config['sigma_noise_measurement'], variation=config['type_channel'], EbNodB_range=EbNodB_range[i],
            EbNodB_min=config['EbNodB_min'], batch_size=10000
        )

    return ber_min_test, t_adapt / num_snr_vals


def summarize_results(EbNodB_range, ber_min_test, std_err_ber_min_test, ber_original, output_dir):
    # Write the performance metrics to a file and stdout
    fname = os.path.join(output_dir, 'ber_vs_snr_adaptation.csv')
    print("\nBER vs. SNR of the original and adapted autoencoder:")
    line = ['Eb_No(dB)', 'BER_no_adaptation', 'BER_with_adaptation', 'BER_with_adaptation_std_err']
    print('\t'.join(line))
    with open(fname, mode='w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(line)
        for i in range(ber_min_test.shape[0]):
            line = ['{:.2f}'.format(EbNodB_range[i]), '{:.8e}'.format(ber_original[i]),
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

    # Distinct constellation symbols and their average power
    unique_x = autoencoder.encoder(autoencoder.inputs_unique)
    # Train, validation, and test data for the autoencoder
    x_train, x_val, x_test, labels_test = generate_data_autoencoder(
        config_basic['data_file_autoenc'], config_basic['mod_order'], args.n_train_per_symbol, args.n_test_per_symbol
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
    avg_time_avg_adapt = 0.
    for t in range(config_basic['n_trials']):
        print("\n\nRunning trial {:d}".format(t + 1))
        ber_min_test, time_avg_adapt = wrapper_finetune_autoencoder(
            mdn_model, args.autoencoder_model_file, unique_x, x_train, x_val, x_test, labels_test,
            args.n_adapt_per_symbol, config_basic, config_optimizer, args.output_dir, last_layer=args.last
        )
        ber_min_test_list.append(ber_min_test)
        avg_time_avg_adapt += time_avg_adapt
        if t > 0:
            avg_ber_min_test += ber_min_test
        else:
            avg_ber_min_test = ber_min_test

    avg_time_avg_adapt /= config_basic['n_trials']
    avg_ber_min_test /= config_basic['n_trials']

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
                      args.output_dir)


if __name__ == '__main__':
    # This is needed to prevent `multiprocessing` calls from getting hung on MacOS
    multiprocessing.set_start_method("spawn")
    main()
