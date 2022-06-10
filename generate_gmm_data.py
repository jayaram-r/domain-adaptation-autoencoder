"""
Script for generating training, adaptation, and test data from different conditional Gaussian mixture densities.
The parameters of the GMMs used for generating the adaptation and test data can be different from that of the GMMs
used for generating the training data.
It is possible to include a random phase shift that is uniformly sampled from an interval `[-s, s]`. The value of `s`
can be specified via the argument `--max-phase-shift` or `--mps`.

USAGE example:
--nb 4  : 4 bits per symbol
--ns 3  : 3 components in the source GMM
--nt 3  : 3 components in the target GMM
--dtr   : path to the training data directory
--da    : path to the adaptation data directory
--dte   : path to the test data directory
--constellation-file or --cof : path to the numpy file with the constellation symbols. If not specified, by default,
                                an M-QAM constellation is used.
--mps   : value of the maximum phase shift. Set to 0 by default.


base_dir='/Data/GMM_expt'
python generate_gmm_data.py --nb 4 --ns 3 --nt 3 --dtr "$base_dir/data_train" --da "$base_dir/data_adapt" --dte "$base_dir/data_test" -o "$base_dir/GMM_expt/summary" --seed 123

Add the option `--no-change` to generate the adaptation and target datasets from the same GMMs.
"""
import sys
import argparse
import os
import csv
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import cdist

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from helpers.standard_modulations import StandardQAM
from helpers.utils import (
    convert_to_complex_ndarray,
    get_samples_per_symbol,
    generate_data_gmm
)
from helpers.constants import *
# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-bits', '--nb', type=int, required=True, help='number of bits per symbol')
    parser.add_argument('--n-comp-src', '--ns', type=int, default=5,
                        help='Number of components in the source Gaussian mixture')
    parser.add_argument('--n-comp-tar', '--nt', type=int, default=5,
                        help='Number of components in the target Gaussian mixture')
    parser.add_argument('--constellation-file', '--cof', default='',
                        help='Path to the numpy file with the constellation symbols. If not specified, by default, '
                             'an M-QAM constellation is used.')
    parser.add_argument('--max-phase-shift', '--mps', type=float, default=0.,
                        help="Maximum phase shift magnitude 's' in degrees. If 's > 0', then the phase shift is "
                             "uniformly distributed in [-s, s]")
    parser.add_argument('--params-gmm', default='',
                        help='Path to a pickle file with the GMM parameters per symbol. If not specified, the GMM '
                             'parameters will be generated randomly.')
    parser.add_argument('--constellation-file-init', '--cof-init', default='',
                        help='Path to the numpy file with the initial constellation symbols. If not specified, by '
                             'default, an M-QAM constellation is used.')
    # Number of samples
    parser.add_argument('--n-train', '--ntr', type=int, default=25000, help='Number of training samples')
    parser.add_argument('--n-adapt-per-symbol', '--nad', default='1,2,4,6,8,10',
                        help='Number of target domain adaptation samples per constellation symbol. Specified as a '
                             'comma-separated string of values, e.g.: 1,2,4,6,8,10')
    parser.add_argument('--n-test', '--nte', type=int, default=25000, help='Number of test samples')
    parser.add_argument('--no-change', action='store_true', default=False,
                        help="Use this option to generate the target domain data from the same GMM as the source")
    # Data and output directories
    parser.add_argument('--data-dir-train', '--dtr', default='./data_train',
                        help='Directory for saving the training data.')
    parser.add_argument('--data-dir-adapt', '--da', default='./data_adapt',
                        help='Directory for saving the adaptation data.')
    parser.add_argument('--data-dir-test', '--dte', default='./data_test',
                        help='Directory for saving the test data.')
    parser.add_argument('--output-dir', '-o', default='./summary', help='Output directory')
    parser.add_argument('--avg-power-symbols', '--aps', type=float, default=1.,
                        help='Average power of the symbols in the constellation (maximum value)')
    parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for the random number generators')
    args = parser.parse_args()

    # Check if directories exist and create them
    args.data_dir_train = os.path.abspath(args.data_dir_train)
    if not os.path.isdir(args.data_dir_train):
        os.makedirs(args.data_dir_train)

    args.data_dir_test = os.path.abspath(args.data_dir_test)
    if not os.path.isdir(args.data_dir_test):
        os.makedirs(args.data_dir_test)

    args.data_dir_adapt = os.path.abspath(args.data_dir_adapt)
    args.output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.constellation_file:
        args.constellation_file = os.path.abspath(args.constellation_file)

    return args


def sample_adaptation_data(x_data, y_data, labels, n_adapt_per_symbol, n_symb):
    # Create the adaptation datasets of different sizes by randomly sampling from the retraining data
    n_arrays = len(n_adapt_per_symbol)
    x_adapt = [None for _ in range(n_arrays)]
    y_adapt = [None for _ in range(n_arrays)]
    labels_adapt = [None for _ in range(n_arrays)]
    for m in range(n_symb):
        # Samples from class `m` randomly shuffled
        ind_curr = np.random.permutation(np.where(labels[:, 0] == m)[0])
        for i in range(n_arrays):
            ind_select = ind_curr[:n_adapt_per_symbol[i]]
            x_select = x_data[ind_select, :]
            y_select = y_data[ind_select, :]
            labels_select = labels[ind_select, :]

            if x_adapt[i] is None:
                x_adapt[i] = x_select
            else:
                x_adapt[i] = np.vstack([x_adapt[i], x_select])

            if y_adapt[i] is None:
                y_adapt[i] = y_select
            else:
                y_adapt[i] = np.vstack([y_adapt[i], y_select])

            if labels_adapt[i] is None:
                labels_adapt[i] = labels_select
            else:
                labels_adapt[i] = np.vstack([labels_adapt[i], labels_select])

    # List of data arrays, one for each of the adaptation sample sizes in `n_adapt_per_symbol`
    return x_adapt, y_adapt, labels_adapt


def scatter_plot_data(x_uniq, y_data_tr, labels_tr, y_data_ad, labels_ad, y_data_te,
                      labels_te, plot_filename, max_samp_per_symb=500):
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
    for m in range(x_uniq.shape[0]):
        # Training data from symbol `m`
        ind = np.where(labels_tr[:, 0] == m)[0]
        n_samp = ind.shape[0]
        if n_samp > max_samp_per_symb:
            ind = np.random.permutation(ind)[:max_samp_per_symb]

        axes[0].scatter(y_data_tr[ind, 0], y_data_tr[ind, 1], alpha=0.5, c=COLORS[m])

        # Adaptation data from symbol `m`
        ind = np.where(labels_ad[:, 0] == m)[0]
        n_samp = ind.shape[0]
        if n_samp > max_samp_per_symb:
            ind = np.random.permutation(ind)[:max_samp_per_symb]

        axes[1].scatter(y_data_ad[ind, 0], y_data_ad[ind, 1], alpha=0.5, c=COLORS[m])

        # Test data from symbol `m`
        ind = np.where(labels_te[:, 0] == m)[0]
        n_samp = ind.shape[0]
        if n_samp > max_samp_per_symb:
            ind = np.random.permutation(ind)[:max_samp_per_symb]

        axes[2].scatter(y_data_te[ind, 0], y_data_te[ind, 1], alpha=0.5, c=COLORS[m])

    for i in (0, 1, 2):
        axes[i].scatter(x_uniq[:, 0], x_uniq[:, 1], alpha=1, c='k', marker='+')

    axes[0].set_title("Training Data", fontsize=9, fontweight='normal')
    axes[1].set_title("Adaptation Data", fontsize=9, fontweight='normal')
    axes[2].set_title("Test Data", fontsize=9, fontweight='normal')
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def calc_loglikelihood(gmm_distr, y_data, labels):
    # Conditional log-likelihood of the data under the Gaussian mixtures
    loglike = None
    for m in range(len(gmm_distr)):
        ind = np.where(labels[:, 0] == m)[0]
        loglike_curr = gmm_distr[m].log_prob(y_data[ind, :]).numpy()
        if loglike is None:
            loglike = loglike_curr
        else:
            loglike = np.concatenate((loglike, loglike_curr), axis=0)

    return np.mean(loglike)


def save_loglikehood(gmm_distr_src, gmm_distr_tar, y_data_tr, labels_tr, y_data_ad, labels_ad, y_data_te, labels_te,
                     output_file):
    # Log-likelihood of the training, adaptation, and test data under the source GMM
    ll_src_train = calc_loglikelihood(gmm_distr_src, y_data_tr, labels_tr)
    ll_src_adapt = calc_loglikelihood(gmm_distr_src, y_data_ad, labels_ad)
    ll_src_test = calc_loglikelihood(gmm_distr_src, y_data_te, labels_te)
    # Log-likelihood of the train, adaptation, and test data under the target GMM
    ll_tar_train = calc_loglikelihood(gmm_distr_tar, y_data_tr, labels_tr)
    ll_tar_adapt = calc_loglikelihood(gmm_distr_tar, y_data_ad, labels_ad)
    ll_tar_test = calc_loglikelihood(gmm_distr_tar, y_data_te, labels_te)

    with open(output_file, mode='w') as fp:
        cw = csv.writer(fp, delimiter='\t', lineterminator='\n')
        cw.writerow(['model', 'data_train', 'data_adapt', 'data_test'])
        cw.writerow(['Source GMM, loglike'] + ['{:.4f}'.format(v) for v in (ll_src_train, ll_src_adapt, ll_src_test)])
        cw.writerow(['Target GMM, loglike'] + ['{:.4f}'.format(v) for v in (ll_tar_train, ll_tar_adapt, ll_tar_test)])

        val_tr = (ll_src_train - ll_tar_train) / abs(ll_tar_train)
        val_ad = (ll_tar_adapt - ll_src_adapt) / abs(ll_src_adapt)
        val_te = (ll_tar_test - ll_src_test) / abs(ll_src_test)
        cw.writerow(['Rel. change loglike'] + ['{:.4f}'.format(v) for v in (val_tr, val_ad, val_te)])


def save_gmm_params(params_gmm_src, params_gmm_tar, output_dir):
    fname = os.path.join(output_dir, 'params_gmm_source.pkl')
    with open(fname, 'wb') as fp:
        pickle.dump(params_gmm_src, fp)

    fname = os.path.join(output_dir, 'params_gmm_target.pkl')
    with open(fname, 'wb') as fp:
        pickle.dump(params_gmm_tar, fp)


def save_data_files(x_data, y_data, labels, n_symb, data_dir):
    # Save the data arrays into mat files in the required complex format
    mdict = {'tx_send': convert_to_complex_ndarray(x_data)}
    sio.savemat(os.path.join(data_dir, TX_DATA_BASENAME), mdict)

    mdict = {'rx_channel': convert_to_complex_ndarray(y_data)}
    sio.savemat(os.path.join(data_dir, RX_DATA_BASENAME), mdict)

    # Convert from integer to one-hot-coded labels
    labels_one_hot = tf.one_hot(labels.ravel(), n_symb).numpy()
    mdict = {'labels': labels_one_hot}
    sio.savemat(os.path.join(data_dir, LABELS_BASENAME), mdict)


def remove_excess_components(params_gmm, n_comp_req):
    # Set the component priors of excess components to 0 and renormalize the priors.
    # `params_gmm` is modified in place in the function.
    for m in range(len(params_gmm)):
        priors = params_gmm[m]['priors']
        n_comp = priors.shape[0]
        if n_comp <= n_comp_req:
            continue

        # Remove the components with the smallest priors
        ind = np.argsort(priors)
        ind_remove = ind[:(n_comp - n_comp_req)]
        priors[ind_remove] = 1e-8
        params_gmm[m]['priors'] = (1. / np.sum(priors)) * priors


def generate_gmm_params(constellation, n_comp):
    # Generate the parameters of the source and target Gaussian mixtures (conditioned on each symbol)
    n_symb, n_dim = constellation.shape
    # Pairwise distance between the constellation symbols
    dist_mat = cdist(constellation, constellation, metric='euclidean')
    np.fill_diagonal(dist_mat, -1.)
    scale_sigma = (0.2, 1)
    params_gmm_src = []
    params_gmm_tar = []
    for m in range(n_symb):
        # GMM conditioned on the input symbol `m`
        x = constellation[m, :]
        # Generate the component weights (priors) at random from the uniform distribution on [0.05, 0.95]
        v = np.random.uniform(low=0.05, high=0.95, size=n_comp)
        priors_src = (1. / np.sum(v)) * v
        v = np.random.uniform(low=0.05, high=0.95, size=n_comp)
        priors_tar = (1. / np.sum(v)) * v

        # Distance from the current constellation symbol to the remaining ones
        dist_curr = dist_mat[m, :]
        d_min = np.min(dist_curr[dist_curr > 0.])
        # Spherical covariance matrix used to sample the component means.
        # Setting `2 * sigma_0 = d_min / k`. Try k = 2, 3, 4
        # sigma_0 = d_min / 6.
        sigma_0 = d_min / 4.    # to get a higher overlap between the classes
        cov_0 = (sigma_0 ** 2) * np.eye(n_dim)

        # Generate the component means randomly from a spherical Gaussian centered on `x` with a covariance `cov_0`
        means_src = np.random.multivariate_normal(x, cov_0, size=n_comp)
        means_tar = np.random.multivariate_normal(x, cov_0, size=n_comp)
        # Generate the diagonal elements of the component covariances. Each element is uniformly sampled from the
        # interval `[scale_sigma * sigma_0, sigma_0]`
        covars_src = np.random.uniform(low=(scale_sigma[0] * sigma_0), high=(scale_sigma[1] * sigma_0),
                                       size=(n_comp, n_dim))
        covars_tar = np.random.uniform(low=(scale_sigma[0] * sigma_0), high=(scale_sigma[1] * sigma_0),
                                       size=(n_comp, n_dim))
        # `means_src` and `covars_src` are numpy arrays of shape `(n_comp, n_dim)`.
        # `priors_src` is a numpy array of shape `(n_comp, )`
        params_gmm_src.append(
            {'means': means_src, 'covars': covars_src, 'priors': priors_src}
        )
        params_gmm_tar.append(
            {'means': means_tar, 'covars': covars_tar, 'priors': priors_tar}
        )

    return params_gmm_src, params_gmm_tar


def generate_source_target_gmms(n_comp_src, n_comp_tar, constellation, no_change):
    # Generate the parameters of the source and target Gaussian mixtures (conditioned on each symbol)
    n_comp = max(n_comp_src, n_comp_tar)
    params_gmm_src, params_gmm_tar = generate_gmm_params(constellation, n_comp)
    # Remove any excess component(s) from one of the Gaussian mixtures
    if n_comp > n_comp_src:
        remove_excess_components(params_gmm_src, n_comp_src)

    if no_change:
        params_gmm_tar = params_gmm_src
    else:
        if n_comp > n_comp_tar:
            remove_excess_components(params_gmm_tar, n_comp_tar)

    return params_gmm_src, params_gmm_tar


def load_shared_gmm(params_gmm, constellation, constellation_init):
    # For the Gaussian mixture corresponding to each symbol in the constellation, subtract off the old symbol and
    # add the new symbol to the component means. The component priors and covariances are kept the same.
    params_gmm_new = []
    for i in range(constellation_init.shape[0]):
        tmp_dict = {k: v for k, v in params_gmm[i].items()}
        tmp_dict['means'] = params_gmm[i]['means'] - constellation_init[i, :] + constellation[i, :]
        params_gmm_new.append(tmp_dict)

    return params_gmm_new


def main():
    # Read the command line inputs
    args = parse_inputs()
    # Seed the random number generators
    np.random.seed(args.seed)
    if tf.__version__[0] == '2':
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_random_seed(args.seed)

    # Read the number of adaptation samples from the argument string
    n_adapt_per_symbol = [int(a.strip(' ')) for a in args.n_adapt_per_symbol.split(',')]

    if args.constellation_file:
        constellation = np.load(args.constellation_file)
    else:
        # Constellation of standard M-QAM
        constellation = StandardQAM(args.n_bits, avg_power=args.avg_power_symbols).constellation

    n_symb = constellation.shape[0]
    if args.params_gmm:
        # Load the GMM parameters file
        with open(args.params_gmm, 'rb') as fp:
            params_gmm = pickle.load(fp)

        if args.constellation_file_init:
            constellation_init = np.load(args.constellation_file_init)
        else:
            # Constellation of standard M-QAM
            constellation_init = StandardQAM(args.n_bits, avg_power=args.avg_power_symbols).constellation

        params_gmm_src = load_shared_gmm(params_gmm, constellation, constellation_init)
        params_gmm_tar = params_gmm_src
    else:
        # Generate the parameters of the source and target Gaussian mixtures (conditioned on each symbol)
        params_gmm_src, params_gmm_tar = generate_source_target_gmms(args.n_comp_src, args.n_comp_tar, constellation,
                                                                     args.no_change)

    # Generate data from the source GMM for training
    x_data_tr, y_data_tr, labels_tr, gmm_distr_src = generate_data_gmm(
        params_gmm_src, constellation, args.n_train, max_phase_shift=args.max_phase_shift
    )
    # Generate data from the target GMM for testing
    x_data_te, y_data_te, labels_te, gmm_distr_tar = generate_data_gmm(
        params_gmm_tar, constellation, args.n_test, max_phase_shift=args.max_phase_shift
    )
    # Generate data from the target GMM for retraining on the target domain
    n_max = max(args.n_train, n_symb * np.max(n_adapt_per_symbol))
    x_data_retrain, y_data_retrain, labels_retrain, _ = generate_data_gmm(
        params_gmm_tar, constellation, n_max, max_phase_shift=args.max_phase_shift
    )
    # Create the adaptation datasets of different sizes by randomly sampling from the retraining data
    x_data_ad, y_data_ad, labels_ad = sample_adaptation_data(x_data_retrain, y_data_retrain, labels_retrain,
                                                             n_adapt_per_symbol, n_symb)
    # Save the GMM parameters to a pickle file
    save_gmm_params(params_gmm_src, params_gmm_tar, args.output_dir)

    # Save the training, retraining, and test data to mat files
    save_data_files(x_data_tr, y_data_tr, labels_tr, n_symb, args.data_dir_train)
    save_data_files(x_data_te, y_data_te, labels_te, n_symb, args.data_dir_test)
    # Save the data for retraining the MDN model on the target domain
    direc = os.path.join(os.path.dirname(args.data_dir_train), 'data_retrain_target')
    if not os.path.isdir(direc):
        os.makedirs(direc)
    save_data_files(x_data_retrain, y_data_retrain, labels_retrain, n_symb, direc)

    for i, n_adapt in enumerate(n_adapt_per_symbol):
        # Number of adaptation samples per class is added as suffix to the data directory name
        direc = '{}_{:d}'.format(args.data_dir_adapt, n_adapt)
        if not os.path.isdir(direc):
            os.makedirs(direc)
        # Save the adaptation data to mat files
        save_data_files(x_data_ad[i], y_data_ad[i], labels_ad[i], n_symb, direc)

        # Report the log-likelihood values
        fname = os.path.join(args.output_dir, 'log_likelihood_{:d}.csv'.format(n_adapt))
        save_loglikehood(gmm_distr_src, gmm_distr_tar, y_data_tr, labels_tr, y_data_ad[i], labels_ad[i],
                         y_data_te, labels_te, fname)
        # Plot the data
        plot_filename = os.path.join(args.output_dir, 'plot_data_{:d}.png'.format(n_adapt))
        scatter_plot_data(constellation, y_data_tr, labels_tr, y_data_ad[i], labels_ad[i], y_data_te, labels_te,
                          plot_filename)


if __name__ == '__main__':
    main()
