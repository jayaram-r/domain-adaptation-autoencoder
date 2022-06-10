# Collection of utility functions for data generation, reading files, plotting etc
import csv
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy.spatial.distance import cdist
from scipy.stats import rice
import math
import multiprocessing
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .context import *
import MDN_base
from helpers.constants import *


def load_data_transceiver(tx_file, rx_file, shuffle=False):
    # mat_contents = sio.loadmat(tx_file)
    # x_data = mat_contents['tx_send']
    x_data = loadmat_helper(tx_file)
    nx = len(x_data)
    # mat_contents = sio.loadmat(rx_file)
    # y_data = mat_contents['rx_channel']
    y_data = loadmat_helper(rx_file)
    ny = len(y_data)
    assert nx == ny, "Length of x and y are different"
    n_samp = nx

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

    return x_data1.astype(np.float32), y_data1.astype(np.float32)


def loadmat_helper(fname):
    # Simple utility to find the right data key in a mat file and return the data array
    mat_contents = sio.loadmat(fname)
    data_key = []
    for i in mat_contents.keys():
        if i in ('__header__', '__version__', '__globals__'):
            continue
        else:
            data_key.append(i)

    if len(data_key) == 1:
        return mat_contents[data_key[0]]
    else:
        s = ', '.join(data_key)
        raise ValueError("More than one data key found in the mat file '{}': {}".format(fname, s))


def save_perf_metrics(snr_list, ber_list, filename):
    dir_name, _ = os.path.split(filename)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with open(filename, mode='w') as fp:
        cw = csv.writer(fp, delimiter=',', lineterminator='\n')
        cw.writerow(snr_list)
        cw.writerow(['{:.8e}'.format(v) for v in ber_list])


def load_perf_metrics(filename):
    with open(filename, mode='r') as fp:
        cr = csv.reader(fp, delimiter=',')
        snr_list = next(cr)
        ber_list = next(cr)

    return list(map(float, snr_list)), list(map(float, ber_list))


def load_adaptation_metrics(filename):
    snr = []
    ber_no_adapt = []
    ber_adapt = []
    with open(filename, 'r') as fp:
        cr = csv.reader(fp, delimiter=',')
        for i, row in enumerate(cr):
            if i == 0:
                # header
                continue

            snr.append(float(row[0]))
            ber_no_adapt.append(float(row[3]))
            ber_adapt.append(float(row[4]))

    return snr, ber_no_adapt, ber_adapt


def generate_channel_data_real(tx_data_file, rx_data_file, config):
    x_data, y_data = load_data_transceiver(tx_data_file, rx_data_file, shuffle=True)
    # Maximum likelihood estimate of the measurement noise standard deviation
    sigma_noise = estimate_stddev_awgn(x_data, y_data)
    config['sigma_noise_measurement'] = sigma_noise
    # Unique QAM symbols
    x_unique_qam = tf.convert_to_tensor(np.unique(x_data, axis=0), dtype=DTYPE_TF)
    n_uniq = x_unique_qam.shape[0]
    assert n_uniq == config['mod_order'], ("Number of unique channel inputs {:d} is different from the expected "
                                           "value {:d}".format(n_uniq, config['mod_order']))
    return x_data, y_data, x_unique_qam


def generate_channel_data_simulated(type_channel, EbNodB, n_samp, config, constellation):
    # Generate data from standard channel models.
    # Average power of the symbols
    E_avg_qam = np.sum(constellation ** 2) / constellation.shape[0]
    EbNo = 10. ** (EbNodB / 10.)  # Channel SNR in ratio
    n_samp_val = int(0.1 * n_samp)
    if type_channel in ('fading_ricean', 'fading_rayleigh'):
        EbNo_min = EbNo if (type_channel == 'fading_rayleigh') else config['EbNo_min']
        nu, sigma_a, K = calculate_ricean_fading_params(
            EbNo, EbNo_min, config['sigma_noise_measurement'], config['rate_comm'], E_avg_qam
        )
        print("Channel SNR = {:g}dB. Ricean fading parameters: nu = {:.6f}, sigma_a = {:.6f}, K = {:.4f}dB".
              format(EbNodB, nu, sigma_a, K))
        # Generate data
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_ricean_fading(
            constellation, n_samp, n_samp_val, nu, sigma_a, config['sigma_noise_measurement']
        )

    elif type_channel == 'fading':
        scale_fading = calculate_fading_factor(EbNo, config['sigma_noise_measurement'], config['rate_comm'], E_avg_qam)
        print("Channel SNR = {:g}dB. Scale-fading = {:.4f}".format(EbNodB, scale_fading))
        # Generate data
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_fading(
            constellation, n_samp, n_samp_val, scale_fading, config['sigma_noise_measurement']
        )

    elif type_channel.lower() == 'awgn':
        sigma_noise = get_noise_stddev(EbNodB, rate=config['rate_comm'], E_avg_symbol=E_avg_qam)
        print("Channel SNR = {:g}dB. Noise-stddev = {:.6f}".format(EbNodB, sigma_noise))
        # Generate data
        inputs_target_list, x_target_list, y_target_list = simulate_channel_variations_gaussian(
            None, constellation, n_samp, n_samp_val, [sigma_noise] * config['dim_encoding'], use_channel_output=False
        )
    else:
        raise ValueError("Invalid value '{}' for input 'type_channel'".format(type_channel))

    x_data = x_target_list[0]  # encoded symbols
    y_data = y_target_list[0]  # channel outputs
    return x_data, y_data, tf.convert_to_tensor(constellation, dtype=DTYPE_TF)


def generate_data_gmm(params_gmm, constellation, n_samp, max_phase_shift=0.):
    # Generate the required number of samples from the given Gaussian mixture model
    n_symb, n_dim = constellation.shape
    # Number of samples per symbol
    n_samp_per_symb = get_samples_per_symbol(n_samp, n_symb)
    # Maximum phase shift in radians
    max_phase_shift = (max_phase_shift * np.pi) / 180.
    x_data = None
    y_data = None
    labels = []
    gmm_distr = []
    for m in range(n_symb):
        # Generate samples from the GMM for the current symbol
        means = tf.convert_to_tensor(params_gmm[m]['means'], dtype=DTYPE_TF)
        covars = tf.convert_to_tensor(params_gmm[m]['covars'], dtype=DTYPE_TF)
        priors = tf.convert_to_tensor(params_gmm[m]['priors'], dtype=DTYPE_TF)
        pis = tfd.Categorical(probs=priors)
        comps = [tfd.MultivariateNormalDiag(loc=means[i, :], scale_diag=covars[i, :]) for i in range(priors.shape[0])]
        distr = tfd.Mixture(cat=pis, components=comps)
        y_data_curr = distr.sample(sample_shape=n_samp_per_symb[m]).numpy()
        gmm_distr.append(distr)

        if max_phase_shift > 0.:
            # Random phase shift
            phi_arr = np.random.uniform(low=-max_phase_shift, high=max_phase_shift, size=n_samp_per_symb[m])
            y_data_curr_trans = np.zeros_like(y_data_curr)
            cos_phi = np.cos(phi_arr)
            sin_phi = np.sin(phi_arr)
            y_data_curr_trans[:, 0] = y_data_curr[:, 0] * cos_phi - y_data_curr[:, 1] * sin_phi
            y_data_curr_trans[:, 1] = y_data_curr[:, 0] * sin_phi + y_data_curr[:, 1] * cos_phi
        else:
            y_data_curr_trans = y_data_curr

        x_data_curr = np.tile(constellation[m, :], (n_samp_per_symb[m], 1))
        labels.extend([m] * n_samp_per_symb[m])
        if x_data is None:
            x_data = x_data_curr
        else:
            x_data = np.vstack([x_data, x_data_curr])

        if y_data is None:
            y_data = y_data_curr_trans
        else:
            y_data = np.vstack([y_data, y_data_curr_trans])

    labels = np.array(labels, dtype=int).reshape(-1, 1)
    # shuffle the order of samples
    ind = np.random.permutation(n_samp)

    return x_data[ind, :], y_data[ind, :], labels[ind, :], gmm_distr


def convert_to_complex_ndarray(x):
    ns, nd = x.shape
    if nd != 2:
        raise ValueError("Input array dimension should be 2 for this function to work")

    return np.array([[x[i, 0] + 1j * x[i, 1]] for i in range(ns)], dtype=np.complex)


def get_samples_per_symbol(n_samp, n_symb):
    # Number of samples per symbol - even split
    if n_samp > n_symb:
        n_samp_per_symb = (n_samp // n_symb) * np.ones(n_symb, dtype=int)
        r = n_samp % n_symb
        if r > 0:
            v = np.zeros(n_symb, dtype=int)
            v[:r] = 1
            n_samp_per_symb = n_samp_per_symb + v

    else:
        n_samp_per_symb = np.ones(n_symb, dtype=int)

    assert np.sum(n_samp_per_symb) == n_samp, "Number of samples per symbol does not add up"
    return n_samp_per_symb


def get_labels_from_symbols(target_domain_x, unique_x):
    # Get the one-hot-coded labels from the transmitted symbols (channel inputs). Both inputs should be numpy arrays.
    # Label the rows of `target_domain_x` with the index of the nearest constellation point in `unique_x`
    d_mat = cdist(target_domain_x, unique_x, metric='euclidean')
    n_samp, n_uniq = d_mat.shape
    ind_min = np.argmin(d_mat, axis=1)
    val_min = np.array([d_mat[i, ind_min[i]] for i in range(n_samp)])
    # Ideally `val_min` should be all zeros
    if not np.allclose(val_min, np.zeros(n_samp), rtol=0., atol=1e-6):
        raise ValueError("Transmitted symbols do not match the unique symbols (constellation) of the autoencoder. "
                         "Max distance = {:.6e}".format(np.max(val_min)))

    return tf.one_hot(ind_min, n_uniq)


def plot_channel_data(x_data, y_data, plot_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_data[:, 0], y_data[:, 1], alpha=0.5, c='lightseagreen')
    ax.scatter(x_data[:, 0], x_data[:, 1], alpha=1, c='k')
    plt.xticks(size=14)
    plt.yticks(size=14)
    # plt.show()
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def get_num_jobs(n_jobs):
    """
    Number of processes or jobs to use for multiprocessing.
    :param n_jobs: None or int value that specifies the number of parallel jobs. If set to None, -1, or 0, this will
                   use all the available CPU cores. If set to negative values, this value will be subtracted from
                   the available number of CPU cores. For example, `n_jobs = -2` will use `cpu_count - 2`.
    """
    cc = multiprocessing.cpu_count()
    if n_jobs is None or n_jobs == -1 or n_jobs == 0:
        n_jobs = cc
    elif n_jobs < -1:
        n_jobs = max(1, cc + n_jobs)
    else:
        n_jobs = min(n_jobs, cc)

    return n_jobs


def configure_exponential_lr_schedule(lr_init, lr_final, n_train, batch_size, n_epochs, staircase=True):
    # Given an initial and final learning rate, choose the decay rate and the number of decay steps for an
    # exponential learning rate schedule.
    if lr_final > lr_init:
        raise ValueError("Final learning rate cannot be larger than the initial learning rate for an "
                         "exponential schedule.")

    steps_per_epoch = n_train // batch_size
    decay_rate = np.exp((-1. / n_epochs) * np.log(lr_init / lr_final))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr_init, decay_steps=steps_per_epoch, decay_rate=decay_rate, staircase=staircase
    )
    # This choice will decrease the LR by the factor `decay_rate` at the start of every epoch. All the steps
    # within an epoch will have the same LR.
    return lr_schedule


def get_optimizer(optim_method, use_lr_schedule, n_samples, batch_size, n_epochs,
                  lr_init_adam=LEARNING_RATE_ADAM, lr_init_sgd=LEARNING_RATE_SGD, momentum=0.9, nesterov=False):
    optim_method = optim_method.lower()
    if optim_method not in ('adam', 'sgd'):
        raise ValueError("'{}' is not a supported optimization method".format(optim_method))

    # Learning rate if using Adam
    # For Adam, an initial LR of 0.005 and final LR of 0.0001 also finds a good solution, but does not converge well
    lr_final_adam = lr_init_adam / 10.
    # Learning rate if using SGD with momentum. SGD needs a higher initial learning rate for this problem
    lr_final_sgd = lr_init_sgd / 20.
    if use_lr_schedule:
        # Use an exponential learning rate schedule if it works well
        if optim_method == 'adam':
            lr_schedule = configure_exponential_lr_schedule(lr_init_adam, lr_final_adam, n_samples,
                                                            batch_size, n_epochs)
            optim_obj = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=EPSILON_ADAM)
        else:
            lr_schedule = configure_exponential_lr_schedule(lr_init_sgd, lr_final_sgd, n_samples,
                                                            batch_size, n_epochs)
            optim_obj = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=nesterov)
    else:
        # Fixed learning rate
        if optim_method == 'adam':
            optim_obj = tf.keras.optimizers.Adam(learning_rate=lr_init_adam, epsilon=EPSILON_ADAM)
        else:
            optim_obj = tf.keras.optimizers.SGD(learning_rate=lr_init_sgd, momentum=momentum, nesterov=nesterov)

    return optim_obj


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # source_mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # source_sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # source_pi_logits
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # psi_a
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # psi_b
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # psi_c
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # psi_beta
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # psi_gamma
                     tf.TensorSpec(shape=[], dtype=tf.int64),               # n_comp
                     tf.TensorSpec(shape=[], dtype=tf.int64)                # n_dim
                     ],
    experimental_relax_shapes=True
)
def update_gmm_params(source_mus, source_sigmas, source_pi_logits, psi_a, psi_b, psi_c, psi_beta, psi_gamma,
                      n_comp, n_dim):
    # Update the Gaussian mixture parameters using the affine transformation parameters.
    # print("Tracing 'update_gmm_params'...You should see this line only during the first call.")
    # new_mus = tf.math.add(tf.math.multiply(source_mus, psi_a), psi_b)
    psi_a = tf.reshape(psi_a, [n_comp, n_dim, n_dim])
    # Reshape `source_mus`: [n_target, n_comp * n_dim] -> [n_target, n_comp, n_dim] -> [n_comp, n_target, n_dim]
    reshaped_source_mus = tf.transpose(tf.reshape(source_mus, [-1, n_comp, n_dim]), [1, 0, 2])

    # The inner matrix multiplication is between tensors of shape `[n_comp, n_target, n_dim]` and
    # `[n_comp, n_dim, n_dim]`, and the resultant tensor has shape `[n_comp, n_target, n_dim]`. The first two
    # axes of this tensor are swapped to give a `[n_target, n_comp, n_dim]` tensor, which is then reshaped
    # into a `[n_target, n_comp * n_dim]` tensor.
    new_mus = tf.reshape(tf.transpose(tf.linalg.matmul(reshaped_source_mus, psi_a), [1, 0, 2]),
                         [-1, n_comp * n_dim])
    # Add the `[n_target, n_comp * n_dim]` tensor with the `[n_comp * n_dim]` tensor `psi_b`
    new_mus = tf.math.add(new_mus, psi_b)
    # new_sigmas = tf.math.multiply(source_sigmas, tf.math.square(psi_c))
    new_sigmas = tf.math.multiply(source_sigmas, tf.math.abs(psi_c))
    new_pi_logits = tf.math.add(tf.math.multiply(source_pi_logits, psi_beta), psi_gamma)

    return new_mus, new_sigmas, new_pi_logits


def split_adaptation_params(psi_values, n_comp, n_dim):
    # Split the adaptation parameters into their components. `psi_values` is a TF tensor
    # return tf.split(psi_values, [n_comp * n_dim * n_dim] + [n_comp * n_dim] * 2 + [n_comp] * 2)
    return tf.split(psi_values, tf.constant([n_comp * n_dim * n_dim, n_comp * n_dim, n_comp * n_dim, n_comp, n_comp]))


def affine_transform_gmm_params(mus, sigmas, pi_logits, params_affine, n_comp, n_dim):
    # Update the Gaussian mixture parameters using the affine transformation parameters
    return update_gmm_params(
        mus, sigmas, pi_logits, params_affine[0], params_affine[1], params_affine[2], params_affine[3],
        params_affine[4], tf.constant(n_comp, dtype=tf.int64), tf.constant(n_dim, dtype=tf.int64)
    )


def get_mixture_params(params_mdn, n_comp, n_dim, logits=False):
    # `params_mdn` is the tensor of parameters corresponding to each input.
    # n_samp = params_mdn.shape[0]
    # `mus` and `sigmas` have the same shape `(n_samp, n_comp * n_dim)`.
    # `pis` has shape `(n_samp, n_comp)` and its values should sum to 1 if `logits = False`
    mus, sigmas, pis = tf.split(params_mdn, num_or_size_splits=[n_comp * n_dim,
                                                                n_comp * n_dim,
                                                                n_comp], axis=1)
    if not logits:
        pis = tf.nn.softmax(pis, axis=1)

    # outputs are tensors
    return mus, sigmas, pis


def sample_batch_MDN(y_concat, n_comp, n_dim, temp=1., sigma_temp=1.):
    # `y_concat` is the array of parameters corresponding to each input.
    y_samples = np.apply_along_axis(MDN_base.sample_from_output, 1, y_concat, n_dim, n_comp,
                                    temp=temp, sigma_temp=sigma_temp)
    # `y_samples` should have shape `(y_concat.shape[0], 1, n_dim)`. Remove the single-dimensional axis
    return np.squeeze(y_samples)


def helper_data_generate(x_unique, n_samp):
    # Generates a specified number of one-hot-coded inputs and encoded symbols.
    n_unique, n_dim = x_unique.shape
    inputs_unique = tf.one_hot(tf.range(n_unique), n_unique).numpy()    # unique one-hot-coded inputs
    # Split `n_samp` into roughly equal number of samples per unique symbol
    k = n_samp // n_unique
    r = n_samp % n_unique
    n_samp_per_symbol = k * np.ones(n_unique) + np.concatenate([np.ones(r), np.zeros(n_unique - r)])
    n_samp_per_symbol = np.asarray(np.random.permutation(n_samp_per_symbol), dtype=int)
    x = np.zeros((n_samp, n_dim))
    inp = np.zeros((n_samp, n_unique))
    st = en = 0
    for i in range(n_unique):
        st = en
        en = st + n_samp_per_symbol[i]
        x[st:en, :] = x_unique[i, :]
        inp[st:en, :] = inputs_unique[i, :]

    # Randomly shuffle the order of samples
    shuf = np.random.permutation(n_samp)
    x = x[shuf, :]
    inp = inp[shuf, :]
    return inp, x


def generate_random_symbol_data(n_bits, n_samples):
    # Generate random one-hot-coded inputs and the corresponding labels for training or evaluating the autoencoder
    m = 2 ** n_bits
    eye_matrix = np.eye(m).astype(np.float32)
    labels = np.random.randint(m, size=n_samples)
    inputs = eye_matrix[labels, :]

    return inputs, labels


def get_autoencoder_data_filename(n_bits):
    # Standard file basename for the autoencoder training data
    return 'data_autoencoder_{:d}symbols.npz'.format(2 ** n_bits)


def generate_data_autoencoder(data_file_autoenc, mod_order, n_train_per_symbol, n_test_per_symbol, val_frac=0.2,
                              max_train=640000, max_test=1024000, verbose=True):
    # Helper function to generate training, validation and test data for training an autoencoder.
    if data_file_autoenc and os.path.isfile(data_file_autoenc):
        # Data arrays are saved in the .npz format (zip file containing multiple .npy files).
        # During the first run, the data arrays will be randomly generated and saved to this file.
        # During subsequent runs, the data arrays will be loaded from this file and reused.
        with open(data_file_autoenc, 'rb') as fp:
            npz_obj = np.load(fp)
            x_train = npz_obj['x_train']
            x_val = npz_obj['x_val']
            x_test = npz_obj['x_test']
            labels_test = npz_obj['labels_test']

        return x_train, x_val, x_test, labels_test

    # Restrict the size of the data sets for large modulation orders
    n_train = n_train_per_symbol * mod_order
    if n_train > max_train:
        n_train_per_symbol = max_train // mod_order

    n_val_per_symbol = int(np.ceil(val_frac * n_train_per_symbol))
    n_test = n_test_per_symbol * mod_order
    if n_test > max_test:
        n_test_per_symbol = max_test // mod_order

    # Replicate each unique one-hot coded message multiple times and randomize the order
    eye_matrix = np.eye(mod_order).astype(np.float32)
    x_train = np.tile(eye_matrix, (n_train_per_symbol, 1))
    np.random.shuffle(x_train)
    x_val = np.tile(eye_matrix, (n_val_per_symbol, 1))
    np.random.shuffle(x_val)
    n_test = n_test_per_symbol * mod_order
    labels_test = np.random.randint(mod_order, size=n_test)
    x_test = eye_matrix[labels_test, :]
    if verbose:
        print("\nData for training and evaluating the autoencoder:")
        print("Number of train samples: {:d}. Number of validation samples: {:d}.".format(x_train.shape[0],
                                                                                          x_val.shape[0]))
        print("Number of test samples: {:d}.".format(x_test.shape[0]))

    # Save the data arrays to a .npz file
    data_dict = {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'labels_test': labels_test
    }
    with open(data_file_autoenc, 'wb') as fp:
        np.savez(fp, **data_dict)

    return x_train, x_val, x_test, labels_test


def simulate_channel_variations_gaussian(autoencoder_orig, x_unique, n_target_train, n_target_eval, stddev_target,
                                         use_channel_output=True):
    # Channel variations in the form of additive Gaussian noise.
    # `stddev_target` should be a list of standard deviation values per dimension.
    n_unique, n_dim = x_unique.shape
    if isinstance(x_unique, tf.Tensor):
        x_unique = x_unique.numpy()   # unique symbols

    x_target_list = []
    y_target_list = []
    inputs_target_list = []
    for n_samp in (n_target_train, n_target_eval):
        inp, x = helper_data_generate(x_unique, n_samp)
        x_ten = tf.convert_to_tensor(x, dtype=DTYPE_TF)
        # Predict the mixture density channel model and generate samples from the resulting Gaussian mixture
        if use_channel_output:
            params_mdn = autoencoder_orig.channel_predict(x_ten)
            y_clean = autoencoder_orig.sampling(params_mdn).numpy()
        else:
            y_clean = x

        # Add Gaussian noise with the target standard deviation
        y = y_clean + np.random.normal(loc=np.zeros(n_dim), scale=stddev_target, size=(n_samp, n_dim))
        # Collect the tensors
        inputs_target_list.append(tf.convert_to_tensor(inp, dtype=DTYPE_TF))
        x_target_list.append(x_ten)
        y_target_list.append(tf.convert_to_tensor(y, dtype=DTYPE_TF))

    # Each output is a list of two tensors, one for training and one for evaluation
    return inputs_target_list, x_target_list, y_target_list


def simulate_channel_variations_fading(x_unique, n_target_train, n_target_eval, scale_fading, noise_stddev_base):
    # Simulate frequency-selective fading channel variations.
    # `scale_fading` is the fading amplitude and `noise_stddev_base` is the Gaussian noise standard deviation.
    n_unique, n_dim = x_unique.shape
    if isinstance(x_unique, tf.Tensor):
        x_unique = x_unique.numpy()   # unique symbols

    x_target_list = []
    y_target_list = []
    inputs_target_list = []
    for n_samp in (n_target_train, n_target_eval):
        inp, x = helper_data_generate(x_unique, n_samp)
        # Scale the channel inputs with a uniform random variable of the given power level, and add Gaussian noise
        # of the given standard deviation.
        samples_unif = np.random.uniform(low=0., high=scale_fading, size=(n_samp, n_dim))
        # Set the same scale factor for the in-phase and quadrature phase samples.
        for j in range(n_dim // 2):
            samples_unif[:, 2 * j + 1] = samples_unif[:, 2 * j]

        noise_samp = noise_stddev_base * np.random.normal(size=(n_samp, n_dim))
        y = samples_unif * x + noise_samp
        # Collect the tensors
        inputs_target_list.append(tf.convert_to_tensor(inp, dtype=DTYPE_TF))
        x_target_list.append(tf.convert_to_tensor(x, dtype=DTYPE_TF))
        y_target_list.append(tf.convert_to_tensor(y, dtype=DTYPE_TF))

    # Each output is a list of two tensors, one for training and one for evaluation
    return inputs_target_list, x_target_list, y_target_list


def simulate_channel_variations_ricean_fading(x_unique, n_target_train, n_target_eval, nu, sigma_a, noise_stddev_base):
    # Simulate Ricean fading channel variations.
    # `nu` and `sigma_a` are parameters of the Rice distribition, and `noise_stddev_base` is the standard
    # deviation of the additive Gaussian noise.
    n_unique, n_dim = x_unique.shape
    if isinstance(x_unique, tf.Tensor):
        x_unique = x_unique.numpy()   # unique symbols

    x_target_list = []
    y_target_list = []
    inputs_target_list = []
    for n_samp in (n_target_train, n_target_eval):
        inp, x = helper_data_generate(x_unique, n_samp)
        # Scale the channel inputs with Ricean fading random variables and add Gaussian noise.
        samples_rice = rice.rvs(nu / sigma_a, scale=sigma_a, size=(n_samp, n_dim))
        # Set the same scale factor for the in-phase and quadrature phase samples.
        for j in range(n_dim // 2):
            samples_rice[:, 2 * j + 1] = samples_rice[:, 2 * j]

        noise_samp = noise_stddev_base * np.random.normal(size=(n_samp, n_dim))
        y = samples_rice * x + noise_samp
        # Collect the tensors
        inputs_target_list.append(tf.convert_to_tensor(inp, dtype=DTYPE_TF))
        x_target_list.append(tf.convert_to_tensor(x, dtype=DTYPE_TF))
        y_target_list.append(tf.convert_to_tensor(y, dtype=DTYPE_TF))

    # Each output is a list of two tensors, one for training and one for evaluation
    return inputs_target_list, x_target_list, y_target_list


def check_weights_equality(weights, weights_ref, n_arrays=None):
    # Compare the list of layer weights and biases for equality.
    # `weights` and `weights_ref` are a list of numpy arrays.
    if n_arrays is None:
        n_arrays = len(weights)

    flag_eq = True
    for i in range(n_arrays):
        if not np.allclose(weights[i], weights_ref[i], rtol=0., atol=1e-6):
            flag_eq = False
            break

    return flag_eq


def configure_plot_axes(ax, y_vals, n_ticks=8):
    # Small utility to configure the log-scaled y-axis is BER vs. SNR plots used in multiple places.
    ticks_range = np.array([1e-8, 3.16e-8, 1e-7, 3.16e-7, 1e-6, 3.16e-6, 1e-5, 3.16e-5,
                            1e-4, 3.16e-4, 1e-3, 3.16e-3, 0.01, 0.0316,
                            0.1, 0.15, 0.22, 0.32, 0.46, 0.68, 1.])
    # n_ticks_max = ticks_range.shape[0]
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)

    # mask = np.logical_and(ticks_range > y_min, ticks_range < y_max)
    # ind_select = np.arange(n_ticks_max)[mask]
    # if ind_select[0] > 0:
    #     ind_select = np.insert(ind_select, 0, ind_select[0] - 1)
    # if ind_select[-1] < (n_ticks_max - 1):
    #     ind_select = np.insert(ind_select, ind_select.shape[0], ind_select[-1] + 1)

    # v = ticks_range[ind_select]
    v = np.logspace(np.log10(y_min), np.log10(y_max), num=n_ticks)
    if y_min >= 0.1:
        v = np.unique(np.around(v, decimals=1))

    ax.set_yscale('log')
    ax.set_yticks(v)
    # ax.set_yticklabels(['{:.2g}'.format(a) for a in v])
    # ax.set_ylim(v[0], v[-1])

    if y_min >= 0.1:
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        # try '%.2e' or '%.2g'
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))

    # Suppress minor ticks
    ax.get_yaxis().set_tick_params(which='minor', size=0, width=0)


def get_noise_stddev(EbNodB, rate=1., E_avg_symbol=1.):
    """
    Get the noise standard deviations corresponding to the given SNR in dB.
    `EbNodB` can be a float, list, tuple, array etc.

    According to [1], the expression for noise variance is $\sigma^2 = E_avg_symbol / (2 R E_b / N_0)$, where
    `E_avg_symbol` is the average energy per symbol,
    `E_b / N_0` is the SNR, and
    `R = k / n` is the communication rate, `k` is the number of bits per symbol and `n` is the encoding dimension.
    [1] Section III.A in https://arxiv.org/pdf/1702.00832.pdf

    :param EbNodB: float or a an iterable (list, tuple, array) with the SNR values in dB.
    :param rate: Communication rate, R = (# bits per symbol) / (encoding dimension).
    :param E_avg_symbol: Average energy per symbol.
    :return: Noise standard deviation corresponding to the SNR value(s). Will be a float if `EbNodB` is a float;
             else it will be a list.
    """
    if hasattr(EbNodB, '__iter__'):
        # list, tuple, array etc
        noise_stdev = []
        for v in EbNodB:
            EbNo = 10. ** (v / 10.)  # SNR in ratio
            noise_stdev.append(np.sqrt(E_avg_symbol / (2 * rate * EbNo)))
    else:
        EbNo = 10. ** (EbNodB / 10.)  # SNR in ratio
        noise_stdev = np.sqrt(E_avg_symbol / (2 * rate * EbNo))

    return noise_stdev


def estimate_stddev_awgn(x, y):
    """
    Given a set of channel input and output samples, find the maximum likelihood estimate of the noise standard
    deviation for the additive white Gaussian noise model.
    `x` and `y` should be numpy arrays of the same shape.
    """
    if x.shape != y.shape:
        raise ValueError("Inputs 'x' and 'y' do not have the same shape.")

    n_samp, n_dim = x.shape
    err_norm = np.linalg.norm(y - x, axis=1)
    noise_std = np.sqrt(np.sum(err_norm ** 2) / (n_samp * n_dim))
    return noise_std


def calculate_fading_factor(EbNo, noise_stdev, rate_comm, E_avg_symbol):
    """
    The maximum value of the uniform random variable that scales the modulated signal. Suppose $X$ is the modulated
    signal of dimension $d$, and $Y = a X$, where $a$ is a uniform random variable on $[0, A]$, this function finds
    the the $A$ value required for the signal $Y$ to have a target SNR of `EbNo`.
    The average energy of $X$ is given by `E_avg_symbol`.

    :param EbNo: Target SNR value (should be the ratio and not in dB).
    :param noise_stdev: Noise standard deviation estimate.
    :param rate_comm: Communication rate in bits per channel use.
    :param E_avg_symbol: Average energy per symbol in the channel input.
    :return: Fading factor `a` used to scale the modulated signal.
    """
    return np.sqrt((6. * rate_comm * noise_stdev * noise_stdev * EbNo) / E_avg_symbol)


def calculate_ricean_fading_params(EbNo, EbNo_min, noise_stdev, rate_comm, E_avg_symbol):
    """
    Calculate the parameters of the Rice distribution required to achieve a target SNR.

    :param EbNo: Target SNR value (should be the ratio and not in dB).
    :param EbNo_min: Minimum SNR value (should be the ratio and not in dB).
    :param noise_stdev: Noise standard deviation estimate.
    :param rate_comm: Communication rate in bits per channel use.
    :param E_avg_symbol: Average energy per symbol in the channel input.
    :return: parameters `nu`, `sigma_a`, and K of the Rice distribution.
    """
    sigma_a = np.sqrt((rate_comm * noise_stdev * noise_stdev * EbNo_min) / E_avg_symbol)
    nu = np.sqrt((2 * rate_comm * noise_stdev * noise_stdev * max(0., EbNo - EbNo_min)) / E_avg_symbol)
    # K factor in dB
    if EbNo > EbNo_min:
        K = 10. * np.log10(EbNo / EbNo_min - 1.)
    else:
        K = -1e6

    return nu, sigma_a, K


def get_minimum_SNR_ricean(EbNodB):
    # Sets the minimum value of `E_b/N_0` such that the K-factor of the Ricean fading stays within a predefined range.
    # Range of suitable K-factor values
    k_min_db, k_max_db = K_FACTOR_RANGE
    k_min = 10. ** (k_min_db / 10.)
    k_max = 10. ** (k_max_db / 10.)
    # Target SNR in ratio
    EbNo = 10. ** (EbNodB / 10.)
    v_min = EbNo / (k_max + 1.)
    v_max = EbNo / (k_min + 1.)
    # Initial choice of `EbNo_min` is 1
    EbNo_min = 1.
    if EbNo_min < v_min:
        EbNo_min = v_min
    if EbNo_min > v_max:
        EbNo_min = v_max

    EbNodB_min = 10. * np.log10(EbNo_min)
    return EbNo_min, EbNodB_min


def simulate_channel_variations(y_clean, encoded_signal, variation, EbNodB, E_avg_symbol,
                                noise_stdev_base, rate_comm, EbNodB_min=0., using_channel_outputs=False):
    # Usually `y_clean` and `encoded_signal` are the same TF tensor.
    # `EbNodB_min` is needed only for Ricean fading.
    n_test_samples, n_dim = y_clean.shape
    variation = variation.lower()
    if variation in ('gaussian_noise', 'awgn'):
        # Vary the SNR by changing the variance of the additive Gaussian noise.
        noise_stdev = get_noise_stddev(EbNodB, rate=rate_comm, E_avg_symbol=E_avg_symbol)
        if using_channel_outputs and noise_stdev > noise_stdev_base:
            y = y_clean + tf.random.normal([n_test_samples, n_dim],
                                           stddev=np.sqrt(noise_stdev ** 2 - noise_stdev_base ** 2))
        else:
            # Add Gaussian noise directly to the encoded signal
            y = encoded_signal + tf.random.normal([n_test_samples, n_dim], stddev=noise_stdev)

    elif variation == 'fading':
        EbNo = 10. ** (EbNodB / 10.)  # SNR in ratio
        # scale_fading = EbNo * noise_stdev_base
        scale_fading = calculate_fading_factor(EbNo, noise_stdev_base, rate_comm, E_avg_symbol)
        samples_unif = np.random.uniform(low=0., high=scale_fading, size=(n_test_samples, n_dim))
        # Set the same scale factor for the in-phase and quadrature phase samples.
        for j in range(n_dim // 2):
            samples_unif[:, 2 * j + 1] = samples_unif[:, 2 * j]

        samples_unif = tf.convert_to_tensor(samples_unif, dtype=DTYPE_TF)
        noise_samp = tf.random.normal([n_test_samples, n_dim], stddev=noise_stdev_base)
        y = tf.math.multiply(samples_unif, y_clean) + noise_samp

    elif variation in ('fading_ricean', 'fading_rayleigh'):
        # SNR and minimum SNR in ratio
        EbNo = 10. ** (EbNodB / 10.)
        if variation == 'fading_rayleigh':
            EbNo_min = EbNo
        else:
            EbNo_min = 10. ** (EbNodB_min / 10.)

        nu, sigma_a, K = calculate_ricean_fading_params(EbNo, EbNo_min, noise_stdev_base, rate_comm, E_avg_symbol)
        samples_rice = rice.rvs(nu / sigma_a, scale=sigma_a, size=(n_test_samples, n_dim))
        # Set the same scale factor for the in-phase and quadrature phase samples.
        for j in range(n_dim // 2):
            samples_rice[:, 2 * j + 1] = samples_rice[:, 2 * j]

        samples_rice = tf.convert_to_tensor(samples_rice, dtype=DTYPE_TF)
        noise_samp = tf.random.normal([n_test_samples, n_dim], stddev=noise_stdev_base)
        y = tf.math.multiply(samples_rice, y_clean) + noise_samp

    else:
        raise ValueError("Value '{}' not supported for the input 'variation'".format(variation))

    return y


def get_ber_channel_variations(autoencoder, channel, x_test, labels_test, rate_comm, noise_stdev_base,
                               variation='gaussian_noise', EbNodB_range=None, EbNodB_min=None, batch_size=None,
                               using_channel_outputs=False):
    # Calculates the block or symbol error rate of the autoencoder on a large random test set.
    # Channel variations in the form of additive Gaussian noise or fading with additive Gaussian noise are supported.
    variation = variation.lower()
    if variation not in ('gaussian_noise', 'awgn', 'fading', 'fading_ricean', 'fading_rayleigh'):
        raise ValueError("Value '{}' not supported for the input 'variation'".format(variation))

    ret_float = False
    if EbNodB_range is None:
        EbNodB_range = SNR_RANGE_DEF
    elif not hasattr(EbNodB_range, '__iter__'):
        # single value
        EbNodB_range = np.array([EbNodB_range])
        ret_float = True

    if EbNodB_min is None:
        # Minimum SNR in dB is needed for Ricean fading
        EbNodB_min = min(0., np.min(EbNodB_range))

    n_test_samples = x_test.shape[0]
    n_levels = len(EbNodB_range)
    if batch_size is None:
        batch_size = n_test_samples + 1

    # Average energy of the distinct symbols learned by the autoencoder
    unique_x = autoencoder.encoder(autoencoder.inputs_unique)
    E_avg_symbol = tf.reduce_sum(unique_x ** 2) / unique_x.shape[0]
    # Encoding of the test inputs
    encoded_signal = autoencoder.encoder_predict(x_test)
    if using_channel_outputs:
        # Variations are applied to the channel output of the autoencoder. Takes longer to calculate the BER
        # params_mdn = autoencoder.channel_predict(encoded_signal)
        params_mdn = channel.predict(encoded_signal, batch_size=batch_size)
        # Generate random samples from the mixture density channel model. Batching the predictions to avoid running
        # out of memory. This block is a bit slow. So increase the batch size if memory permits.
        k = n_test_samples // batch_size
        y = []
        for j in range(k):
            y.append(autoencoder.sampling(params_mdn[(j * batch_size): ((j + 1) * batch_size), :]))

        st = k * batch_size
        if st < n_test_samples:
            y.append(autoencoder.sampling(params_mdn[st:, :]))

        y_clean = tf.concat(y, 0)
    else:
        y_clean = encoded_signal

    # Don't use a large batch size here since the scale factor at the decoder is calculated from a batch of outputs
    batch_size_pred = max(BATCH_SIZE_PRED, 2 * unique_x.shape[0])
    ber = [0.] * n_levels
    for i in range(n_levels):
        y = simulate_channel_variations(y_clean, encoded_signal, variation, EbNodB_range[i], E_avg_symbol,
                                        noise_stdev_base, rate_comm, EbNodB_min=EbNodB_min,
                                        using_channel_outputs=using_channel_outputs)
        # Predict the decoder model to get the probability over the distinct symbols
        prob_output = autoencoder.decoder_predict(y, batch_size=batch_size_pred)
        # BER of the decoded signal
        pred_output = np.argmax(prob_output, axis=1)
        mask_errors = (pred_output != labels_test)
        ber[i] = mask_errors.astype(np.float).sum() / n_test_samples
        # Set BER of exactly 0 to a small non-zero value (avoids taking log(0) in the BER plots)
        if ber[i] <= 0.:
            ber[i] = 0.1 / n_test_samples

    if ret_float:
        return ber[0]
    else:
        # list of same length as `EbNodB_range`
        return ber
