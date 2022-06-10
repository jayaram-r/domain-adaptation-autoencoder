# Code related to standard modulation and demodulation techniques.
import numpy as np
from scipy.spatial.distance import cdist
import tensorflow as tf
from .context import *
from helpers.utils import simulate_channel_variations
from helpers.constants import SNR_RANGE_DEF, BATCH_SIZE_PRED


def generate_QAM_constellation(mod_order, avg_power=1.):
    val_range = [-1., 1.]
    n_bits = int(np.log2(mod_order))
    assert (2 ** n_bits == mod_order), "Modulation order is not a power of 2"
    k = int(np.ceil(mod_order ** 0.5))
    if mod_order == 8:
        k1 = 4
        k2 = 2
    elif (k * k) == mod_order:
        k1 = k2 = k
    else:
        # Set `k` to be even
        if k % 2 == 1:
            k += 1

        found = False
        k1 = k
        while (not found) and (k <= mod_order):
            val = k ** 2 - mod_order
            val_sqrt = int(val ** 0.5)
            if val == (val_sqrt * val_sqrt):    # perfect square
                k1 = k
                found = True
            else:
                k += 2

        if found:
            k2 = k1 - int(np.sqrt(k1 * k1 - mod_order))
            # print("mod_order = {:d}, k1 = {:d}, k2 = {:d}".format(mod_order, k1, k2))
        else:
            raise ValueError("Failed to find factors for 'mod_order = {:d}'".format(mod_order))

    constellation = np.zeros((mod_order, 2))
    if mod_order == (k1 * k2):
        x = np.linspace(val_range[0], val_range[1], num=k1, endpoint=True)
        y = np.linspace(val_range[0], val_range[1], num=k2, endpoint=True)
        xx, yy = np.meshgrid(x, y)
        n = 0
        for i in range(k2):
            for j in range(k1):
                constellation[n, :] = [xx[i, j], yy[i, j]]
                n += 1
    else:
        # Non-square constellation pattern
        x = np.linspace(val_range[0], val_range[1], num=k1, endpoint=True)
        y = np.linspace(val_range[0], val_range[1], num=k1, endpoint=True)
        xx, yy = np.meshgrid(x, y)
        offset1 = int((k1 - k2) / 2.)
        offset2 = int((k1 + k2) / 2.)
        n = 0
        for i in range(k1):
            if (i >= offset1) and (i < offset2):
                # Include all columns
                for j in range(k1):
                    constellation[n, :] = [xx[i, j], yy[i, j]]
                    n += 1
            else:
                # Include only the central chunk of the columns
                for j in range(offset1, offset2):
                    constellation[n, :] = [xx[i, j], yy[i, j]]
                    n += 1

        if n != mod_order:
            print("ERROR: something wrong in the constellation generation")

    # Normalize the symbols to achieve the target average power
    val = np.sum(constellation ** 2) / mod_order
    return np.sqrt(avg_power / val) * constellation


class StandardQAM:
    def __init__(self, n_bits, avg_power=1., use_scale=True):
        self.n_bits = n_bits
        self.avg_power = avg_power
        # Scale (normalize) the received signals by their average power before decoding
        self.use_scale = use_scale
        self.mod_order = 2 ** n_bits
        # Unique one-hot-coded inputs and the constellation symbols
        self.inputs_unique = tf.one_hot(tf.range(self.mod_order), self.mod_order).numpy()
        self.constellation = generate_QAM_constellation(self.mod_order, avg_power=self.avg_power)

    def encode(self, inputs):
        ind = tf.argmax(inputs, axis=1).numpy()
        return np.take(self.constellation, ind, axis=0)

    def decode(self, y, batch_size=BATCH_SIZE_PRED):
        # Maximum likelihood decoding under AWGN assumption: assign the nearest constellation symbol as the
        # decoded output. `y` can be a TF tensor or a numpy array
        n_samp = y.shape[0]
        if isinstance(y, tf.Tensor):
            y = y.numpy()

        scale = 1.
        # Split the pairwise distance computation into batches to avoid memory issues
        k = n_samp // batch_size
        d_mat = np.zeros((n_samp, self.constellation.shape[0]))
        for j in range(k):
            st = j * batch_size
            en = (j + 1) * batch_size
            if self.use_scale:
                scale = np.sqrt(np.mean(np.sum(y[st:en, :] ** 2, axis=1)))

            d_mat[st:en, :] = cdist(y[st:en, :], self.constellation * scale, metric='euclidean')

        st = k * batch_size
        if st < n_samp:
            if self.use_scale:
                scale = np.sqrt(np.mean(np.sum(y[st:, :] ** 2, axis=1)))

            d_mat[st:, :] = cdist(y[st:, :], self.constellation * scale, metric='euclidean')

        # Return the index of the nearest constellation symbol for each sample
        return np.argmin(d_mat, axis=1)


def calculate_BLER(modulation, x_test, labels_test, rate_comm, noise_stdev_base,
                   variation='gaussian_noise', EbNodB_range=None, EbNodB_min=None):
    # Calculates the block or symbol error rate of a standard modulation technique on a large random test set.
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

    # Average power of the constellation symbols
    n_const = modulation.constellation.shape[0]
    E_avg_symbol = np.sum(modulation.constellation ** 2) / n_const
    # Encoding of the test inputs
    encoded_signal = modulation.encode(x_test)
    y_clean = encoded_signal

    n_test_samples = x_test.shape[0]
    n_levels = len(EbNodB_range)
    batch_size = max(BATCH_SIZE_PRED, 2 * n_const)
    ber = [0.] * n_levels
    for i in range(n_levels):
        y = simulate_channel_variations(y_clean, encoded_signal, variation, EbNodB_range[i], E_avg_symbol,
                                        noise_stdev_base, rate_comm, EbNodB_min=EbNodB_min)
        # Prediction and BER
        pred_output = modulation.decode(y, batch_size=batch_size)
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
