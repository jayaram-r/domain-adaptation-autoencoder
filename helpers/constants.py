import numpy as np
import tensorflow as tf

DTYPE_TF = tf.float32

# Seed for the random initialization of the layer weights
SEED_WEIGHTS = 1234

# Range of SNR values in dB
SNR_RANGE_DEF = np.array([4., 6., 8, 10, 12, 13, 14, 15, 16, 17, 18, 20])

# Range of values for the regularization constant `lambda` in the channel adaptation loss function
# LAMBDA_VALUES = np.array([1e-5, 1e-4, 0.001, 0.01, 0.1, 1, 5, 10, 50])
# Logarithmically-spaced values between 10^-5 and 100. Keep the number of values a multiple of 4 or 8 for best use
# of parallel processing.
LAMBDA_VALUES = np.logspace(-5, 2, num=8, base=10.)

# Configuration for the temperature annealing schedule used to smooth the argmax function.
# Applies only to the autoencoder based on MAP symbol estimation at the receiver.
CONFIG_ANNEAL = {
    'anneal': True,         # If set to `False`, training will be done only at the final temperature
    'temp_init': 1.,        # Initial temperature
    'temp_final': 0.05,     # Final temperature
    'n_tsteps': 9,          # Number of temperature steps
    'n_epochs_per_tstep': 10   # Number of epochs for each temperature step
}

# Batch size for prediction. The scale factor for normalization at the decoder is calculated over this batch size
BATCH_SIZE_PRED = 256
MAX_BATCH_SIZE = 10000

# Standard file basenames for the FPGA data
TX_DATA_BASENAME = 'tx_symbols.mat'     # Tx symbols
RX_DATA_BASENAME = 'rx_symbols.mat'     # Rx symbols
LABELS_BASENAME = 'labels.mat'          # One-hot-coded labels or message inputs

# Standard file basenames
CONSTELLATION_BASENAME = 'constellation_autoencoder.npy'
ADAPTATION_PARAMS_BASENAME = 'adaptation_params.npy'

# List of evaluation metrics
METRICS_EVAL_LIST = ['log_loss', 'error_rate', 'inverse_trans_log_like', 'inverse_trans_log_post']

# Minimum SNR in dB used to set the Rice distribution parameters
SNR_MIN_RICEAN = 0.

# Range of suitable values of the K-factor of the Rice distribution (in dB)
K_FACTOR_RANGE = [6., 14.]

# Settings for the Adam and SGD optimizers
LEARNING_RATE_ADAM = 0.001
LEARNING_RATE_SGD = 0.1
EPSILON_ADAM = 1e-7

MAX_N_SAMPLES = 2000000

# For plots
POINT_STYLES = ['o', '^', 'v', 's', '*', 'x', 'd', '>', '<', '1', 'h', 'P', '+', '_', '2', '|', '3', '4', '.']
COLORS = ['b', 'g', 'r', 'c', 'm', 'orange', 'lawngreen', 'grey', 'hotpink', 'y', 'steelblue', 'tan',
          'lightsalmon', 'navy', 'gold', 'lightseagreen']
