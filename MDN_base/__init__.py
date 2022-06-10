# Implementation of the Gaussian mixture density network.
from .version import __version__
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

# Seed for the random initialization of the layer weights
SEED_WEIGHTS = 1234


def elu_plus_one_plus_epsilon(x):
    return tf.nn.elu(x) + 1. + tf.keras.backend.epsilon()


def relu_plus_epsilon(x):
    # Used as activation for the standard deviation outputs to ensure that they don't become too small
    return tf.nn.relu(x) + 1e-4


class MDN(layers.Layer):
    # Mixture density network layer that predicts the parameters of a Gaussian mixture model
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super(MDN, self).__init__(**kwargs)
        self.output_dimension = output_dimension
        self.num_mixtures = num_mixtures
        with tf.name_scope('MDN'):
            self.mdn_mus = layers.Dense(self.num_mixtures * self.output_dimension, name='mdn_mus',
                                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
            self.mdn_sigmas = layers.Dense(self.num_mixtures * self.output_dimension,
                                           # activation=relu_plus_epsilon,
                                           activation=elu_plus_one_plus_epsilon,
                                           name='mdn_sigmas',
                                           kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
            self.mdn_pi = layers.Dense(self.num_mixtures, name='mdn_pi',
                                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))

    '''
    # Not required since the build method of the Dense layer will be called automatically
    def build(self, input_shape):
        with tf.name_scope('mus'):
            self.mdn_mus.build(input_shape)
        with tf.name_scope('sigmas'):
            self.mdn_sigmas.build(input_shape)
        with tf.name_scope('pis'):
            self.mdn_pi.build(input_shape)

        super(MDN, self).build(input_shape)
    '''

    @property
    def trainable_weights(self):
        return self.mdn_mus.trainable_weights + self.mdn_sigmas.trainable_weights + self.mdn_pi.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.mdn_mus.non_trainable_weights + self.mdn_sigmas.non_trainable_weights + \
               self.mdn_pi.non_trainable_weights
    
    def call(self, x):
        with tf.name_scope('MDN'):
            mdn_out = layers.concatenate([self.mdn_mus(x),
                                          self.mdn_sigmas(x),
                                          self.mdn_pi(x)],
                                          name='mdn_outputs')
        return mdn_out

    def compute_output_shape(self, input_shape):
        return input_shape[0], (2 * self.output_dimension * self.num_mixtures) + self.num_mixtures

    def get_config(self):
        config = {
            "output_dimension": self.output_dimension,
            "num_mixtures": self.num_mixtures
        }
        base_config = super(MDN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

def size_MDN_outputs(n_mixes, n_dim):
    return 2 * n_mixes * n_dim + n_mixes


def construct_mixture_model(y_pred, n_mixes, n_dim):
    # Construct the Gaussian mixture model given the parameters `y_pred` output by the mixture density network
    # Split the output into paramaters
    mus, sigmas, pi_logits = tf.split(y_pred,
                                      num_or_size_splits=[n_mixes * n_dim, n_mixes * n_dim, n_mixes],
                                      axis=-1, name='mdn_coef_split')
    pis = tfd.Categorical(logits=pi_logits)
    # List of Gaussian components
    comps = [tfd.MultivariateNormalDiag(
        loc=mus[:, (i * n_dim): ((i + 1) * n_dim)], scale_diag=sigmas[:, (i * n_dim): ((i + 1) * n_dim)])
        for i in range(n_mixes)
    ]
    return tfd.Mixture(cat=pis, components=comps)


# Standard log-likelihood loss function for generative training of the MDN model
def get_mixture_loss_func(output_dim, num_mixes):
    # Construct a loss function with the right number of mixtures and outputs. Negative log-likelihood
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, size_MDN_outputs(num_mixes, output_dim)], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Construct the mixture model from the output parameters produced by the mixture density network
        mixture = construct_mixture_model(y_pred, num_mixes, output_dim)
        # Negative log-likelihood of `y_true` under the mixture model
        loss = mixture.log_prob(y_true)
        return -1. * tf.reduce_mean(loss)

    # Actually return the loss function
    with tf.name_scope('MDN'):
        return mdn_loss_func
    

# Posterior log-likelihood loss function for discriminative training of the MDN model
def get_mixture_loss_func_posterior(output_dim, num_mixes, nx_unique):
    # Construct a loss function with the right number of mixtures and outputs. Negative posterior log-likelihood.
    # Assumes a uniform prior over the inputs (hence the prior is ignored)
    def mdn_loss_func_posterior(y_true, y_pred):
        # `params_batch` should have shape `(n_batch, n_params)`
        # `params_unique` should have shape `(nx_unique, n_params)`
        y_pred = tf.reshape(y_pred, [-1, size_MDN_outputs(num_mixes, output_dim)])
        y_true = tf.reshape(y_true, [-1, output_dim])
        params_batch = y_pred[:-nx_unique, :]
        params_unique = y_pred[-nx_unique:, :]
        # Log of the conditional density of `y_true` under the mixture model: `log p(y_i | x_i)`
        mixture = construct_mixture_model(params_batch, num_mixes, output_dim)
        loss1 = mixture.log_prob(y_true)

        # Mixture model parameters corresponding to the unique inputs
        mus, sigmas, pi_logits = tf.split(params_unique,
                                          [num_mixes * output_dim, num_mixes * output_dim, num_mixes], axis=1)
        # Stack the means, variances, and mixing weights from all the mixture models into a larger mixture model
        # Normalize the logits into probabilities prior to stacking
        pis = (1. / nx_unique) * tf.nn.softmax(pi_logits, axis=1)
        n_comp_concat = nx_unique * num_mixes
        mus = tf.reshape(mus, [n_comp_concat * output_dim])
        sigmas = tf.reshape(sigmas, [n_comp_concat * output_dim])
        pis = tf.reshape(pis, [n_comp_concat])
        # List of Gaussian components of the concatenated mixture
        comps = [tfd.MultivariateNormalDiag(loc=mus[(i * output_dim): ((i + 1) * output_dim)],
                                            scale_diag=sigmas[(i * output_dim): ((i + 1) * output_dim)])
                 for i in range(n_comp_concat)]
        mixture = tfd.Mixture(cat=tfd.Categorical(probs=pis), components=comps)
        # Log of the marginal density of `y_true` under the mixture model: `log p(y_i)`
        loss2 = mixture.log_prob(y_true)

        return tf.reduce_mean(loss2 - loss1)

    with tf.name_scope('MDN'):
        return mdn_loss_func_posterior


def get_mixture_sampling_fun(output_dim, num_mixes):
    # Sampling from the mixture model
    def sampling_func(y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, size_MDN_outputs(num_mixes, output_dim)], name='reshape_ypreds')
        # Construct the mixture model from the output parameters produced by the mixture density network
        mixture = construct_mixture_model(y_pred, num_mixes, output_dim)
        # Sample from the mixture model
        # TODO: temperature adjustment for sampling function.
        return mixture.sample()

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return sampling_func


def get_mixture_mse_accuracy(output_dim, num_mixes):
    # Construct a loss function with the right number of mixtures and outputs. Mean squared error
    def mse_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistribued layer
        y_pred = tf.reshape(y_pred, [-1, size_MDN_outputs(num_mixes, output_dim)], name='reshape_ypreds')
        y_true = tf.reshape(y_true, [-1, output_dim], name='reshape_ytrue')
        # Construct the mixture model from the output parameters produced by the mixture density network
        mixture = construct_mixture_model(y_pred, num_mixes, output_dim)
        # Mean squared error
        # TODO: temperature adjustment for sampling functon.
        samp = mixture.sample()
        return tf.reduce_mean(tf.square(samp - y_true), axis=-1)

    # Actually return the loss_func
    with tf.name_scope('MDNLayer'):
        return mse_func


def split_mixture_params(params, output_dim, num_mixes):
    mus = params[:(num_mixes * output_dim)]
    sigs = params[(num_mixes * output_dim): (2 * num_mixes * output_dim)]
    pi_logits = params[-num_mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    return e / np.sum(e)


def sample_from_categorical(dist):
    cum_dist = np.cumsum(dist)
    mask = cum_dist >= np.random.rand()
    ind = np.arange(dist.shape[0])[mask]
    return ind[0]


def sample_from_output(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    # Diagonal covariance matrix for component `m`
    cov_matrix = sigma_temp * np.diag(sigs[(m * output_dim): ((m + 1) * output_dim)] ** 2)

    return np.random.multivariate_normal(mus[(m * output_dim): ((m + 1) * output_dim)], cov_matrix, size=1)


def sample_from_output_layer(params, output_dim, num_mixes, temp=1.0, sigma_temp=1.0):
    return sample_from_output(params, output_dim, num_mixes, temp=temp, sigma_temp=sigma_temp)


def softmax_layer(w, t=1.0):
    return softmax(w, t=t)
