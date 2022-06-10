# Code related to the autoencoder and sampling
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import math
from .context import *
import MDN_base
from helpers.utils import (
    split_adaptation_params,
    affine_transform_gmm_params,
    get_mixture_params,
    get_optimizer
)
import helpers.MDN_classes
from helpers.constants import *


def get_autoencoder_name(autoencoder):
    basename = 'autoencoder'
    if autoencoder.name == 'AutoencoderSymbolEstimation':
        if autoencoder.map_estimation:
            basename += '_symbol_estimation_map'
        else:
            basename += '_symbol_estimation_mmse'
    elif autoencoder.name == 'AutoencoderAdaptGenerative':
        basename += '_adapt_generative'

    return basename


def get_autoencoder_type(autoencoder):
    # Get the autoencoder type from the model
    if autoencoder.name == 'AutoencoderSymbolEstimation':
        if autoencoder.map_estimation:
            return 'symbol_estimation_map'
        else:
            return 'symbol_estimation_mmse'
    elif autoencoder.name == 'AutoencoderAdaptGenerative':
        return 'adapt_generative'
    else:
        return 'standard'


def load_autoencoder_from_file(filename, n_dims, n_components, nx_unique):
    # Specify custom functions and classes to avoid deserialization errors
    custom_objects = {
        'mdn_loss_func': MDN_base.get_mixture_loss_func(n_dims, n_components),
        'mdn_loss_func_posterior': MDN_base.get_mixture_loss_func_posterior(n_dims, n_components, nx_unique),
        'MDN': MDN_base.MDN,
        'MDN_model': helpers.MDN_classes.MDN_model,
        'MDN_model_disc': helpers.MDN_classes.MDN_model_disc,
        'Encoder': Encoder,
        'Decoder': Decoder,
        'L2SymmetryRegularizer': L2SymmetryRegularizer,
        'SamplingMDN': SamplingMDN,
        'SamplingMDNGumbel': SamplingMDNGumbel,
        'SymbolEstimationExpectation': SymbolEstimationExpectation,
        'AutoencoderInverseAffine': AutoencoderInverseAffine,
        'AutoencoderAdaptGenerative': AutoencoderAdaptGenerative,
        'AutoencoderSymbolEstimation': AutoencoderSymbolEstimation
    }
    autoencoder = tf.keras.models.load_model(filename, custom_objects=custom_objects, compile=True)

    return autoencoder


def initialize_autoencoder(type_autoencoder, mdn_model, config, config_optimizer, n_train,
                           temperature=CONFIG_ANNEAL['temp_final'], freeze_encoder=False, freeze_decoder=False):
    # Helper function to initialize and compile the right type of autoencoder.
    n_symbols = config['dim_encoding'] // 2
    if type_autoencoder == 'standard':
        autoencoder = AutoencoderInverseAffine(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.)
        )
    elif type_autoencoder == 'adapt_generative':
        autoencoder = AutoencoderAdaptGenerative(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.)
        )
    elif type_autoencoder == 'symbol_estimation_mmse':
        autoencoder = AutoencoderSymbolEstimation(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.)
        )
    elif type_autoencoder == 'symbol_estimation_map':
        autoencoder = AutoencoderSymbolEstimation(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.),
            map_estimation=True, temperature=temperature
        )
    else:
        raise ValueError("Invalid value '{}' for the input 'type_autoencoder'".format(type_autoencoder))

    # Has to be done before compiling the model
    if freeze_encoder:
        autoencoder.encoder.trainable = False
    if freeze_decoder:
        autoencoder.decoder.trainable = False

    # The MAP symbol estimation autoencoder runs only a few epochs per temperature step. However, its learning rate
    # schedule is configured using the maximum number of epochs. This is intentional because it ensures that the same
    # learning rate schedule is maintained in all cases.
    optim_obj = get_optimizer(
        config_optimizer['optim_method'], config_optimizer['use_lr_schedule'], n_train,
        config_optimizer['batch_size'], config_optimizer['n_epochs'],
        lr_init_adam=config_optimizer['learning_rate_adam'], lr_init_sgd=config_optimizer['learning_rate_sgd']
    )
    # If needed, pass `metrics=['categorical_accuracy']` to the `compile` method
    autoencoder.compile(optimizer=optim_obj, loss='categorical_crossentropy')
    _ = autoencoder(tf.keras.Input(shape=(config['mod_order'],)))

    return autoencoder


def initialize_autoencoder_adapted(type_autoencoder, mdn_model, encoder, decoder, psi_values, config,
                                   temperature=CONFIG_ANNEAL['temp_final']):
    # Create a new autoencoder model with the adapted channel parameters and the pre-trained encoder and
    # decoder networks.
    n_symbols = config['dim_encoding'] // 2
    if type_autoencoder == 'standard':
        autoencoder = AutoencoderInverseAffine(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.),
            psi_values=psi_values, encoder=encoder, decoder=decoder
        )
    elif type_autoencoder == 'adapt_generative':
        autoencoder = AutoencoderAdaptGenerative(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.),
            psi_values=psi_values, encoder=encoder, decoder=decoder
        )
    elif type_autoencoder == 'symbol_estimation_mmse':
        autoencoder = AutoencoderSymbolEstimation(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.),
            psi_values=psi_values, encoder=encoder, decoder=decoder
        )
    elif type_autoencoder == 'symbol_estimation_map':
        autoencoder = AutoencoderSymbolEstimation(
            mdn_model, config['n_bits'], n_symbols, n_hidden=config['n_hidden'],
            scale_outputs=config.get('scale_outputs', True), l2_reg_strength=config.get('l2_reg_strength', 0.),
            psi_values=psi_values, encoder=encoder, decoder=decoder, map_estimation=True, temperature=temperature
        )
    else:
        raise ValueError("Invalid value '{}' taken by 'type_autoencoder'".format(type_autoencoder))

    # Call the method on a symbolic input to build it
    _ = autoencoder(tf.keras.Input(shape=(config['mod_order'],)))

    return autoencoder


# Comment out the `@tf.function` decorator to debug in eager mode
@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # y
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # pi_logits
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # prior_prob_x
                     tf.TensorSpec(shape=[3], dtype=tf.int32)               # sizes_arr
                     ]
)
def posterior_x_given_y(y, mus, sigmas, pi_logits, prior_prob_x, sizes_arr):
    """
    Posterior probability of each constellation symbol (channel input) `x` given the channel output `y`.
    Using `tf.function` to compile this function into a python-independent computational graph.
    Same inputs as the function `posterior_component_and_x_given_y`; see its doc-string for details.

    To transform the logits to probabilities, run `tf.nn.softmax(ret, axis=1)` on the output of this function.

    :return: Tensor of shape `(n_samp, nx_unique)` with the posterior logits.
    """
    # print("Tracing 'posterior_x_given_y'...You should see this line only during the first call.")
    # Cannot do `n_comp, n_dim, nx_unique = sizes_arr` in a graph-compiled function
    n_comp = sizes_arr[0]
    n_dim = sizes_arr[1]
    nx_unique = sizes_arr[2]
    log_prior_prob_x = tf.math.log(prior_prob_x)
    offset = tf.reduce_logsumexp(pi_logits, 1)
    # Accumulate the posterior logits in this tensor. Similar to a list append
    post_logits_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for j in tf.range(nx_unique, dtype=tf.int32):  # x \in \mathcal{X}
        temp_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for i in tf.range(n_comp, dtype=tf.int32):
            mvn = tfd.MultivariateNormalDiag(loc=mus[j, (i * n_dim): ((i + 1) * n_dim)],
                                             scale_diag=sigmas[j, (i * n_dim): ((i + 1) * n_dim)])
            vals = mvn.log_prob(y) + pi_logits[j, i] - offset[j] + log_prior_prob_x[j]  # tensor of shape `(n_samp, )`
            temp_arr = temp_arr.write(i, vals)

        post_logits_arr = post_logits_arr.write(j, tf.reduce_logsumexp(temp_arr.stack(), 0))

    return tf.transpose(post_logits_arr.stack())


'''
# Alternate implementation for `posterior_x_given_y` that cannot be compiled into a TF graph.
# Suitable for running in eager-mode.
def posterior_x_given_y_alt(y, mus, sigmas, pi_logits, prior_prob_x, sizes_arr):
    n_comp, n_dim, nx_unique = sizes_arr
    log_prior_prob_x = tf.math.log(prior_prob_x)
    post_logits_arr = []
    for i in range(nx_unique):
        comps = [tfd.MultivariateNormalDiag(
            loc=mus[i, (j * n_dim): ((j + 1) * n_dim)], scale_diag=sigmas[i, (j * n_dim): ((j + 1) * n_dim)])
            for j in range(n_comp)
        ]
        mixture = tfd.Mixture(cat=tfd.Categorical(logits=pi_logits[i, :]), components=comps)
        post_logits_arr.append(tf.expand_dims(mixture.log_prob(y) + log_prior_prob_x[i], 1))

    return tf.concat(post_logits_arr, 1)

'''


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # y
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # pi_logits
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # prior_prob_x
                     tf.TensorSpec(shape=[3], dtype=tf.int32)               # sizes_arr
                     ]
)
def posterior_component_given_y(y, mus, sigmas, pi_logits, prior_prob_x, sizes_arr):
    """
    Posterior probability of the mixture component `i` given the channel output `y`.
    Using `tf.function` to compile this function into a python-independent computational graph.
    Same inputs as the function `posterior_component_and_x_given_y`; see its doc-string for details.

    To transform the logits to probabilities, run `tf.nn.softmax(ret, axis=1)` on the output of this function.

    :return: Tensor of shape `(n_samp, n_comp)` with the component posterior logits.
    """
    # print("Tracing...You should see this line only during the first call.")
    # Cannot do `n_comp, n_dim, nx_unique = sizes_arr` in a graph-compiled function
    n_comp = sizes_arr[0]
    n_dim = sizes_arr[1]
    nx_unique = sizes_arr[2]
    log_prior_prob_x = tf.math.log(prior_prob_x)
    offset = tf.reduce_logsumexp(pi_logits, 1)
    # Accumulate the posterior logits in this tensor. Similar to a list append
    post_logits_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(n_comp, dtype=tf.int32):
        st = i * n_dim
        en = (i + 1) * n_dim
        temp_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for j in tf.range(nx_unique, dtype=tf.int32):  # x \in \mathcal{X}
            mvn = tfd.MultivariateNormalDiag(loc=mus[j, st:en], scale_diag=sigmas[j, st:en])
            vals = mvn.log_prob(y) + pi_logits[j, i] - offset[j] + log_prior_prob_x[j]  # tensor of shape `(n_samp, )`
            temp_arr = temp_arr.write(j, vals)

        post_logits_arr = post_logits_arr.write(i, tf.reduce_logsumexp(temp_arr.stack(), 0))

    return tf.transpose(post_logits_arr.stack())


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # y
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # pi_logits
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # prior_prob_x
                     tf.TensorSpec(shape=[3], dtype=tf.int32)               # sizes_arr
                     ]
)
def posterior_component_and_x_given_y(y, mus, sigmas, pi_logits, prior_prob_x, sizes_arr):
    """
    Posterior probability of the pair of constellation symbol `x` and the mixture component `i` given the
    channel output `y`.
    Using `tf.function` to compile this function into a python-independent computational graph. Be careful while
    modifying this implementation. It has been written to be compatible with the requirements of `tf.function`.
    For example, all operations are done using Tensorflow tensors instead of standard Python or Numpy objects.
    See the following docs for more details:
    https://www.tensorflow.org/api_docs/python/tf/function
    https://www.tensorflow.org/guide/function

    To transform the logits to probabilities, run `tf.nn.softmax(ret, axis=1)` on the output of this function.

    :param y: Channel outputs. Tensor of shape `(n_samp, n_dim)`, where `n_samp` and `n_dim` are the number of
                               samples and dimension respectively.
    :param mus: Tensor of means from each component of shape `(nx_unique, n_dim * n_comp)`, where `nx_unique` is
                the number of distinct values of the channel input `x`, and `n_comp` is the number of components.
    :param sigmas: Tensor of standard deviations from each component of shape `(nx_unique, n_dim * n_comp)`.
    :param pi_logits: Tensor of log of the mixture priors of shape `(n_unique, nx_comp)`.
    :param prior_prob_x: Tensor with the prior probability of each unique input. Should have length `nx_unique`.
    :param sizes_arr: Tensor of length 3 with the following array sizes: `[n_comp, n_dim, nx_unique]`. Should have
                      dtype `tf.int32`. `n_comp` is the number of components in the mixture, `n_dim` is the number
                      of dimensions, and `nx_unique` is the number of unique constellation symbols.
    :return:
        Tensor of shape `(n_samp, n_comp * nx_unique)` with the posterior logits.
        - The values (for all samples) corresponding to component `i` and symbol `j` can be accessed as
        `ret[:, i * nx_unique + j]`.
        - A column index `c` can be mapped to the component index `i` and symbol index `j` as follows:
          i = c // nx_unique
          j = c % nx_unique
    """
    # print("Tracing...You should see this line only during the first call.")
    # Cannot do `n_comp, n_dim, nx_unique = sizes_arr` in a graph-compiled function
    n_comp = sizes_arr[0]
    n_dim = sizes_arr[1]
    nx_unique = sizes_arr[2]
    log_prior_prob_x = tf.math.log(prior_prob_x)
    offset = tf.reduce_logsumexp(pi_logits, 1)
    # Accumulate the posterior logits in this tensor. Similar to a list append
    post_logits_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(n_comp, dtype=tf.int32):
        st = i * n_dim
        en = (i + 1) * n_dim
        for j in tf.range(nx_unique, dtype=tf.int32):  # x \in \mathcal{X}
            mvn = tfd.MultivariateNormalDiag(loc=mus[j, st:en], scale_diag=sigmas[j, st:en])
            vals = mvn.log_prob(y) + pi_logits[j, i] - offset[j] + log_prior_prob_x[j]   # tensor of shape `(n_samp, )`
            post_logits_arr = post_logits_arr.write(i * nx_unique + j, vals)

    return tf.transpose(post_logits_arr.stack())


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                     tf.TensorSpec(shape=[None], dtype=tf.float32),
                     tf.TensorSpec(shape=[None], dtype=tf.int64),
                     tf.TensorSpec(shape=[None], dtype=tf.int64),
                     tf.TensorSpec(shape=[], dtype=tf.int64),
                     tf.TensorSpec(shape=[], dtype=tf.int64)]
)
def transform_inverse_affine(samples_, mus_orig_, mus_, psi_c_, idx_symb_, idx_comp_, n_samp_, n_dim_):
    """
    For each sample (row) in `samples_`, apply the appropriate (component and symbol specific) inverse affine
    transformation. `tf.function` is used to compile this function into a python-independent computational graph.

    :param samples_: TF tensor containing the channel outputs. Should have shape `(n_samp_, n_dim_)`.
    :param mus_orig_: TF tensor with the component means of the original Gaussian mixture corresponding to each
                      distinct constellation symbol. Should have shape `(nx_unique, n_comp * n_dim_)`, where
                      `n_comp` is the number of components and `nx_unique` is the number of distinct
                      constellation symbols.
    :param mus_: Same as `mus_orig_` but this corresponds to the component means of the adapted Gaussian mixture.
    :param psi_c_: TF tensor with the scale parameters transforming the component variances. Should have shape
                   `(n_comp * n_dim_, )`.
    :param idx_symb_: TF tensor with the index of the constellation symbol corresponding to each row in `samples_`.
                      Should have shape `(n_samp_, )`.
    :param idx_comp_: TF tensor with the index of the mixture component corresponding to each row in `samples_`.
                      Should have shape `(n_samp_, )`.
    :param n_samp_: Number of samples specified as a TF constant.
    :param n_dim_: Number of samples specified as a TF constant.
    :return: TF tensor of shape `(n_samp_, n_dim_)` with the transformed samples.
    """
    # print("Tracing...You should see this line only during the first call.")
    output_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for n in tf.range(n_samp_, dtype=tf.int32):
        # array slice corresponding to component `idx_comp_[n]`
        st = idx_comp_[n] * n_dim_
        en = (idx_comp_[n] + 1) * n_dim_
        vals = tf.math.add(
            tf.math.divide(samples_[n, :] - mus_[idx_symb_[n], st:en], psi_c_[st:en]),
            mus_orig_[idx_symb_[n], st:en]
        )
        output_arr = output_arr.write(n, vals)

    return output_arr.stack()


class SamplingMDN(Layers.Layer):
    # Sampling layer for the Gaussian mixture density network
    def __init__(self, n_mixes, n_dim, **kwargs):
        kwargs.update({'name': 'sampling_mdn'})
        super(SamplingMDN, self).__init__(**kwargs)
        self.n_mixes = n_mixes
        self.n_dim = n_dim

    def call(self, params_mixture, training=None):
        # Build a Gaussian mixture distribution from the parameter tensor and generate samples from the distribution
        # params = tf.reshape(params, [-1, 2 * self.n_mixes * self.n_dim + self.n_mixes])
        mus, sigmas, pi_logits = tf.split(
            params_mixture, [self.n_mixes * self.n_dim, self.n_mixes * self.n_dim, self.n_mixes], axis=1
        )
        # List of Gaussian components
        comps = [
            tfd.MultivariateNormalDiag(loc=mus[:, (i * self.n_dim): ((i + 1) * self.n_dim)],
                                       scale_diag=sigmas[:, (i * self.n_dim): ((i + 1) * self.n_dim)])
            for i in range(self.n_mixes)
        ]
        mixture = tfd.Mixture(cat=tfd.Categorical(logits=pi_logits), components=comps)

        return mixture.sample()

    def get_config(self):
        config = super(SamplingMDN, self).get_config()
        config.update(
            {'n_mixes': self.n_mixes,
             'n_dim': self.n_dim}
        )
        return config


class SamplingMDNGumbel(Layers.Layer):
    # Sampling layer for the Gaussian mixture density network. Uses the Gumbel Softmax method to make the sampling
    # layer differentiable. Lower temperature values provide a closer approximation to the samples from a Gaussian
    # mixture, but it can lead to numerical issues like very large gradients.
    def __init__(self, n_mixes, n_dim, temperature=0.01, **kwargs):
        kwargs.update({'name': 'sampling_mdn_gumbel'})
        super(SamplingMDNGumbel, self).__init__(**kwargs)
        self.n_mixes = n_mixes
        self.n_dim = n_dim
        self.temperature = temperature
        # Standard Gumbel distribution
        self.dist_gumbel = tfd.Gumbel(loc=0., scale=1.)

    def call(self, params_mixture, training=None):
        # Split the parameter tensor into the means, variances, and mixing weights of the Gaussian mixture.
        # `mus` and `sigmas` should each have shape `[nb, self.n_mixes * self.n_dim]`.
        # `pi_logits` should have shape `[nb, self.n_mixes]`.
        mus, sigmas, pi_logits = tf.split(
            params_mixture, [self.n_mixes * self.n_dim, self.n_mixes * self.n_dim, self.n_mixes], axis=1
        )
        bs = tf.shape(params_mixture)[0]
        # IID samples from the standard Gumbel and standard Gaussian distributions
        samp_gumbel = self.dist_gumbel.sample([bs, self.n_mixes])
        samp_gaussian = tf.random.normal([bs, self.n_dim], mean=0., stddev=1.)
        if training:
            # Temperature-scaled softmax weights across the components
            weights_mix = tf.nn.softmax(
                tf.math.scalar_mul(1. / self.temperature, samp_gumbel + pi_logits), axis=1
            )
        else:
            # During inference, we use argmax instead of temperature-scaled softmax to perform exact sampling
            weights_mix = tf.one_hot(tf.math.argmax(samp_gumbel + pi_logits, axis=1), self.n_mixes)

        '''
        # Loop version: use this if the list comprehension leads to memory issues
        concat_arr = []
        for i in range(self.n_mixes):
            z = tf.math.add(
                tf.math.multiply(sigmas[:, (i * self.n_dim): ((i + 1) * self.n_dim)], samp_gaussian),
                mus[:, (i * self.n_dim): ((i + 1) * self.n_dim)]
            )
            concat_arr.append(
                tf.math.multiply(tf.expand_dims(weights_mix[:, i], 1), z)
            )
        '''
        concat_arr = [
            tf.math.multiply(tf.expand_dims(weights_mix[:, i], 1),
                             tf.math.add(mus[:, (i * self.n_dim): ((i + 1) * self.n_dim)],
                                         tf.math.multiply(sigmas[:, (i * self.n_dim): ((i + 1) * self.n_dim)],
                                                          samp_gaussian)))
            for i in range(self.n_mixes)
        ]
        return tf.math.reduce_sum(concat_arr, 0)

    def get_config(self):
        config = super(SamplingMDNGumbel, self).get_config()
        config.update(
            {'n_mixes': self.n_mixes,
             'n_dim': self.n_dim,
             'temperature': self.temperature}
        )
        return config


class L2SymmetryRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, l2=1.):
        self.l2 = l2

    def __call__(self, x):
        # L2 norm of the centroid vector
        return self.l2 * tf.norm(tf.math.reduce_sum(x, 0), ord='euclidean')

    def get_config(self):
        return {'l2': float(self.l2)}


class Encoder(tf.keras.Model):
    def __init__(self, n_bits, n_symbols, n_hidden=None, normalization='average_power', scale_amplitude=True,
                 l2_reg_strength=0., **kwargs):
        '''
        - `normalization` can be set to 'average_power' or 'energy'.
        - If `scale_amplitude` is set to True, it scales the encoded symbols to lie in [-1, 1] along each dimension.
        - `l2_reg_strength` is the strength of the L2 regularization penalty on the centroid of the encoded symbols.
           Set this to 0 to disable regularization.
        '''
        super(Encoder, self).__init__(name='Encoder', **kwargs)
        self.n_bits = n_bits
        self.n_symbols = n_symbols
        self.n_hidden = n_hidden
        self.normalization = normalization
        self.scale_amplitude = scale_amplitude
        self.l2_reg_strength = l2_reg_strength

        print("Encoder details. Normalization: {}, scale_amplitude: {}, l2_reg_strength: {:g}".
              format(self.normalization, self.scale_amplitude, self.l2_reg_strength))
        # Dimension of the encoded symbol is `2 * n_symbols`
        self.dim_output = 2 * n_symbols
        # Hidden layer dimension can be different from `2 ** n_bits`
        nx_unique = 2 ** n_bits
        if self.n_hidden is None:
            self.n_hidden = nx_unique

        # Set of distinct one-hot-coded inputs
        self.inputs_unique = tf.one_hot(tf.range(nx_unique), nx_unique)
        # Add an optional regularizer to the activations of the dense layer just prior to the normalization layer
        reg = L2SymmetryRegularizer(l2=self.l2_reg_strength) if (self.l2_reg_strength > 0.) else None
        # Fully connected layers
        self.dense1 = Layers.Dense(self.n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.dense2 = Layers.Dense(self.dim_output, activation='linear',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS),
                                   activity_regularizer=reg)
        # Normalization layer
        if self.normalization == 'average_power':
            # Average power of all the symbols $E[\|x\|^2]$ is set to be 1.
            # The layer takes a list of two tensors; the first one corresponds to the input batch, and the second one
            # is a tensor of the unique encoded inputs.
            self.complex = Layers.Lambda(lambda x: x[0] / tf.math.sqrt(tf.reduce_sum(x[1] ** 2) / x[1].shape[0]))
        elif self.normalization == 'energy':
            # Energy constraint: sets the norm of each constellation symbols equal to 1.
            # This will result in the constellation points to lie on a unit sphere
            self.complex = Layers.Lambda(lambda x: x / tf.norm(x, axis=1, keepdims=True))
        else:
            raise ValueError("Invalid value '{}' specified for the input 'normalization'".format(normalization))

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        if self.normalization == 'average_power':
            x_unique = self.dense2(self.dense1(self.inputs_unique))
            x = self.complex([x, x_unique])
        else:
            x = self.complex(x)

        if self.scale_amplitude:
            # Apply tanh to scale the encoder outputs to the range [-1, 1]. This is a practical requirement
            # of the antenna design
            return tf.math.tanh(x)
        else:
            return x

    def get_config(self):
        config = {
            'n_bits': self.n_bits,
            'n_symbols': self.n_symbols,
            'n_hidden': self.n_hidden,
            'normalization': self.normalization,
            'scale_amplitude': self.scale_amplitude,
            'l2_reg_strength': self.l2_reg_strength,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Decoder(tf.keras.Model):
    def __init__(self, n_bits, n_hidden=None, scale_outputs=True, **kwargs):
        super(Decoder, self).__init__(name='Decoder', **kwargs)
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.scale_outputs = scale_outputs
        self.dim_output = 2 ** n_bits
        # Hidden layer dimension can be different from `2 ** n_bits`
        if self.n_hidden is None:
            self.n_hidden = self.dim_output

        self.dense1 = Layers.Dense(self.n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.dense2 = Layers.Dense(self.dim_output, activation='softmax',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))

    def call(self, outputs):
        # Should we apply inverse of tanh here since we apply tanh at the encoder?
        # x = tf.math.atanh(outputs)
        if self.scale_outputs:
            # Scale the channel outputs to have unit average power
            scale = tf.math.sqrt(tf.math.reduce_mean(tf.math.reduce_sum(outputs ** 2, 1)))
            outputs = tf.math.scalar_mul(1. / scale, outputs)

        x = self.dense1(outputs)
        x = self.dense2(x)
        return x

    def get_config(self):
        config = {
            'n_bits': self.n_bits,
            'n_hidden': self.n_hidden,
            'scale_outputs': self.scale_outputs
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AutoencoderInverseAffine(tf.keras.Model):
    """
    Class implementing an autoencoder-based communication system with a Gaussian mixture density network (MDN)
    channel model. The channel model is allowed to be adapted using component-conditional affine transformation
    parameters. The affine transformation parameters are also used to compensate for changes in the channel
    conditions at the receiver end.
    """
    def __init__(self, mdn_model, n_bits, n_symbols, n_hidden=None, scale_outputs=True, l2_reg_strength=0.,
                 psi_values=None, encoder=None, decoder=None, estimate_input_priors=False, **kwargs):
        super(AutoencoderInverseAffine, self).__init__(name='AutoencoderInverseAffine', **kwargs)
        self.n_bits = n_bits
        self.n_symbols = n_symbols
        self.n_hidden = n_hidden
        self.scale_outputs = scale_outputs
        self.l2_reg_strength = l2_reg_strength
        self.psi_values = psi_values
        self.estimate_input_priors = estimate_input_priors
        # Encoder dimension and the number of mixture components in the channel model
        self.n_dims = mdn_model.n_dims
        self.n_mixes = mdn_model.n_mixes
        if self.n_dims != (2 * n_symbols):
            raise ValueError("Mismatch in the number of dimensions.")

        if encoder is None:
            self.encoder = Encoder(n_bits, n_symbols, n_hidden=self.n_hidden, l2_reg_strength=self.l2_reg_strength)
        else:
            # Pre-trained encoder
            self.encoder = encoder
            self.encoder.trainable = False

        # Channel MDN model
        self.channel = mdn_model
        self.channel.trainable = False
        # Sampling layer for the channel model
        # self.sampling = SamplingMDN(self.n_mixes, self.n_dims)
        self.sampling = SamplingMDNGumbel(self.n_mixes, self.n_dims)
        if decoder is None:
            self.decoder = Decoder(n_bits, n_hidden=self.n_hidden, scale_outputs=self.scale_outputs)
        else:
            # Pre-trained decoder
            self.decoder = decoder
            self.decoder.trainable = False

        # Set of distinct one-hot-coded inputs and their prior probability
        self.nx_unique = 2 ** n_bits
        self.inputs_unique = tf.one_hot(tf.range(self.nx_unique), self.nx_unique)
        self.prior_prob_inputs = (1. / self.nx_unique) * tf.ones([self.nx_unique])
        # Adaptation parameters defining the per-component affine transformations
        if self.psi_values is None:
            self.params_affine = None
        else:
            self.params_affine = split_adaptation_params(self.psi_values, self.n_mixes, self.n_dims)

    def fit(self, x=None, y=None, **kwargs):
        # Expecting `x` to be a tensor or ndarray
        if self.estimate_input_priors:
            # Estimate the prior probability of each distinct input from the training data
            self.prior_prob_inputs = (1. / x.shape[0]) * tf.reduce_sum(x, axis=0)

        return super(AutoencoderInverseAffine, self).fit(x=x, y=y, **kwargs)

    def call(self, inputs, training=None):
        # Encode the one-hot-represented input messages into symbols
        x = self.encoder(inputs)
        # Predict the Gaussian mixture parameters using the MDN channel model
        params_mdn_orig = self.channel(x)
        if training or (self.params_affine is None):
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn_orig, training=training)
            return self.decoder(y)
        else:
            # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
            params_mdn = self.transform_gmm_params(params_mdn_orig)
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn, training=training)
            # Inverse affine transformation corresponding to each symbol and component is applied to the output IQ
            # samples prior to decoding
            return self.adapted_decoder(y, False)

    def transform_gmm_params(self, params_mdn_orig):
        # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
        mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_orig, self.n_mixes,
                                                                   self.n_dims, logits=True)
        mus, sigmas, pi_logits = affine_transform_gmm_params(mus_orig, sigmas_orig, pi_logits_orig,
                                                             self.params_affine, self.n_mixes, self.n_dims)
        return tf.concat([mus, sigmas, pi_logits], 1)

    def adapted_decoder(self, y, is_predict, batch_size=MAX_BATCH_SIZE, map_assign=False):
        # Distinct symbols `x` output by the encoder network
        x_unique = self.encoder(self.inputs_unique)
        # Gaussian mixture parameters predicted by the MDN on each distinct `x`
        params_mdn_unique_orig = self.channel(x_unique)
        # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
        params_mdn_unique = self.transform_gmm_params(params_mdn_unique_orig)

        # Parameters of the original and adapted Gaussian mixture models
        mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_unique_orig, self.n_mixes, self.n_dims,
                                                                   logits=True)
        mus, sigmas, pi_logits = get_mixture_params(params_mdn_unique, self.n_mixes, self.n_dims, logits=True)
        # Posterior probability of the pair of constellation symbol `x` and the mixture component `i` given the
        # channel output `y`. The function returns only the unnormalized logits
        sizes_arr = tf.constant([self.n_mixes, self.n_dims, self.nx_unique], dtype=tf.int32)
        post_logits = posterior_component_and_x_given_y(
            y, mus, sigmas, pi_logits, self.prior_prob_inputs, sizes_arr
        )
        if map_assign:
            # Select the most probable (symbol, component) pair for each output sample from the posterior. Then apply
            # the corresponding inverse affine transformation before passing the channel outputs to the decoder.
            idx = tf.argmax(post_logits, axis=1, output_type=tf.int64)
            idx_comp = tf.math.floordiv(idx, self.nx_unique)
            idx_symb = tf.math.floormod(idx, self.nx_unique)
            # For each channel output sample, apply the inverse affine transformation corresponding to the selected
            # best (symbol, component) pair
            y = transform_inverse_affine(
                y, mus_orig, mus, tf.math.abs(self.params_affine[2]), idx_symb, idx_comp,
                tf.shape(y, out_type=tf.int64)[0], tf.constant(self.n_dims, dtype=tf.int64)
            )
            # Predictions of the decoder on the transformed channel outputs
            if is_predict:
                preds = self.decoder.predict(y, batch_size=batch_size)
            else:
                preds = self.decoder(y)

        elif not map_assign:
            # First the probabilistic (weighted) sum of the inverse-transformed channel outputs is calculated.
            # The decoder is predicted on this transformed channel output.
            post_proba = tf.nn.softmax(post_logits, axis=1)
            n_dims_ = tf.constant(self.n_dims, dtype=tf.int64)
            n_samp_ = tf.shape(y, out_type=tf.int64)[0]
            # nx_unique_ = tf.constant(self.nx_unique, dtype=tf.int64)
            psi_c = tf.math.abs(self.params_affine[2])
            y_hat_weighted = tf.zeros([n_samp_, self.n_dims])
            for idx_comp in tf.range(self.n_mixes, dtype=tf.int64):
                for idx_symb in tf.range(self.nx_unique, dtype=tf.int64):
                    st = n_dims_ * idx_comp
                    en = n_dims_ * (idx_comp + 1)
                    y_hat = tf.math.add(
                        tf.math.divide(y - mus[idx_symb, st:en], psi_c[st:en]), mus_orig[idx_symb, st:en]
                    )
                    y_hat_curr = tf.math.multiply(
                        y_hat, tf.expand_dims(post_proba[:, self.nx_unique * idx_comp + idx_symb], 1)
                    )
                    y_hat_weighted = tf.math.accumulate_n([y_hat_weighted, y_hat_curr])

            # Predictions of the decoder on the transformed channel outputs
            if is_predict:
                preds = self.decoder.predict(y_hat_weighted, batch_size=batch_size)
            else:
                preds = self.decoder(y_hat_weighted)

        # Else case will never be called (intentional)
        else:
            # Probabilistic (weighted) sum of the decoder predictions
            post_proba = tf.nn.softmax(post_logits, axis=1)
            n_dims_ = tf.constant(self.n_dims, dtype=tf.int64)
            n_samp_ = tf.shape(y, out_type=tf.int64)[0]
            # nx_unique_ = tf.constant(self.nx_unique, dtype=tf.int64)
            psi_c = tf.math.abs(self.params_affine[2])
            preds = tf.zeros([n_samp_, self.nx_unique])
            for idx_comp in tf.range(self.n_mixes, dtype=tf.int64):
                for idx_symb in tf.range(self.nx_unique, dtype=tf.int64):
                    st = n_dims_ * idx_comp
                    en = n_dims_ * (idx_comp + 1)
                    y_hat = tf.math.add(
                        tf.math.divide(y - mus[idx_symb, st:en], psi_c[st:en]), mus_orig[idx_symb, st:en]
                    )
                    # Predictions of the decoder on the transformed channel outputs for the current
                    # (component, symbol) pair
                    if is_predict:
                        preds_curr = self.decoder.predict(y_hat, batch_size=batch_size)
                    else:
                        preds_curr = self.decoder(y_hat)

                    preds_curr = tf.math.multiply(
                        preds_curr, tf.expand_dims(post_proba[:, self.nx_unique * idx_comp + idx_symb], 1)
                    )
                    preds = tf.math.accumulate_n([preds, preds_curr])

        return preds

    def encoder_predict(self, s, batch_size=MAX_BATCH_SIZE):
        return self.encoder.predict(s, batch_size=batch_size)

    def decoder_predict(self, y, batch_size=MAX_BATCH_SIZE):
        if self.params_affine is None:
            # Standard decoder
            return self.decoder.predict(y, batch_size=batch_size)
        else:
            return self.adapted_decoder(y, True, batch_size=batch_size)

    def channel_predict(self, x, batch_size=MAX_BATCH_SIZE):
        return self.channel.predict(x, batch_size=batch_size)

    def get_config(self):
        psi_values = self.psi_values
        if isinstance(self.psi_values, tf.Tensor):
            psi_values = self.psi_values.numpy()

        config = {
            'n_bits': self.n_bits,
            'n_symbols': self.n_symbols,
            'n_hidden': self.n_hidden,
            'scale_outputs': self.scale_outputs,
            'l2_reg_strength': self.l2_reg_strength,
            'estimate_input_priors': self.estimate_input_priors,
            'mdn_model': self.channel,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'psi_values': psi_values
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SymbolEstimationExpectation(Layers.Layer):
    """
    Estimate the channel inputs `x` given the channel outputs `y` using one of the two methods:
    1) Conditional expectation or MMSE estimate: `x_hat = E[x | y]`.
    2) MAP estimate: `x_hat = argmax_x P(x | y)`.
    This layer has no trainable parameters and applies a deterministic nonlinear transformation.
    The estimated channel inputs should have the same dimension as the channel outputs.
    """
    def __init__(self, n_mixes, n_dim, map_estimation=False, temperature=0.01, **kwargs):
        kwargs.update(
            {'name': 'symbol_estim_expec', 'trainable': False}
        )
        super(SymbolEstimationExpectation, self).__init__(**kwargs)
        self.n_mixes = n_mixes
        self.n_dim = n_dim
        self.map_estimation = map_estimation
        self.temperature = temperature

    def call(self, inputs, training=None):
        """
        :param inputs: A list/tuple with the following four tensors: `[y, x_unique, params_mixture, prior_prob_x]`.
                       - `y` is the channel output. Tensor of shape `(n_samp, self.n_dim)`, where `n_samp` is the
                       number of samples.
                       - `x_unique` has the unique channel inputs. Tensor of shape `(nx_unique, self.n_dim)`, where
                       `nx_unique` is the number of unique inputs.
                       - `params_mixture` has the mixture model parameters corresponding to the unique channel inputs.
                       Tensor of shape `(nx_unique, 2 * self.n_mixes * self.n_dim + self.n_mixes)`.
                       - `prior_prob_x` is a tensor with the prior probability of each unique channel input.
                       Has shape `(nx_unique, )`.
        :param training: boolean, whether the call is in inference mode or training mode.

        :return: `x_hat` is an estimate of the channel input. A tensor with the same shape as `y`.
        """
        y, x_unique, params_mixture, prior_prob_x = inputs
        # Component means, variances, and prior logits corresponding to the unique channel inputs
        mus, sigmas, pi_logits = tf.split(
            params_mixture, [self.n_mixes * self.n_dim, self.n_mixes * self.n_dim, self.n_mixes], axis=1
        )
        # Posterior probability over the distinct `x` values given `y`
        sizes_arr = tf.constant([self.n_mixes, self.n_dim, x_unique.shape[0]], dtype=tf.int32)
        post_logits = posterior_x_given_y(y, mus, sigmas, pi_logits, prior_prob_x, sizes_arr)
        if self.map_estimation:
            if training:
                x_hat = tf.linalg.matmul(
                    tf.nn.softmax(tf.math.scalar_mul(1. / self.temperature, post_logits), axis=1), x_unique
                )
            else:
                # During inference, we directly use `argmax` instead of the temperature-scaled softmax
                x_hat = tf.gather(x_unique, tf.argmax(post_logits, axis=1), axis=0, batch_dims=1)
        else:
            x_hat = tf.linalg.matmul(tf.nn.softmax(post_logits, axis=1), x_unique)

        return x_hat

    def get_config(self):
        config = super(SymbolEstimationExpectation, self).get_config()
        config.update(
            {'n_mixes': self.n_mixes,
             'n_dim': self.n_dim,
             'map_estimation': self.map_estimation,
             'temperature': self.temperature
             }
        )
        return config


class AutoencoderSymbolEstimation(tf.keras.Model):
    """
    Class implementing an autoencoder-based communication system with a Gaussian mixture density network (MDN)
    channel model. The channel model is allowed to be adapted using component-conditional affine transformation
    parameters. The conditional expectation of the input symbol to the channel `x`, given the observed channel
    output `y` is used to transform the channel outputs prior to the decoder (receiver) neural network.
    MAP estimation can also be used as an alternative to the conditional expectation method.
    This is used to improve the decoding performance and compensate for changes in the channel conditions.
    """
    def __init__(self, mdn_model, n_bits, n_symbols, n_hidden=None, map_estimation=False, temperature=0.01,
                 scale_outputs=True, l2_reg_strength=0., psi_values=None, encoder=None, decoder=None,
                 estimate_input_priors=False, **kwargs):
        super(AutoencoderSymbolEstimation, self).__init__(name='AutoencoderSymbolEstimation', **kwargs)
        self.n_bits = n_bits
        self.n_symbols = n_symbols
        self.n_hidden = n_hidden
        self.map_estimation = map_estimation
        self.temperature = temperature
        self.scale_outputs = scale_outputs
        self.l2_reg_strength = l2_reg_strength
        self.psi_values = psi_values
        self.estimate_input_priors = estimate_input_priors
        # Encoder dimension and the number of mixture components in the channel model
        self.n_dims = mdn_model.n_dims
        self.n_mixes = mdn_model.n_mixes
        if self.n_dims != (2 * n_symbols):
            raise ValueError("Mismatch in the number of dimensions.")

        if encoder is None:
            self.encoder = Encoder(n_bits, n_symbols, n_hidden=self.n_hidden, l2_reg_strength=self.l2_reg_strength)
        else:
            # Pre-trained encoder
            self.encoder = encoder
            self.encoder.trainable = False

        # Channel MDN model
        self.channel = mdn_model
        self.channel.trainable = False
        # Sampling layer for the channel model
        # self.sampling = SamplingMDN(self.n_mixes, self.n_dims)
        self.sampling = SamplingMDNGumbel(self.n_mixes, self.n_dims)
        # Symbol estimation layer
        self.symbol_estimation = SymbolEstimationExpectation(
            self.n_mixes, self.n_dims, map_estimation=self.map_estimation, temperature=self.temperature
        )
        if decoder is None:
            # Not applying scaling at the decoder because this method estimates the channel input given
            # the channel output
            self.decoder = Decoder(n_bits, n_hidden=self.n_hidden, scale_outputs=False)
        else:
            # Pre-trained decoder
            self.decoder = decoder
            self.decoder.trainable = False

        # Set of distinct one-hot-coded inputs and their prior probability
        self.nx_unique = 2 ** n_bits
        self.inputs_unique = tf.one_hot(tf.range(self.nx_unique), self.nx_unique)
        self.prior_prob_inputs = (1. / self.nx_unique) * tf.ones([self.nx_unique])
        # Average power of the channel outputs from the original channel model and adapted channel model
        # (if applicable). These will be set during training and applied to scale the channel output during inference.
        self.avg_power_output = self.estimate_average_power(False)
        self.avg_power_output_adapted = 1.

        # Adaptation parameters defining the per-component affine transformations
        if self.psi_values is None:
            self.params_affine = None
        else:
            self.params_affine = split_adaptation_params(self.psi_values, self.n_mixes, self.n_dims)
            # Calculate the average power of the channel outputs with the adapted channel parameters
            self.avg_power_output_adapted = self.estimate_average_power(True)

    def fit(self, x=None, y=None, **kwargs):
        # Expecting `x` to be a tensor or ndarray
        if self.estimate_input_priors:
            # Estimate the prior probability of each distinct input from the training data
            self.prior_prob_inputs = (1. / x.shape[0]) * tf.reduce_sum(x, axis=0)

        # Output scaling by the average power ratio is disabled during training
        so = self.scale_outputs
        self.scale_outputs = False
        hist = super(AutoencoderSymbolEstimation, self).fit(x=x, y=y, **kwargs)
        # Reset `self.scale_outputs` and calculate the average power of the channel outputs
        self.scale_outputs = so
        self.avg_power_output = self.estimate_average_power(False)

        return hist

    def call(self, inputs, training=None):
        # Encode the one-hot-represented input messages into symbols
        x = self.encoder(inputs)
        # Predict the Gaussian mixture parameters using the MDN channel model
        params_mdn_orig = self.channel(x)
        if training or (self.params_affine is None):
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn_orig, training=training)
            # Estimation of the channel inputs given channel outputs
            x_hat = self.estimate_x_from_y(y, training=training)
        else:
            # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
            params_mdn = self.transform_gmm_params(params_mdn_orig)
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn, training=training)
            # Estimation of the channel inputs given channel outputs
            x_hat = self.estimate_x_from_y(y, transform=True, training=training)

        return self.decoder(x_hat)

    def estimate_x_from_y(self, y, transform=False, training=None):
        # Unique input symbols
        x_unique = self.encoder(self.inputs_unique)
        # Mixture parameters predicted by the MDN on the unique symbols
        params_mdn_unique = self.channel(x_unique)
        if transform:
            # Transform the mixture parameters using component-conditional affine transformations
            params_mdn_unique = self.transform_gmm_params(params_mdn_unique)

        if self.scale_outputs and (not training):
            # During inference, scale the channel outputs by the square root of the average power ratio
            avg_power = self.avg_power_output_adapted if transform else self.avg_power_output
            avg_power_batch = tf.math.reduce_mean(tf.math.reduce_sum(y ** 2, 1))
            y = tf.math.scalar_mul(tf.math.sqrt(avg_power / avg_power_batch), y)

        return self.symbol_estimation([y, x_unique, params_mdn_unique, self.prior_prob_inputs], training=training)

    def transform_gmm_params(self, params_mdn_orig):
        # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
        mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_orig, self.n_mixes,
                                                                   self.n_dims, logits=True)
        mus, sigmas, pi_logits = affine_transform_gmm_params(mus_orig, sigmas_orig, pi_logits_orig,
                                                             self.params_affine, self.n_mixes, self.n_dims)
        return tf.concat([mus, sigmas, pi_logits], 1)

    def estimate_average_power(self, transform, params_mdn_unique=None, n_samp_per_symbol=100):
        # Estimate the average power of the channel output
        if params_mdn_unique is None:
            # Mixture parameters predicted by the channel MDN model on the unique symbols
            params_mdn_unique = self.channel(self.encoder(self.inputs_unique))
            if transform:
                # Transform the mixture parameters using component-conditional affine transformations
                params_mdn_unique = self.transform_gmm_params(params_mdn_unique)

        # Replicate the MDN parameters `n_samp_per_symbol` times along the rows
        params_rep = tf.tile(params_mdn_unique, tf.constant([n_samp_per_symbol, 1], dtype=tf.int32))
        # Channel outputs sampled from the MDN
        y = self.sampling(params_rep, training=False)

        return tf.math.reduce_mean(tf.math.reduce_sum(y ** 2, 1))

    def encoder_predict(self, s, batch_size=MAX_BATCH_SIZE):
        return self.encoder.predict(s, batch_size=batch_size)

    def decoder_predict(self, y, batch_size=MAX_BATCH_SIZE):
        transform = (self.params_affine is not None)
        x_hat = self.estimate_x_from_y(y, transform=transform, training=False)
        return self.decoder.predict(x_hat, batch_size=batch_size)

    def channel_predict(self, x, batch_size=MAX_BATCH_SIZE):
        return self.channel.predict(x, batch_size=batch_size)

    def get_config(self):
        psi_values = self.psi_values
        if isinstance(self.psi_values, tf.Tensor):
            psi_values = self.psi_values.numpy()

        config = {
            'n_bits': self.n_bits,
            'n_symbols': self.n_symbols,
            'n_hidden': self.n_hidden,
            'map_estimation': self.map_estimation,
            'temperature': self.temperature,
            'scale_outputs': self.scale_outputs,
            'l2_reg_strength': self.l2_reg_strength,
            'estimate_input_priors': self.estimate_input_priors,
            'mdn_model': self.channel,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'psi_values': psi_values
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AutoencoderAdaptGenerative(tf.keras.Model):
    """
    Class implementing an autoencoder-based communication system with a Gaussian mixture density network (MDN)
    channel model. The channel model is allowed to be adapted using component-conditional affine transformation
    parameters.
    The adaptation method at the decoder is a more general approach compared to that applied in
    `AutoencoderInverseAffine`. Any generative model (not only MDN) can be used with this method. At a high level,
    the channel outputs (IQ samples) are first passed through a MAP transformation based on the adapted channel
    probability distribution, followed by a sample expectation of the decoder's outputs w.r.t to the original channel
    probability distribution. More details can be found in the paper.
    """
    def __init__(self, mdn_model, n_bits, n_symbols, n_hidden=None, scale_outputs=True, l2_reg_strength=0.,
                 psi_values=None, encoder=None, decoder=None, estimate_input_priors=False, **kwargs):
        super(AutoencoderAdaptGenerative, self).__init__(name='AutoencoderAdaptGenerative', **kwargs)
        self.n_bits = n_bits
        self.n_symbols = n_symbols
        self.n_hidden = n_hidden
        self.scale_outputs = scale_outputs
        self.l2_reg_strength = l2_reg_strength
        self.psi_values = psi_values
        self.estimate_input_priors = estimate_input_priors
        # Encoder dimension and the number of mixture components in the channel model
        self.n_dims = mdn_model.n_dims
        self.n_mixes = mdn_model.n_mixes
        if self.n_dims != (2 * n_symbols):
            raise ValueError("Mismatch in the number of dimensions.")

        if encoder is None:
            self.encoder = Encoder(n_bits, n_symbols, n_hidden=self.n_hidden, l2_reg_strength=self.l2_reg_strength)
        else:
            # Pre-trained encoder
            self.encoder = encoder
            self.encoder.trainable = False

        # Channel MDN model
        self.channel = mdn_model
        self.channel.trainable = False
        # Sampling layer for the channel model
        # self.sampling = SamplingMDN(self.n_mixes, self.n_dims)
        self.sampling = SamplingMDNGumbel(self.n_mixes, self.n_dims)
        if decoder is None:
            self.decoder = Decoder(n_bits, n_hidden=self.n_hidden, scale_outputs=self.scale_outputs)
        else:
            # Pre-trained decoder
            self.decoder = decoder
            self.decoder.trainable = False

        # Sample average of the decoder predictions on the unique input symbols. To keep the prediction fast, this
        # is only computed once after the autoencoder is trained.
        self.avg_decoder_preds_unique = None
        # Set of distinct one-hot-coded inputs and their prior probability
        self.nx_unique = 2 ** n_bits
        self.inputs_unique = tf.one_hot(tf.range(self.nx_unique), self.nx_unique)
        self.prior_prob_inputs = (1. / self.nx_unique) * tf.ones([self.nx_unique])
        # Adaptation parameters defining the per-component affine transformations
        if self.psi_values is None:
            self.params_affine = None
        else:
            self.params_affine = split_adaptation_params(self.psi_values, self.n_mixes, self.n_dims)

    def fit(self, x=None, y=None, **kwargs):
        # Expecting `x` to be a tensor or ndarray
        if self.estimate_input_priors:
            # Estimate the prior probability of each distinct input from the training data
            self.prior_prob_inputs = (1. / x.shape[0]) * tf.reduce_sum(x, axis=0)

        return super(AutoencoderAdaptGenerative, self).fit(x=x, y=y, **kwargs)

    def call(self, inputs, training=None):
        # Encode the one-hot-represented input messages into symbols
        x = self.encoder(inputs)
        # Predict the Gaussian mixture parameters using the MDN channel model
        params_mdn_orig = self.channel(x)
        if training or (self.params_affine is None):
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn_orig, training=training)
            return self.decoder(y)
        else:
            # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
            params_mdn = self.transform_gmm_params(params_mdn_orig)
            # Channel outputs sampled from the MDN. Should have shape `(inputs.shape[0], self.n_dims)`
            y = self.sampling(params_mdn, training=training)
            # Apply the adapted decoder on the channel outputs
            return self.adapted_decoder(y, False)

    def transform_gmm_params(self, params_mdn_orig):
        # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
        mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_orig, self.n_mixes,
                                                                   self.n_dims, logits=True)
        mus, sigmas, pi_logits = affine_transform_gmm_params(mus_orig, sigmas_orig, pi_logits_orig,
                                                             self.params_affine, self.n_mixes, self.n_dims)
        return tf.concat([mus, sigmas, pi_logits], 1)

    def adapted_decoder(self, y, is_predict, batch_size=MAX_BATCH_SIZE, n_samp_per_symbol=100):
        """
        The channel outputs (IQ samples) are first passed through a MAP transformation based on the adapted
        channel probability distribution, followed by a sample expectation of the decoder's outputs w.r.t to
        the original channel probability distribution.
        """
        # Unique encoded symbols
        x_unique = self.encoder(self.inputs_unique)
        # Gaussian mixture parameters predicted by the MDN on the unique symbols
        params_mdn_unique_orig = self.channel(x_unique)
        # Transform the Gaussian mixture parameters using component-conditional affine transformations
        params_mdn_unique = self.transform_gmm_params(params_mdn_unique_orig)
        # Component means, variances, and prior logits of the adapted Gaussian mixture
        mus, sigmas, pi_logits = get_mixture_params(params_mdn_unique, self.n_mixes, self.n_dims, logits=True)

        # Posterior probability of the unique input symbols given the channel outputs w.r.t the adapted Gaussian mixture
        sizes_arr = tf.constant([self.n_mixes, self.n_dims, self.nx_unique], dtype=tf.int32)
        post_logits = posterior_x_given_y(y, mus, sigmas, pi_logits, self.prior_prob_inputs, sizes_arr)
        # Most probable input symbols corresponding to the channel outputs (based on the posterior of the adapted
        # Gaussian mixture)
        # x_hat = tf.gather(x_unique, tf.math.argmax(post_logits, axis=1), axis=0, batch_dims=1)

        preds_avg = self.avg_decoder_preds_unique
        if self.avg_decoder_preds_unique is None:
            # Replicate the MDN parameters corresponding to the unique symbols `n_samp_per_symbol` times along the rows
            params_rep = tf.tile(params_mdn_unique_orig, tf.constant([n_samp_per_symbol, 1], dtype=tf.int32))
            # Generate random samples from the channel conditional density
            y_samp = self.sampling(params_rep, training=False)
            # Decoder's predictions on the random samples
            if is_predict:
                preds_samp = self.decoder.predict(y_samp, batch_size=batch_size)
            else:
                preds_samp = self.decoder(y_samp)

            ind_slice_base = self.nx_unique * tf.range(n_samp_per_symbol, dtype=tf.int32)
            preds_avg_ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for i in tf.range(self.nx_unique, dtype=tf.int32):
                # Average of the decoder's predictions over the random samples from the i-th symbol
                preds_curr = tf.gather(preds_samp, ind_slice_base + i, axis=0, batch_dims=1)
                preds_avg_ta = preds_avg_ta.write(i, tf.math.reduce_mean(preds_curr, axis=0))

            # Should be a tensor of shape `[self.nx_unique, self.nx_unique]`
            preds_avg = preds_avg_ta.stack()
            if is_predict:
                # During inference, the sample average of the decoder predictions given each input symbol can be
                # computed once and reused
                self.avg_decoder_preds_unique = preds_avg

        return tf.linalg.matmul(tf.nn.softmax(post_logits, axis=1), preds_avg)

    def encoder_predict(self, s, batch_size=MAX_BATCH_SIZE):
        return self.encoder.predict(s, batch_size=batch_size)

    def decoder_predict(self, y, batch_size=MAX_BATCH_SIZE):
        if self.params_affine is None:
            # Standard decoder
            return self.decoder.predict(y, batch_size=batch_size)
        else:
            return self.adapted_decoder(y, True, batch_size=batch_size)

    def channel_predict(self, x, batch_size=MAX_BATCH_SIZE):
        return self.channel.predict(x, batch_size=batch_size)

    def get_config(self):
        psi_values = self.psi_values
        if isinstance(self.psi_values, tf.Tensor):
            psi_values = self.psi_values.numpy()

        config = {
            'n_bits': self.n_bits,
            'n_symbols': self.n_symbols,
            'n_hidden': self.n_hidden,
            'scale_outputs': self.scale_outputs,
            'l2_reg_strength': self.l2_reg_strength,
            'estimate_input_priors': self.estimate_input_priors,
            'mdn_model': self.channel,
            'encoder': self.encoder,
            'decoder': self.decoder,
            'psi_values': psi_values
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
