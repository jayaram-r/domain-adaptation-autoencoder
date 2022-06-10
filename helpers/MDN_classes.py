# Code related to the mixture density network
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as Layers
import tensorflow.keras.optimizers as Optimizer
import math
from .context import *
import MDN_base
from helpers.constants import *


def load_channel_model_from_file(filename, n_dims, n_components, nx_unique):
    # Specify custom functions and classes to avoid deserialization errors
    custom_objects = {
        'mdn_loss_func': MDN_base.get_mixture_loss_func(n_dims, n_components),
        'mdn_loss_func_posterior': MDN_base.get_mixture_loss_func_posterior(n_dims, n_components, nx_unique),
        'MDN': MDN_base.MDN,
        'MDN_model': MDN_model,
        'MDN_model_disc': MDN_model_disc
    }
    mdn_model = tf.keras.models.load_model(filename, custom_objects=custom_objects, compile=True)

    return mdn_model


def initialize_MDN_model(n_comp, n_dim, n_hidden, lr=LEARNING_RATE_ADAM):
    # Initialize and compile the MDN model
    mdn_model = MDN_model(n_hidden, n_dim, n_comp)
    # Loss function is the negative log-likelihood
    loss_mdn = MDN_base.get_mixture_loss_func(n_dim, n_comp)
    # Default learning rate is 0.001
    optim_obj = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=EPSILON_ADAM)
    mdn_model.compile(loss=loss_mdn, optimizer=optim_obj)
    _ = mdn_model(tf.keras.Input(shape=(n_dim,)))

    return mdn_model


def initialize_MDN_model_disc(n_comp, n_dim, n_hidden, x_unique, lr=LEARNING_RATE_ADAM):
    # Initialize and compile a model instance of the class `MDN_model_disc`
    mdn_model = MDN_model_disc(n_hidden, n_dim, n_comp, x_unique)
    # Loss function is the negative posterior log-likelihood
    loss_mdn = MDN_base.get_mixture_loss_func_posterior(n_dim, n_comp, x_unique.shape[0])
    optim_obj = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=EPSILON_ADAM)
    mdn_model.compile(loss=loss_mdn, optimizer=optim_obj)
    _ = mdn_model(tf.keras.Input(shape=(n_dim,)))

    return mdn_model


# @tf.keras.utils.register_keras_serializable()
class MDN_model(tf.keras.Model):
    # Wrapper class for the mixture density network model
    def __init__(self, n_hidden, n_dims, n_mixes, **kwargs):
        super(MDN_model, self).__init__(name='MDN_Channel', **kwargs)
        self.n_hidden = n_hidden
        self.n_dims = n_dims
        self.n_mixes = n_mixes
        self.dim_output = 2 * n_mixes * n_dims + n_mixes
        self.dense1 = Layers.Dense(n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.dense2 = Layers.Dense(n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.mdn = MDN_base.MDN(n_dims, n_mixes)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mdn_out = self.mdn(x)
        return mdn_out

    def get_config(self):
        # NOTE: `get_config` is not implemented by the `tf.keras.Model` class
        # base_config = super(MDN_model, self).get_config()
        config = {
            'n_hidden': self.n_hidden,
            'n_dims': self.n_dims,
            'n_mixes': self.n_mixes
        }
        return config

    # Same as the default implementation of the `tf.keras.Layer` class
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MDN_model_disc(tf.keras.Model):
    # Variation of the `MDN_model` class that trains discriminatively using a posterior log-likelihood as the
    # loss function. Makes sense to use only when the set of inputs is discrete and not large.
    def __init__(self, n_hidden, n_dims, n_mixes, x_unique, **kwargs):
        super(MDN_model_disc, self).__init__(name='MDN_Channel_disc', **kwargs)
        self.n_hidden = n_hidden
        self.n_dims = n_dims
        self.n_mixes = n_mixes
        self.dim_output = 2 * n_mixes * n_dims + n_mixes
        # Unique inputs (`x` vectors)
        self.set_unique_x(x_unique)
        self.return_augmented = False
        # Prior probability of the unique inputs - set to uniform
        self.log_prior = tf.math.log(1. / self.x_unique.shape[0])
        # Layers
        self.dense1 = Layers.Dense(n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.dense2 = Layers.Dense(n_hidden, activation='relu',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED_WEIGHTS))
        self.mdn = MDN_base.MDN(n_dims, n_mixes)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mdn_out = self.mdn(x)
        if self.return_augmented:
            # During training the parameters predicted on the unique inputs are concatenated to the output.
            # This is used to compute the posterior log-likelihood loss function. This is not needed during inference.
            preds_unique = self.mdn(self.dense2(self.dense1(self.x_unique)))
            # For a batch of size `b` and `m` distinct inputs, the output has size `(b + m, self.dim_output)`
            return tf.concat([mdn_out, preds_unique], axis=0)
        else:
            return mdn_out

    def fit(self, x=None, y=None, **kwargs):
        # During training, the `call` method augments the output with the parameters predicted on the unique inputs.
        # This is used for calculating the loss function. During inference, it returns only the parameters predicted
        # on the input batch.
        self.return_augmented = True
        ret = super(MDN_model_disc, self).fit(x=x, y=y, **kwargs)
        self.return_augmented = False

        return ret

    def evaluate(self, x=None, y=None, **kwargs):
        val = self.return_augmented
        self.return_augmented = True
        ret = super(MDN_model_disc, self).evaluate(x=x, y=y, **kwargs)
        self.return_augmented = val

        return ret

    def set_unique_x(self, x_unique):
        if isinstance(x_unique, tf.Tensor):
            self.x_unique = x_unique
        else:
            self.x_unique = tf.convert_to_tensor(x_unique, dtype=DTYPE_TF)

    def get_config(self):
        # NOTE: `get_config` is not implemented by the `tf.keras.Model` class
        if isinstance(self.x_unique, tf.Tensor):
            x_unique = self.x_unique.numpy()
        else:
            x_unique = self.x_unique

        config = {
            'n_hidden': self.n_hidden,
            'n_dims': self.n_dims,
            'n_mixes': self.n_mixes,
            'x_unique': x_unique
        }
        return config

    # Same as the default implementation of the `tf.keras.Layer` class
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Normalizedcomplex(Layers.Layer):
    def __init__(self):
        super(Normalizedcomplex, self).__init__(name='Normalizedcomplex')

    def build(self, input_shape):
        self.N = int(input_shape[1] / 2.)

    def call(self, inputs):
        x = tf.nn.l2_normalize(inputs, axis=1) * np.sqrt(self.N)
        x = tf.reshape(x, [-1, self.N, 2])
        return tf.complex(x[:, :, 0], x[:, :, 1])


class MDN_model_extra_layer(tf.keras.Model):
    # A baseline method for adapting the MDN channel model.
    # Model augmenting the mixture density network with an additional fully connected layer.
    # Freezing the weights of the MDN and fine tuning the weights of the additional layer.
    def __init__(self, mdn_model, **kwargs):
        super(MDN_model_extra_layer, self).__init__(name='MDN_model_extra_layer', **kwargs)
        # Pretrained mixture density network whose weights are kept frozen
        self.mdn_model = mdn_model
        self.mdn_model.trainable = False
        # Number of mixture components, input and output dimension
        self.n_mixes = self.mdn_model.mdn.num_mixtures
        self.n_dims = self.mdn_model.mdn.output_dimension
        self.dim_output = self.mdn_model.dim_output

        kernel_initializer = tf.keras.initializers.Constant(1.)
        bias_initializer = tf.keras.initializers.Constant(0.)
        self.dense_mus = [
            Layers.Dense(1, activation='linear',
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)
            for _ in range(self.n_dims * self.n_mixes)
        ]
        # Layers predicting the standard deviation parameters. Weight is constrained to be positive and there
        # is no bias term.
        # https://www.tensorflow.org/api_docs/python/tf/keras/constraints
        self.dense_sigmas = [
            Layers.Dense(1, activation=MDN_base.relu_plus_epsilon,
                         kernel_initializer=kernel_initializer,
                         use_bias=False,
                         # kernel_constraint=tf.keras.constraints.NonNeg(),
                         # bias_constraint=tf.keras.constraints.MaxNorm(max_value=0.)
                         )
            for _ in range(self.n_dims * self.n_mixes)
        ]
        self.dense_pis = [
            Layers.Dense(1, activation='linear',
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer)
            for _ in range(self.n_mixes)
        ]
        self.dense = self.dense_mus + self.dense_sigmas + self.dense_pis
        self.len_dense = len(self.dense)

    def call(self, inputs):
        x = self.mdn_model(inputs)
        x = tf.concat([
            self.dense[i](tf.expand_dims(x[:, i], 1))
            for i in range(self.len_dense)
        ], 1)
        return x

    def get_config(self):
        return {'mdn_model': self.mdn_model}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
