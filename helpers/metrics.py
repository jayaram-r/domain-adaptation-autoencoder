# Validation metric for selecting lambda and evaluating the autoencoder or MDN
import numpy as np
import tensorflow as tf
from .context import *
import MDN_base
from helpers.utils import (
    split_adaptation_params,
    affine_transform_gmm_params,
    get_mixture_params
)
import helpers.autoencoder_classes
from helpers.constants import *


def validation_metric_mdn(mdn_model, psi_values, x_unique, prior_proba_x, inputs_test, x_test, y_test,
                          metric='inverse_trans_log_like', map_assign=False):
    '''
    Validation metric for selecting `lambda` in the adaptation objective.
    Evaluating the adapted MDN channel on samples from the target distribution.
    The samples are transformed using the class and component conditional inverse-affine transformations.
    The metric is either the log-likelihood or the log-posterior-likelihood of the inverse-affine-transformed
    samples relative to the original (source) MDN channel distribution.
    '''
    assert metric in ('inverse_trans_log_like', 'inverse_trans_log_post'), "{} is not a valid metric".format(metric)
    n_comp = mdn_model.n_mixes
    n_dims = mdn_model.n_dims
    n_unique = x_unique.shape[0]
    # batch_size = max(BATCH_SIZE_PRED, 2 * n_unique)
    # Affine transformation parameters for adaptation
    params_affine = split_adaptation_params(psi_values, n_comp, n_dims)
    # Gaussian mixture parameters predicted by the MDN on the distinct `x`
    params_mdn_unique_orig = mdn_model.predict(x_unique)
    # Parameters of the original and adapted Gaussian mixture models
    mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_unique_orig, n_comp, n_dims, logits=True)
    mus, sigmas, pi_logits = affine_transform_gmm_params(mus_orig, sigmas_orig, pi_logits_orig, params_affine,
                                                         n_comp, n_dims)
    # Posterior probability of the pair of constellation symbol `x` and the mixture component `i` given the
    # channel output `y`. The function returns the unnormalized logits.
    sizes_arr = tf.constant([n_comp, n_dims, n_unique], dtype=tf.int32)
    post_logits = helpers.autoencoder_classes.posterior_component_and_x_given_y(
        y_test, mus, sigmas, pi_logits, prior_proba_x, sizes_arr
    )
    if map_assign:
        # Multiply the posterior logits element-wise with the one-hot-coded labels. This will ensure that only the
        # component corresponding to the class (symbol) of each sample will get selected
        post_logits = tf.math.multiply(post_logits, tf.tile(inputs_test, tf.constant([1, n_comp])))

        # Select the most probable component for each output sample from the posterior. Then apply the corresponding
        # inverse affine transformation.
        idx = tf.argmax(post_logits, axis=1, output_type=tf.int64)
        idx_comp = tf.math.floordiv(idx, n_unique)
        idx_symb = tf.math.floormod(idx, n_unique)
        y_test_trans = helpers.autoencoder_classes.transform_inverse_affine(
            y_test, mus_orig, mus, tf.math.abs(params_affine[2]), idx_symb, idx_comp,
            tf.shape(y_test, out_type=tf.int64)[0], tf.constant(n_dims, dtype=tf.int64)
        )
    else:
        # Multiply the posterior probability element-wise with the one-hot-coded labels. This will ensure that only the
        # component corresponding to the class (symbol) of each sample will get selected
        post_proba = tf.math.multiply(tf.nn.softmax(post_logits, axis=1),
                                      tf.tile(inputs_test, tf.constant([1, n_comp])))
        # Calculate `P(i | s, y)` for component `i` and symbol `s`, where `s` is the true class/symbol
        den = tf.clip_by_value(tf.math.reduce_sum(post_proba, axis=1), 1e-16, 1.)
        post_proba_cond = tf.math.divide(post_proba, tf.expand_dims(den, 1))
        psi_c = tf.math.abs(params_affine[2])
        y_test_trans = tf.zeros(tf.shape(y_test))
        for idx_comp in range(n_comp):
            for idx_symb in range(n_unique):
                st = n_dims * idx_comp
                en = n_dims * (idx_comp + 1)
                y_hat = tf.math.add(
                    tf.math.divide(y_test - mus[idx_symb, st:en], psi_c[st:en]), mus_orig[idx_symb, st:en]
                )
                y_hat_curr = tf.math.multiply(
                    y_hat, tf.expand_dims(post_proba_cond[:, n_unique * idx_comp + idx_symb], 1)
                )
                y_test_trans = tf.math.accumulate_n([y_test_trans, y_hat_curr])

    if metric == 'inverse_trans_log_like':
        # Negative log-likelihood loss
        loss_mdn = MDN_base.get_mixture_loss_func(n_dims, n_comp)
        # Output parameters of the MDN on the encoded test inputs
        params_test = mdn_model.predict(x_test)
    else:
        # Negative posterior log-likelihood loss
        loss_mdn = MDN_base.get_mixture_loss_func_posterior(n_dims, n_comp, n_unique)
        # Output parameters of the MDN on the encoded test inputs
        params_test = mdn_model.predict(tf.concat([x_test, x_unique], axis=0))

    return loss_mdn(y_test_trans, params_test)


def calculate_metric_autoencoder(autoencoder, inputs_test, y_test, metric='log_loss'):
    # Calculate the log-loss or error rate of the autoencoder on the given test samples.
    # `inputs_test` is a TF tensor of one-hot-coded inputs.
    # `y_test` is a TF tensor of the channel outputs (decoder inputs).
    n_samp = inputs_test.shape[0]
    # x_test = autoencoder.encoder_predict(inputs_test)

    # Don't use a large batch size here since the scale factor at the decoder is calculated from the batch of outputs
    batch_size = max(BATCH_SIZE_PRED, 2 * autoencoder.inputs_unique.shape[0])
    # Predict the decoder to get the probability over the distinct symbols
    prob_output = autoencoder.decoder_predict(y_test, batch_size=batch_size)
    if metric == 'log_loss':
        return -1. * tf.math.reduce_mean(tf.math.reduce_sum(tf.math.xlogy(inputs_test, prob_output), axis=1))

    elif metric == 'error_rate':
        preds = tf.math.argmax(prob_output, axis=1)
        labels = tf.math.argmax(inputs_test, axis=1)
        mask_errors = (preds != labels)
        ber = mask_errors.numpy().astype(np.float).sum() / n_samp
        # Set BER of exactly 0 to a small non-zero value (avoids taking log(0) in the BER plots)
        return max(ber, 0.1 / n_samp)

    elif (metric == 'inverse_trans_log_like') or (metric == 'inverse_trans_log_post'):
        '''
        Evaluating the adapted MDN channel on samples from the target distribution.
        The samples are transformed using the class and component conditional inverse-affine transformations.
        The metric is either the log-likelihood or the log-posterior-likelihood of the inverse-affine-transformed 
        samples relative to the original (source) MDN channel distribution.
        This metric does not depend on the decoder's performance.
        '''
        if autoencoder.params_affine is None:
            raise ValueError("Affine transformation parameters of the autoencoder have not been set. Cannot compute "
                             "this metric.")

        mdn_model = autoencoder.channel
        n_comp = autoencoder.n_mixes
        n_dims = autoencoder.n_dims
        n_unique = autoencoder.nx_unique
        # Distinct symbols `x` output by the encoder network
        x_unique = autoencoder.encoder_predict(autoencoder.inputs_unique)
        # Gaussian mixture parameters predicted by the MDN on each distinct `x`
        params_mdn_unique_orig = autoencoder.channel_predict(x_unique)
        # Transform the Gaussian mixture parameters using the per-component affine transformation parameters
        params_mdn_unique = autoencoder.transform_gmm_params(params_mdn_unique_orig)
        # Parameters of the original and adapted Gaussian mixture models
        mus_orig, sigmas_orig, pi_logits_orig = get_mixture_params(params_mdn_unique_orig, n_comp, n_dims, logits=True)
        mus, sigmas, pi_logits = get_mixture_params(params_mdn_unique, n_comp, n_dims, logits=True)

        # Posterior probability of the pair of constellation symbol `x` and the mixture component `i` given the
        # channel output `y`. The function returns the unnormalized logits.
        sizes_arr = tf.constant([n_comp, n_dims, n_unique], dtype=tf.int32)
        post_logits = helpers.autoencoder_classes.posterior_component_and_x_given_y(
            y_test, mus, sigmas, pi_logits, autoencoder.prior_prob_inputs, sizes_arr
        )
        # Multiply the posterior logits element-wise with the one-hot-coded labels. This will ensure that only the
        # component corresponding to the class (symbol) of each sample will get selected
        post_logits = tf.math.multiply(post_logits, tf.tile(inputs_test, tf.constant([1, n_comp])))

        # Select the most probable (symbol, component) pair for each output sample from the posterior. Then apply
        # the corresponding inverse affine transformation.
        idx = tf.argmax(post_logits, axis=1, output_type=tf.int64)
        idx_comp = tf.math.floordiv(idx, n_unique)
        idx_symb = tf.math.floormod(idx, n_unique)
        # For each channel output sample, apply the inverse affine transformation corresponding to the selected
        # best (symbol, component) pair
        y_test_trans = helpers.autoencoder_classes.transform_inverse_affine(
            y_test, mus_orig, mus, tf.math.abs(autoencoder.params_affine[2]), idx_symb, idx_comp,
            tf.shape(y_test, out_type=tf.int64)[0], tf.constant(n_dims, dtype=tf.int64)
        )
        x_test = autoencoder.encoder_predict(inputs_test, batch_size=batch_size)
        if metric == 'inverse_trans_log_like':
            # Negative log-likelihood loss
            loss_mdn = MDN_base.get_mixture_loss_func(n_dims, n_comp)
            # Output parameters of the MDN on the encoded test inputs
            params_test = mdn_model.predict(x_test)
        else:
            # Negative posterior log-likelihood loss
            loss_mdn = MDN_base.get_mixture_loss_func_posterior(n_dims, n_comp, n_unique)
            # Output parameters of the MDN on the encoded test inputs
            params_test = mdn_model.predict(tf.concat([x_test, x_unique], axis=0))

        return loss_mdn(y_test_trans, params_test)

    else:
        raise ValueError("Invalid value '{}' specified for the input 'metric'".format(metric))
