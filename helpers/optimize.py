# Loss functions and optimization related code
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as Optimizer
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from scipy.spatial.distance import cdist
from functools import partial
from .context import *
from helpers.utils import (
    update_gmm_params,
    split_adaptation_params,
    affine_transform_gmm_params,
    get_mixture_params
)
import helpers.autoencoder_classes
import helpers.MDN_classes
from helpers.constants import *


def get_loss_func_proposed(mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x,
                           n_comp, n_dim, lambda_value=1., type_objec='log_likelihood'):
    # Construct the loss function and its gradient for the proposed channel adaptation method.

    # Declaring as TF constants for better compatibility with compilation using `tf.function`
    n_comp = tf.constant(n_comp, dtype=tf.int64)
    n_dim = tf.constant(n_dim, dtype=tf.int64)
    n_symb = tf.constant(unique_x.shape[0], dtype=tf.int64)
    # n_target = target_domain_x.shape[0]
    # GMM parameters (output of the MDN) predicted on the target domain training data.
    # `source_mus` and `source_sigmas` should each have shape `(n_target, n_comp * n_dim)`.
    # `source_pi_logits` should have shape `(n_target, n_comp)`.
    y_temp = mdn_model(target_domain_x)
    source_mus, source_sigmas, source_pi_logits = get_mixture_params(y_temp, n_comp, n_dim, logits=True)
    # GMM parameters predicted on the distinct channel inputs.
    # `constellation_mus` and `constellation_sigmas` should each have shape `(n_symb, n_comp * n_dim)`.
    # `constellation_pi_logits` should have shape `(n_symb, n_comp)`.
    y_temp = mdn_model(unique_x)
    constellation_mus, constellation_sigmas, constellation_pi_logits = get_mixture_params(
        y_temp, n_comp, n_dim, logits=True
    )

    if type_objec == 'log_likelihood':
        # Loss function based on the negative log-likelihood `log p(y_i | x_i)`
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def loss_func_proposed_loglike(psi_values):
            # First term of the loss function: negative log-likelihood of `target_domain_y` under the adapted
            # Gaussian mixture distribution
            # print("Tracing 'loss_func_proposed_loglike'...You should see this line only during the first call.")
            psi_a, psi_b, psi_c, psi_beta, psi_gamma = tf.split(
                psi_values, [n_comp * n_dim * n_dim, n_comp * n_dim, n_comp * n_dim, n_comp, n_comp]
            )
            # psi_a, psi_b, psi_c, psi_beta, psi_gamma = split_adaptation_params(psi_values, n_comp, n_dim)

            # Adapted Gaussian mixture parameters
            new_mus, new_sigmas, new_pi_logits = update_gmm_params(
                source_mus, source_sigmas, source_pi_logits, psi_a, psi_b, psi_c, psi_beta, psi_gamma, n_comp, n_dim
            )
            '''
            # The block below cannot be compiled using tf.function.
            # Construct the target Gaussian mixture distribution
            comps = [
                tfd.MultivariateNormalDiag(loc=new_mus[:, (i * n_dim): ((i + 1) * n_dim)],
                                           scale_diag=new_sigmas[:, (i * n_dim): ((i + 1) * n_dim)])
                for i in tf.range(n_comp)
            ]
            gm_dist = tfd.Mixture(cat=tfd.Categorical(logits=new_pi_logits), components=comps)
            first_term = -1. * tf.reduce_mean(gm_dist.log_prob(target_domain_y))
            '''
            # First term in the training loss
            temp_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            new_pi_log_softmax = tf.nn.log_softmax(new_pi_logits, axis=1)
            # offset = tf.reduce_logsumexp(new_pi_logits, 1)
            for i in tf.range(n_comp, dtype=tf.int64):
                mvn = tfd.MultivariateNormalDiag(loc=new_mus[:, (i * n_dim): ((i + 1) * n_dim)],
                                                 scale_diag=new_sigmas[:, (i * n_dim): ((i + 1) * n_dim)])
                temp_arr = temp_arr.write(tf.cast(i, tf.int32),
                                          mvn.log_prob(target_domain_y) + new_pi_log_softmax[:, i])
                # temp_arr = temp_arr.write(tf.cast(i, tf.int32),
                #                           mvn.log_prob(target_domain_y) + new_pi_logits[:, i] - offset)

            first_term = -1. * tf.reduce_mean(tf.reduce_logsumexp(temp_arr.stack(), 0))

            # Adapted Gaussian mixture parameters corresponding to the constellation symbols
            new_const_mus, new_const_sigmas, new_const_pi_logits = update_gmm_params(
                constellation_mus, constellation_sigmas, constellation_pi_logits, psi_a, psi_b, psi_c, psi_beta,
                psi_gamma, n_comp, n_dim
            )
            # Second term of the loss function: KL divergence between the component prior probabilities
            constellation_pi_proba = tf.nn.softmax(constellation_pi_logits)
            temp_sum = tf.reduce_sum(
                tf.multiply(constellation_pi_proba,
                            tf.nn.log_softmax(constellation_pi_logits) - tf.nn.log_softmax(new_const_pi_logits)),
                axis=1
            )
            second_term = tf.reduce_sum(tf.multiply(prior_prob_x, temp_sum))

            # Third term of the loss function (KL divergence between two multivariate gaussians)
            shape_inter = tf.stack([n_symb, n_comp, n_dim])
            third_term = kl_divergence_gaussians(constellation_mus, constellation_sigmas, new_const_mus,
                                                 new_const_sigmas, constellation_pi_proba, prior_prob_x, shape_inter)

            return first_term + lambda_value * (second_term + third_term)

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def value_and_gradient_proposed_loglike(psi_values):
            loss = loss_func_proposed_loglike(psi_values)
            grad_loss = tf.gradients(loss, psi_values)[0]

            return loss, grad_loss

        return loss_func_proposed_loglike, value_and_gradient_proposed_loglike
    elif type_objec == 'log_posterior':
        '''
        Most of the function `loss_func_proposed_logpost` overlaps with `loss_func_proposed_loglike`.
        Only the first term of the loss function is different. In case of any changes or bug fixes to one of them, 
        make sure the same change is made to the other function.
        '''
        # Label the rows of `target_domain_x` with the index of the constellation points in `unique_x`
        d_mat = cdist(target_domain_x.numpy(), unique_x.numpy(), metric='euclidean')
        labels_x = tf.math.argmin(d_mat, axis=1)

        # Loss function based on the negative posterior log-likelihood `log p(x_i | y_i)`
        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def loss_func_proposed_logpost(psi_values):
            # First term of the loss function: negative log-posterior of `target_domain_x` given `target_domain_y`
            # under the adapted Gaussian mixture distribution.
            # print("Tracing 'loss_func_proposed_logpost'...You should see this line only during the first call.")
            psi_a, psi_b, psi_c, psi_beta, psi_gamma = tf.split(
                psi_values, [n_comp * n_dim * n_dim, n_comp * n_dim, n_comp * n_dim, n_comp, n_comp]
            )
            # psi_a, psi_b, psi_c, psi_beta, psi_gamma = split_adaptation_params(psi_values, n_comp, n_dim)

            # Adapted Gaussian mixture parameters corresponding to the distinct constellation (channel) inputs
            new_const_mus, new_const_sigmas, new_const_pi_logits = update_gmm_params(
                constellation_mus, constellation_sigmas, constellation_pi_logits, psi_a, psi_b, psi_c, psi_beta,
                psi_gamma, n_comp, n_dim
            )
            # Posterior probability over the distinct `x` values given `y` values in `target_domain_y`
            sizes_arr = tf.cast(tf.stack([n_comp, n_dim, n_symb]), tf.int32)
            logits_post_prob_mat = helpers.autoencoder_classes.posterior_x_given_y(
                target_domain_y, new_const_mus, new_const_sigmas, new_const_pi_logits, prior_prob_x, sizes_arr
            )
            temp_ten = tf.nn.log_softmax(logits_post_prob_mat, axis=1)
            # `tf.gather` selects the specified column index from each row of `temp_ten`. This is equivalent to
            # multiplying `temp_ten` with a one-hot matrix of the same shape, but more efficient.
            first_term = -1. * tf.reduce_mean(tf.gather(temp_ten, labels_x, axis=1, batch_dims=1))

            # Second term of the loss function: KL divergence between the component prior probabilities
            constellation_pi_proba = tf.nn.softmax(constellation_pi_logits)
            temp_sum = tf.reduce_sum(
                tf.multiply(constellation_pi_proba,
                            tf.nn.log_softmax(constellation_pi_logits) - tf.nn.log_softmax(new_const_pi_logits)),
                axis=1
            )
            second_term = tf.reduce_sum(tf.multiply(prior_prob_x, temp_sum))

            # Third term of the loss function (KL divergence between two multivariate gaussians)
            shape_inter = tf.stack([n_symb, n_comp, n_dim])
            third_term = kl_divergence_gaussians(constellation_mus, constellation_sigmas, new_const_mus,
                                                 new_const_sigmas, constellation_pi_proba, prior_prob_x, shape_inter)

            return first_term + lambda_value * (second_term + third_term)

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def value_and_gradient_proposed_logpost(psi_values):
            loss = loss_func_proposed_logpost(psi_values)
            grad_loss = tf.gradients(loss, psi_values)[0]

            return loss, grad_loss

        return loss_func_proposed_logpost, value_and_gradient_proposed_logpost


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # constellation_mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # constellation_sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # new_constellation_mus
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # new_constellation_sigmas
                     tf.TensorSpec(shape=[None, None], dtype=tf.float32),   # constellation_pi_proba
                     tf.TensorSpec(shape=[None], dtype=tf.float32),         # prior_prob_x
                     tf.TensorSpec(shape=[3], dtype=tf.int64)               # shape_inter
                     ]
)
def kl_divergence_gaussians(constellation_mus, constellation_sigmas, new_constellation_mus, new_constellation_sigmas,
                            constellation_pi_proba, prior_prob_x, shape_inter):
    # Third term of the adaptation objective (KL divergence between two multivariate gaussians). For the formula, see:
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # print("Tracing 'kl_divergence_gaussians'...You should see this line only during the first call.")
    # log-ratio of determinants of the covariance matrices
    log_term = 2. * (tf.math.log(new_constellation_sigmas) - tf.math.log(constellation_sigmas))
    log_term = tf.reduce_sum(tf.reshape(log_term, shape_inter), axis=2)
    # trace term
    trace_term = tf.math.square(tf.math.divide(constellation_sigmas, new_constellation_sigmas))
    trace_term = tf.math.reduce_sum(tf.reshape(trace_term, shape_inter), axis=2)
    # quadratic form term
    multi_term = tf.math.square(tf.math.divide(new_constellation_mus - constellation_mus, new_constellation_sigmas))
    multi_term = tf.math.reduce_sum(tf.reshape(multi_term, shape_inter), axis=2)

    # The terms `log_term`, `trace_term`, and `multi_term` each have shape `(n_symb, n_comp)`
    kl_divergence = 0.5 * (log_term + trace_term + multi_term - tf.cast(shape_inter[2], tf.float32))
    temp_sum = tf.reduce_sum(tf.multiply(constellation_pi_proba, kl_divergence), axis=1)
    return tf.reduce_sum(tf.multiply(prior_prob_x, temp_sum))


def minimize_channel_adaptation_bfgs(mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x,
                                     n_comp, n_dim, lambda_value, type_objec='log_likelihood',
                                     num_init=1, verbose=False):
    # Loss function and its gradient for the proposed method
    loss_func_proposed, loss_func_val_and_grad = get_loss_func_proposed(
        mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, n_comp, n_dim,
        lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
    )
    # Alternative is to use a lambda function, but this cannot be graph-compiled using `tf.function`
    # loss_func_val_and_grad = lambda x: tfp.math.value_and_gradient(loss_func_proposed, x)

    # Minimize the loss function starting from different initial points, and choose the best local minimum
    psi_values_best = None
    optim_results_best = None
    loss_best = loss_init_best = sys.float_info.max
    for t in range(num_init):
        if t == 0:
            psi_a_init = tf.reshape(tf.eye(n_dim, batch_shape=[n_comp]), [n_comp * n_dim * n_dim])
            psi_init = tf.concat([psi_a_init,                  # scale parameters for the means
                                  tf.zeros([n_comp * n_dim]),  # shift parameters for the means
                                  tf.ones([n_comp * n_dim]),   # scale parameters for the variances
                                  tf.ones([n_comp]),           # scale parameters for the prior logits
                                  tf.zeros([n_comp])           # shift parameters for the prior logits
                                  ], 0)
        else:
            # Random initialization:
            # Diagonal entries of the scale matrix `A` are randomly selected from an interval `[1 / a_max, a_max]`
            # shift parameters are randomly selected from an interval `[-b_max, b_max]`
            a_max = 4.
            b_max = 2.
            # All entries of the matrix `A` are randomly selected from the interval `[1 / a_max, a_max]`
            # psi_a_init = tf.random.uniform([n_comp * n_dim * n_dim], minval=(1. / a_max), maxval=a_max)
            eye_mat = tf.eye(n_dim)
            psi_a_init = tf.stack([tf.random.uniform([n_dim, n_dim], minval=(1. / a_max), maxval=a_max) * eye_mat
                                   for _ in range(n_comp)])
            psi_init = tf.concat([tf.reshape(psi_a_init, [n_comp * n_dim * n_dim]),
                                  tf.random.uniform([n_comp * n_dim], minval=-b_max, maxval=b_max),
                                  tf.random.uniform([n_comp * n_dim], minval=(1. / a_max), maxval=a_max),
                                  tf.random.uniform([n_comp], minval=(1. / a_max), maxval=a_max),
                                  tf.random.uniform([n_comp], minval=-b_max, maxval=b_max)
                                  ], 0)

        psi_init = tf.cast(psi_init, tf.float32)
        loss_init = loss_func_proposed(psi_init)
        # Minimize the loss function using the BFGS method
        optim_results = tfp.optimizer.bfgs_minimize(
            loss_func_val_and_grad,
            initial_position=psi_init,
            tolerance=1e-8,  # tolerance on the supremum norm of the gradient
            f_relative_tolerance=1e-8,  # tolerance on the relative change in the objective function value
            max_iterations=200
        )
        # Minimum value and minima of the loss function
        loss_final = optim_results.objective_value.numpy()
        psi_values = optim_results.position
        if loss_final < loss_best:
            loss_best = loss_final
            psi_values_best = psi_values
            optim_results_best = optim_results
            loss_init_best = loss_init

    # if not optim_results_best.converged.numpy():
    #     print("WARNING: Optimization did not converge.")
    if loss_init_best <= loss_best:
        print("WARNING: optimization did not decrease the loss function. loss_init = {:.4f}, loss_final = {:.4f}".
              format(loss_init_best, loss_best))

    if verbose:
        # print("Number of objective function evaluations: {:d}".format(optim_results_best.num_objective_evaluations))
        print("Initial value of the loss function: {:.6f}".format(loss_init_best))
        print("Final value of the loss function: {:.6f}".format(loss_best))
        print("\nAffine transformation parameters that minimize the loss function:")
        psi_a, psi_b, psi_c, psi_beta, psi_gamma = split_adaptation_params(psi_values_best, n_comp, n_dim)
        print("Scale means: ", psi_a.numpy())
        print("Shift means: ", psi_b.numpy())
        print("Scale variances: ", psi_c.numpy())
        # print("Scale variances: ", tf.math.square(psi_c).numpy())
        print("Scale logits: ", psi_beta.numpy())
        print("Shift logits: ", psi_gamma.numpy())

    return psi_values_best, loss_best


def minimize_channel_adaptation_bfgs_parallel(mdn_model_weights, target_domain_x, target_domain_y, unique_x,
                                              prior_prob_x, n_comp, n_dim, n_hidden, type_objec, num_init, lambda_value):
    # Copy of the function `minimize_channel_adaptation_bfgs` with some changes to make it suitable to be run in
    # parallel using `multiprocessing`.
    #  `mdn_model` is not passed as an input argument to this function because it cannot be serialized by Pickle.
    #  Instead only the list of layer weights is passed to recreate the MDN.
    mdn_model = helpers.MDN_classes.initialize_MDN_model(n_comp, n_dim, n_hidden)
    mdn_model.set_weights(mdn_model_weights)
    mdn_model.trainable = False
    # Loss function and its gradient for the proposed method
    loss_func_proposed, loss_func_val_and_grad = get_loss_func_proposed(
        mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, n_comp, n_dim,
        lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
    )
    # Using partial instead of the lambda function because lambda functions cannot be serialized using Pickle
    # loss_func_val_and_grad = partial(tfp.math.value_and_gradient, loss_func_proposed)

    # Minimize the loss function starting from different initial points, and choose the best local minimum
    psi_values_best = None
    optim_results_best = None
    loss_best = loss_init_best = sys.float_info.max
    for t in range(num_init):
        if t == 0:
            psi_a_init = tf.reshape(tf.eye(n_dim, batch_shape=[n_comp]), [n_comp * n_dim * n_dim])
            psi_init = tf.concat([psi_a_init,                  # scale parameters for the means
                                  tf.zeros([n_comp * n_dim]),  # shift parameters for the means
                                  tf.ones([n_comp * n_dim]),   # scale parameters for the variances
                                  tf.ones([n_comp]),           # scale parameters for the prior logits
                                  tf.zeros([n_comp])           # shift parameters for the prior logits
                                  ], 0)
        else:
            # Random initialization:
            # Diagonal entries of the scale matrix `A` are randomly selected from an interval `[1 / a_max, a_max]`
            # shift parameters are randomly selected from an interval `[-b_max, b_max]`
            a_max = 4.
            b_max = 2.
            # All entries of the matrix `A` are randomly selected from the interval `[1 / a_max, a_max]`
            # psi_a_init = tf.random.uniform([n_comp * n_dim * n_dim], minval=(1. / a_max), maxval=a_max)
            eye_mat = tf.eye(n_dim)
            psi_a_init = tf.stack([tf.random.uniform([n_dim, n_dim], minval=(1. / a_max), maxval=a_max) * eye_mat
                                   for _ in range(n_comp)])
            psi_init = tf.concat([tf.reshape(psi_a_init, [n_comp * n_dim * n_dim]),
                                  tf.random.uniform([n_comp * n_dim], minval=-b_max, maxval=b_max),
                                  tf.random.uniform([n_comp * n_dim], minval=(1. / a_max), maxval=a_max),
                                  tf.random.uniform([n_comp], minval=(1. / a_max), maxval=a_max),
                                  tf.random.uniform([n_comp], minval=-b_max, maxval=b_max)
                                  ], 0)

        psi_init = tf.cast(psi_init, tf.float32)
        loss_init = loss_func_proposed(psi_init)
        # Minimize the loss function using the BFGS method
        optim_results = tfp.optimizer.bfgs_minimize(
            loss_func_val_and_grad,
            initial_position=psi_init,
            tolerance=1e-8,  # tolerance on the supremum norm of the gradient
            f_relative_tolerance=1e-8,  # tolerance on the relative change in the objective function value
            max_iterations=200
        )
        # Minimum value and minima of the loss function
        loss_final = optim_results.objective_value.numpy()
        psi_values = optim_results.position
        if loss_final < loss_best:
            loss_best = loss_final
            psi_values_best = psi_values
            optim_results_best = optim_results
            loss_init_best = loss_init

    if loss_init_best <= loss_best:
        print("WARNING: optimization did not decrease the loss function. loss_init = {:.4f}, loss_final = {:.4f}".
              format(loss_init_best, loss_best))

    return psi_values_best, loss_best


def minimize_channel_adaptation_adam(mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x,
                                     n_comp, n_dim, lambda_value, type_objec='log_likelihood',
                                     n_epochs=200, verbose=False):
    # Loss function and its gradient for the proposed method
    loss_func_proposed, _ = get_loss_func_proposed(
        mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, n_comp, n_dim,
        lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
    )
    # loss_func_val_and_grad = lambda x: tfp.math.value_and_gradient(loss_func_proposed, x)
    # Initial value of `psi`
    psi_a_init = tf.reshape(tf.eye(n_dim, batch_shape=[n_comp]), [n_comp * n_dim * n_dim])
    psi_init = tf.concat([psi_a_init,                  # scale parameters for the means
                          tf.zeros([n_comp * n_dim]),  # shift parameters for the means
                          tf.ones([n_comp * n_dim]),   # scale parameters for the variances
                          tf.ones([n_comp]),           # scale parameters for the prior logits
                          tf.zeros([n_comp])           # shift parameters for the prior logits
                          ], 0)
    psi_init = tf.cast(psi_init, tf.float32)
    loss_init = loss_func_proposed(psi_init)

    # Minimize the loss function using the Adam method
    opt = Optimizer.Adam(
        learning_rate=LEARNING_RATE_ADAM, epsilon=EPSILON_ADAM,
    )
    psi_values = tf.Variable(psi_init)
    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    n_target = target_domain_x.shape[0]
    batch_size = max(10, int(np.ceil(0.01 * n_target)))
    n_epochs = tf.constant(n_epochs)
    n_batches = tf.constant(n_target // batch_size)
    ind_samp = tf.range(n_target)

    @tf.function
    def adam_wrapper():
        # print("Tracing 'adam_wrapper'...You should see this line only during the first call.")
        for i in tf.range(n_epochs):
            # print("Epoch {:d}/{:d}".format(i + 1, n_epochs))
            ind_curr = tf.random.shuffle(ind_samp)  # shuffle the order of samples
            for b in tf.range(n_batches):
                if b < (n_batches - 1):
                    ind_batch = ind_curr[(b * batch_size): ((b + 1) * batch_size)]
                else:
                    ind_batch = ind_curr[(b * batch_size):]

                # Loss function and its gradient on the current batch
                loss_func_val_and_grad_batch = get_loss_func_proposed(
                    mdn_model,
                    tf.gather(target_domain_x, ind_batch, axis=0, batch_dims=1),
                    tf.gather(target_domain_y, ind_batch, axis=0, batch_dims=1),
                    unique_x, prior_prob_x, n_comp, n_dim,
                    lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
                )[1]
                grad = loss_func_val_and_grad_batch(psi_values)[1]
                opt.apply_gradients(zip([grad], [psi_values]))

    adam_wrapper()
    # Minimum value and minima of the loss function
    loss_final = loss_func_proposed(psi_values)
    if loss_final < loss_init:
        loss_best = loss_final
        psi_values_best = psi_values
    else:
        loss_best = loss_init
        psi_values_best = psi_init
        print("WARNING: optimization did not decrease the loss function. loss_init = {:.4f}, loss_final = {:.4f}".
              format(loss_init, loss_final))

    if verbose:
        print("Initial value of the loss function: {:.6f}".format(loss_init))
        print("Final value of the loss function: {:.6f}".format(loss_best))
        print("\nAffine transformation parameters that minimize the loss function:")
        psi_a, psi_b, psi_c, psi_beta, psi_gamma = split_adaptation_params(psi_values_best, n_comp, n_dim)
        print("Scale means: ", psi_a.numpy())
        print("Shift means: ", psi_b.numpy())
        print("Scale variances: ", psi_c.numpy())
        # print("Scale variances: ", tf.math.square(psi_c).numpy())
        print("Scale logits: ", psi_beta.numpy())
        print("Shift logits: ", psi_gamma.numpy())

    return psi_values_best, loss_best


def minimize_channel_adaptation_adam_parallel(mdn_model_weights, target_domain_x, target_domain_y, unique_x,
                                              prior_prob_x, n_comp, n_dim, n_hidden, type_objec, n_epochs, lambda_value):
    # Copy of the function `minimize_channel_adaptation_adam` with some changes to make it suitable to be run in
    # parallel using `multiprocessing`.
    #  `mdn_model` is not passed as an input argument to this function because it cannot be serialized by Pickle.
    #  Instead only the list of layer weights is passed to recreate the MDN.
    mdn_model = helpers.MDN_classes.initialize_MDN_model(n_comp, n_dim, n_hidden)
    mdn_model.set_weights(mdn_model_weights)
    mdn_model.trainable = False

    # Loss function and its gradient for the proposed method
    loss_func_proposed, _ = get_loss_func_proposed(
        mdn_model, target_domain_x, target_domain_y, unique_x, prior_prob_x, n_comp, n_dim,
        lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
    )
    # loss_func_val_and_grad = lambda x: tfp.math.value_and_gradient(loss_func_proposed, x)
    # Initial value of `psi`
    psi_a_init = tf.reshape(tf.eye(n_dim, batch_shape=[n_comp]), [n_comp * n_dim * n_dim])
    psi_init = tf.concat([psi_a_init,                  # scale parameters for the means
                          tf.zeros([n_comp * n_dim]),  # shift parameters for the means
                          tf.ones([n_comp * n_dim]),   # scale parameters for the variances
                          tf.ones([n_comp]),           # scale parameters for the prior logits
                          tf.zeros([n_comp])           # shift parameters for the prior logits
                          ], 0)
    psi_init = tf.cast(psi_init, tf.float32)
    loss_init = loss_func_proposed(psi_init)

    # Minimize the loss function using the Adam method
    opt = Optimizer.Adam(
        learning_rate=LEARNING_RATE_ADAM, epsilon=EPSILON_ADAM,
    )
    psi_values = tf.Variable(psi_init)
    # Batch size is taken as 1% of the sample size or 10, whichever is larger
    n_target = target_domain_x.shape[0]
    batch_size = max(10, int(np.ceil(0.01 * n_target)))
    n_epochs = tf.constant(n_epochs)
    n_batches = tf.constant(n_target // batch_size)
    ind_samp = tf.range(n_target)

    @tf.function
    def adam_wrapper():
        # print("Tracing 'adam_wrapper'...You should see this line only during the first call.")
        for i in tf.range(n_epochs):
            # print("Epoch {:d}/{:d}".format(i + 1, n_epochs))
            ind_curr = tf.random.shuffle(ind_samp)      # shuffle the order of samples
            for b in tf.range(n_batches):
                if b < (n_batches - 1):
                    ind_batch = ind_curr[(b * batch_size): ((b + 1) * batch_size)]
                else:
                    ind_batch = ind_curr[(b * batch_size):]

                # Loss function and its gradient on the current batch
                loss_func_val_and_grad_batch = get_loss_func_proposed(
                    mdn_model,
                    tf.gather(target_domain_x, ind_batch, axis=0, batch_dims=1),
                    tf.gather(target_domain_y, ind_batch, axis=0, batch_dims=1),
                    unique_x, prior_prob_x, n_comp, n_dim,
                    lambda_value=tf.constant(lambda_value, dtype=tf.float32), type_objec=type_objec
                )[1]
                grad = loss_func_val_and_grad_batch(psi_values)[1]
                opt.apply_gradients(zip([grad], [psi_values]))

    adam_wrapper()
    # Minimum value and minima of the loss function
    loss_final = loss_func_proposed(psi_values)
    if loss_final < loss_init:
        loss_best = loss_final
        psi_values_best = psi_values
    else:
        loss_best = loss_init
        psi_values_best = psi_init

    return psi_values_best, loss_best
