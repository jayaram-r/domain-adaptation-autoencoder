#!/bin/bash
base_dir='/Users/jr/Documents/research/code/domain_adaptation/expts_icml22/autoencoder'
# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# SNR of the source channel training data
snr=14
# Iterative training of autoencoder and channel. Set to 'true' or 'false' all lowercase
iterative_training='true'
# Set to 'true' to skip training the autoencoder (models are already trained)
skip_training='true'
# Number of epochs for training the autoencoder
n_epochs=20
# Number of adaptation samples per symbol from the target channel distribution
# n_adapt=10
# Main (data-dependent) term in the objective function for adaptation: 'log_posterior' or 'log_likelihood'
adapt_objective='log_posterior'
# Number of trials to repeat the adaptation runs. Averaged results are reported
n_trials=10
# Regularization strength (or coefficient) multiplying the regularization term for the encoder layer activations in
# the loss function. This can help the autoencoder learn a more balanced or symmetric constellation.
# Set this to the same value that was used in the training script.
l2_reg_strength='0.0'

# Number of CPU cores to use for adaptation (upper limit)
n_jobs=8

# Range of SNR values for the target channel
snr_min=14
snr_max=24

# type_autoenc in 'standard' 'symbol_estimation_mmse' 'symbol_estimation_map' 'adapt_generative'
# type_channel_src in 'fading_ricean' 'fading' 'AWGN'
# Set only the required combinations in the for loops below.
for type_autoenc in 'standard' 'symbol_estimation_map'; do
    for type_channel_src in 'AWGN'; do

        # Train the autoencoder on the source channel distribution
        if [[ $iterative_training = 'true' ]]; then
            model_dir="${base_dir}/${type_channel_src}_${snr}dB/models/${type_autoenc}_iter"
            output_dir="${base_dir}/${type_channel_src}_${snr}dB/outputs/${type_autoenc}_iter"
            if [[ $skip_training = 'false' ]]; then
                python train_autoencoder_channel_iterative.py --n-bits $n_bits --tc $type_channel_src --SNR-channel $snr --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs --reg $l2_reg_strength
            fi
        else
            model_dir="${base_dir}/${type_channel_src}_${snr}dB/models/${type_autoenc}"
            output_dir="${base_dir}/${type_channel_src}_${snr}dB/outputs/${type_autoenc}"
            if [[ $skip_training = 'false' ]]; then
                python train_autoencoder.py --n-bits $n_bits --sim --tc $type_channel_src --SNR-channel $snr --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs --reg $l2_reg_strength
            fi
        fi

        # Adapt to the channel and autoencoder models to the target channel distribution
        # type_channel_tar in 'fading_ricean' 'fading' 'AWGN'
        for type_channel_tar in 'fading'; do
            if [[ $type_channel_tar != $type_channel_src ]]; then

                # Saved channel model, autoencoder, and the constellation files
                cmf="${model_dir}/channel_model/channel"
                if [[ $type_autoenc = 'standard' ]]; then
                    amf="${model_dir}/autoencoder/autoencoder"
                else
                    amf="${model_dir}/autoencoder_${type_autoenc}/autoencoder"
                fi
                cof="${model_dir}/constellation_autoencoder.npy"
                for n_adapt in 5 10 20 30 40 50; do
                    # Output directory for adaptation
                    if [[ $iterative_training = 'true' ]]; then
                        output_dir="${base_dir}/${type_channel_src}_${snr}dB/adapt_${type_channel_tar}_${n_adapt}/${type_autoenc}_iter"
                    else
                        output_dir="${base_dir}/${type_channel_src}_${snr}dB/adapt_${type_channel_tar}_${n_adapt}/${type_autoenc}"
                    fi
                    python adapt_autoencoder_sim.py --n-bits $n_bits --cmf $cmf --amf $amf --ta $type_autoenc --tc $type_channel_tar --snr-min $snr_min --snr-max $snr_max --ao $adapt_objective -o $output_dir --nad $n_adapt --ne $n_epochs --reg $l2_reg_strength --n-jobs $n_jobs --n-trials $n_trials
                done
            fi
        done
    done
done
