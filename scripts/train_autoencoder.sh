#!/bin/bash
# base_dir='/Users/jr/Documents/research/code/domain_adaptation/results_sim'
base_dir='/nobackup/jr/domain_adapt/repo_jr/domain_adaptation/results_sim'
# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# SNR of the channel training data
snr=14
# Number of epochs for training the autoencoder
n_epochs=20
# Regularization strength (or coefficient) multiplying the regularization term for the encoder layer activations in
# the loss function. This can help the autoencoder learn a more balanced or symmetric constellation.
l2_reg_strength='0.0'
# Set to 'true' in order to continue training the autoencoder from the saved model files. Typically set to 'false'.
continue_training='false'

# type_channel in 'fading_ricean' 'fading' 'AWGN'
# type_autoenc in 'standard' 'symbol_estimation_mmse' 'symbol_estimation_map' 'adapt_generative'
# Set only the required combinations in the for loops below.
for type_channel in 'fading_ricean' 'fading' 'AWGN'; do
    for type_autoenc in 'standard' 'symbol_estimation_map'; do

        model_dir="${base_dir}/${type_channel}_${snr}dB/models/${type_autoenc}"
        output_dir="${base_dir}/${type_channel}_${snr}dB/outputs/${type_autoenc}"
        python train_autoencoder.py --n-bits $n_bits --sim --tc $type_channel --SNR-channel $snr --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs --reg $l2_reg_strength

        if [[ $continue_training = 'true' ]]; then
            # OPTIONAL: continue training from the saved autoencoder, channel model, and constellation files
            cmf="${model_dir}/channel_model/channel"
            if [[ $type_autoenc = 'standard' ]]; then
                amf="${model_dir}/autoencoder/autoencoder"
            else
                amf="${model_dir}/autoencoder_${type_autoenc}/autoencoder"
            fi
            cof="${model_dir}/constellation_autoencoder.npy"
            for t in 1 2 3 4; do
                echo -e "\nResuming training from the saved model files, t = $t"
                output_dir="${base_dir}/${type_channel}_${snr}dB/outputs/${type_autoenc}_$t"
                python train_autoencoder.py --n-bits $n_bits --sim --tc $type_channel --SNR-channel $snr --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs --reg $l2_reg_strength --cmf $cmf --amf $amf --cof $cof
            done
         fi

    done
done
