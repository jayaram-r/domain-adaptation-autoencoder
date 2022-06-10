#!/bin/bash
base_dir='/Users/jr/Documents/research/code/domain_adaptation/expts_icml22/mdn_simulated'
# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# Number of CPU cores to use for adaptation
n_jobs=8
# Type of the source and target channels: 'fading_ricean' 'fading' 'AWGN'
type_source='fading'
type_target='fading_ricean'
# Number of adaptation samples per symbol: 5, 10, 20, 40, 60
# n_adapt=10
# Main (data-dependent) term in the objective function for adaptation: 'log_posterior' or 'log_likelihood'
adapt_objective='log_likelihood'

# Range of SNR values in dB
snr_min=10
snr_max=20
snr_range=$((snr_max - snr_min + 1))

RANDOM=123  # seed the shell RNG for reproducible results
# Repeat the adaptation for a number of trials
for t in {1..50}; do

    echo -e "\nRunning trial $t"
    # Randomly set the seed and SNR of the source and target channels
    # seed=$(($RANDOM % 1000 + 1))
    seed=$((t + 100))
    snr_source=$(($(($RANDOM % $snr_range)) + $snr_min))
    snr_target=$(($(($RANDOM % $snr_range)) + $snr_min))
    echo -e "SNR_source = $snr_source, SNR_target = $snr_target"

    # Train the MDN on the source channel
    output_dir="${base_dir}/outputs_train"
    model_dir="${base_dir}/models_${type_source}_${snr_source}dB"
    check_dir="${model_dir}/channel_model"
    if [[ ! -d $check_dir ]]; then
        # Train the MDN channel if it does not already exist
        python train_mdn_channel.py --n-bits $n_bits --de 2 --sim --tc $type_source --SNR-channel $snr_source -m $model_dir -o $output_dir --skip-summary
    fi

    # Train the MDN on the target channel (benchmark for adaptation performance)
    output_dir_tar="${base_dir}/outputs_train_target"
    model_dir_tar="${base_dir}/models_${type_target}_${snr_target}dB"
    check_dir="${model_dir_tar}/channel_model"
    if [[ ! -d $check_dir ]]; then
        # Train the MDN channel if it does not already exist
        python train_mdn_channel.py --n-bits $n_bits --de 2 --sim --tc $type_target --SNR-channel $snr_target -m $model_dir_tar -o $output_dir_tar --skip-summary
    fi

    # Adapt the MDN
    cmf="${model_dir}/channel_model/channel"
    cof="${model_dir}/constellation_init.npy"
    cmf_tar="${model_dir_tar}/channel_model/channel"
    # Vary the number of adaptation samples per symbol
    for n_adapt in 1 2 4 6 8 10 12 14 16 18 20; do

        results_dir="${base_dir}/${type_source}_to_${type_target}_${n_adapt}"
        if [[ $t -gt 1 ]]; then
            # Append results
            python adapt_mdn_channel.py --n-bits $n_bits --cmf $cmf --tcmf $cmf_tar --cof $cof --ao $adapt_objective --sim --tc $type_target --snr $snr_target --nad $n_adapt -m $results_dir -o $results_dir --n-jobs $n_jobs --append-output --seed $seed
        else
            python adapt_mdn_channel.py --n-bits $n_bits --cmf $cmf --tcmf $cmf_tar --cof $cof --ao $adapt_objective --sim --tc $type_target --snr $snr_target --nad $n_adapt -m $results_dir -o $results_dir --n-jobs $n_jobs --seed $seed
        fi

    done

done
