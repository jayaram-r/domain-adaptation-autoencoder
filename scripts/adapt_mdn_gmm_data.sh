#!/bin/bash
base_dir='/Users/jr/Documents/research/code/domain_adaptation/expts_icml22/mdn_gmm_data'
# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# Number of CPU cores to use for adaptation
n_jobs=8
# Number of adaptation samples per symbol
n_adapt_str='1,2,4,6,8,10,12,14,16,18,20'
# Main (data-dependent) term in the objective function for adaptation: 'log_posterior' or 'log_likelihood'
adapt_objective='log_likelihood'

# Number of components in the source and target Gaussian mixtures
n_comp_src=3
n_comp_tar=3
n_comp_mdn=5

# Repeat the adaptation for a number of trials
for t in {1..50}; do
    # Randomly set the seed and SNR of the source and target channels
    # seed=$(($RANDOM % 1000 + 1))
    seed=$(($t + 100))
    echo -e "\nRunning trial ${t}"

    # Sub-directory for this trial
    curr_dir="${base_dir}/trial$t"
    if [[ ! -d "$curr_dir" ]]; then
        mkdir -p "$curr_dir"
    fi
    # Generate the training, adaptation, and test datasets
    data_train="${curr_dir}/data_train"
    data_adapt="${curr_dir}/data_adapt"
    data_test="${curr_dir}/data_test"
    data_retrain_target="${curr_dir}/data_retrain_target"
    python generate_gmm_data.py --nb $n_bits --ns $n_comp_src --nt $n_comp_tar --nad "$n_adapt_str" --dtr "$data_train" --da "$data_adapt" --dte "$data_test" -o "${curr_dir}/summary" --seed $seed

    # Train the MDN channel model on the source domain training dataset
    model_dir="${curr_dir}/mdn_source"
    output_dir="${curr_dir}/mdn_source"
    python train_mdn_channel.py --n-bits $n_bits --de 2 --nc $n_comp_mdn --tdf "${data_train}/tx_symbols.mat" --rdf "${data_train}/rx_symbols.mat" -m "$model_dir" -o "$output_dir"

    # Train the MDN channel model on the target domain re-training dataset (benchmark for adaptation performance)
    model_dir_target="${curr_dir}/mdn_retrain_target"
    output_dir_target="${curr_dir}/mdn_retrain_target"
    python train_mdn_channel.py --n-bits $n_bits --de 2 --nc $n_comp_mdn --tdf "${data_retrain_target}/tx_symbols.mat" --rdf "${data_retrain_target}/rx_symbols.mat" -m "$model_dir_target" -o "$output_dir_target"

    # Adapt the MDN channel model
    cmf="${model_dir}/channel_model/channel"
    cof="${model_dir}/constellation_init.npy"
    cmf_tar="${model_dir_target}/channel_model/channel"
    for n_adapt in 1 2 4 6 8 10 12 14 16 18 20; do
        echo -e "\nRunning adaptation for $n_adapt samples per symbol"
        # Results directory for all the trials
        results_dir="${base_dir}/results_${n_adapt}"
        if [[ ! -d "$results_dir" ]]; then
            mkdir -p "$results_dir"
        fi
        adapted_model_dir="${curr_dir}/mdn_target_${n_adapt}"
        data_adapt_curr="${data_adapt}_${n_adapt}"
        if [[ $t -gt 1 ]]; then
            # Append results
            python adapt_mdn_channel.py --n-bits $n_bits --de 2 --nc $n_comp_mdn --cmf $cmf --tcmf $cmf_tar --cof $cof --ao $adapt_objective --da "$data_adapt_curr" --dt "$data_test" -m "$adapted_model_dir" -o "$results_dir" --n-jobs $n_jobs --append-output --seed $seed
        else
            python adapt_mdn_channel.py --n-bits $n_bits --de 2 --nc $n_comp_mdn --cmf $cmf --tcmf $cmf_tar --cof $cof --ao $adapt_objective --da "$data_adapt_curr" --dt "$data_test" -m "$adapted_model_dir" -o "$results_dir" --n-jobs $n_jobs --seed $seed
        fi
    done

done
