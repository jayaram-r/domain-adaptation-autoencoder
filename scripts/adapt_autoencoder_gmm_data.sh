#!/bin/bash
base_dir='/Users/jr/Documents/research/code/domain_adaptation/expts_icml22/autoencoder_gmm_data'
# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# Number of CPU cores to use for adaptation
n_jobs=8
# Number of adaptation samples per symbol
n_adapt_str='5,10,20,30,40,50'
# Number of test samples to evaluate the decoding performance
n_test=300000
# Main (data-dependent) term in the objective function for adaptation: 'log_posterior' or 'log_likelihood'
adapt_objective='log_posterior'

# Number of epochs for training the autoencoder
n_epochs=20
# Regularization strength (or coefficient) multiplying the regularization term for the encoder layer activations
l2_reg_strength='0.0'
# Set to 'standard'. Does not support 'symbol_estimation_map' autoencoder
type_autoenc='standard'

# Number of components in the source and target Gaussian mixtures
n_comp_src=3
n_comp_tar=3
n_comp_mdn=5

# Maximum phase shift in degrees for the target domain distribution. Set to 0 in order to exclude random phase shifts
mps=0

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

    # To set a random number of mixture components in the source and target Gaussian mixtures, uncomment the lines below.
    # Suppose the range of components is {3, 4, 5, 6}:
    # n_comp_src=$(($RANDOM % 4 + 3))
    # n_comp_tar=$(($RANDOM % 4 + 3))

    # Generate the training data from a source domain Gaussian mixture. An M-QAM constellation is used
    data_train="${curr_dir}/data_train"
    data_adapt="${curr_dir}/data_adapt"
    data_test="${curr_dir}/data_test"
    data_retrain_target="${curr_dir}/data_retrain_target"
    python generate_gmm_data.py --nb $n_bits --ns $n_comp_src --nt $n_comp_tar --nad "$n_adapt_str" --dtr "$data_train" --da "$data_adapt" --dte "$data_test" -o "${curr_dir}/summary_src" --seed $seed

    # Train the autoencoder on the source domain training dataset
    model_dir="${curr_dir}/models/${type_autoenc}"
    output_dir="${curr_dir}/outputs/${type_autoenc}"
    python train_autoencoder.py --n-bits $n_bits --tdf "${data_train}/tx_symbols.mat" --rdf "${data_train}/rx_symbols.mat" --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs

    # Saved channel model, autoencoder, and the constellation file from the source domain
    cmf="${model_dir}/channel_model/channel"
    if [[ $type_autoenc = 'standard' ]]; then
        amf="${model_dir}/autoencoder/autoencoder"
    else
        amf="${model_dir}/autoencoder_${type_autoenc}/autoencoder"
    fi
    cof="${model_dir}/constellation_autoencoder.npy"

    # Generate the adaptation and test data from a target domain Gaussian mixture. The autoencoder's constellation is used
    data_temp="${curr_dir}/data_temp"
    python generate_gmm_data.py --nb $n_bits --ns $n_comp_src --nt $n_comp_tar --mps $mps --nad "$n_adapt_str" --dtr "$data_temp" --da "$data_adapt" --dte "$data_test" -o "${curr_dir}/summary_tar" --seed $seed --cof "$cof" --n-test $n_test
    rm -r "$data_temp"

    # Train the autoencoder on the target domain re-training dataset (used as a benchmark for adaptation performance).
    # The channel model and autoencoder from the source domain are used to initialize - this leads to better performance of the retrained autoencoder
    model_dir_alt="${curr_dir}/models_retrained/${type_autoenc}"
    output_dir_alt="${curr_dir}/outputs_retrained/${type_autoenc}"
    params_gmm="${curr_dir}/summary_tar/params_gmm_target.pkl"
    # Iterative training of autoencoder and MDN channel
    python train_autoencoder_iterative_gmm.py --n-bits $n_bits --params-gmm $params_gmm --mps $mps -m $model_dir_alt -o $output_dir_alt --ne $n_epochs --cmf $cmf --amf $amf --cof $cof
    # MDN is trained once followed by training of the autoencoder
    # python train_autoencoder.py --n-bits $n_bits --tdf "${data_retrain_target}/tx_symbols.mat" --rdf "${data_retrain_target}/rx_symbols.mat" --ta $type_autoenc -m $model_dir_alt -o $output_dir_alt --ne $n_epochs --cmf $cmf --amf $amf

    # Model files of the retrained (target domain) autoencoder
    cmf_retr="${model_dir_alt}/channel_model/channel"
    if [[ $type_autoenc = 'standard' ]]; then
        amf_retr="${model_dir_alt}/autoencoder/autoencoder"
    else
        amf_retr="${model_dir_alt}/autoencoder_${type_autoenc}/autoencoder"
    fi
    cof_retr="${model_dir_alt}/constellation_autoencoder.npy"

    # Evaluate the performance of the unadapted (source domain) autoencoder on the test data
    python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_test --cmf $cmf --amf $amf --ta $type_autoenc -o $output_dir

    # Adaptation for different number of target samples
    for n_adapt in 5 10 20 30 40 50; do
        echo -e "\nRunning adaptation for $n_adapt samples per symbol"
        data_dir_curr="${data_adapt}_$n_adapt"

        ################## Proposed adaptation method
        output_dir_curr="${curr_dir}/adapt_${n_adapt}/${type_autoenc}"
        python adapt_autoencoder_measure.py --n-bits $n_bits --cmf $cmf --amf $amf --ta $type_autoenc -m $output_dir_curr -d $data_dir_curr --n-jobs $n_jobs

        # Evaluate performance of the proposed adaptation method on the test data
        apf="${output_dir_curr}/adaptation_params.npy"
        python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_test --cmf $cmf --amf $amf --apf $apf --ta $type_autoenc -o $output_dir_curr


        ################## Finetuning baseline method (entire MDN is finetuned)
        model_dir_curr="${curr_dir}/finetune_${n_adapt}/${type_autoenc}"
        python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_curr -d $data_dir_curr

        # Evaluate performance of the finetuned autoencoder on the test data
        cmf_finetuned="${model_dir_curr}/channel_model/channel"
        amf_finetuned="${model_dir_curr}/autoencoder/autoencoder"
        python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_test --cmf $cmf_finetuned --amf $amf_finetuned -o $model_dir_curr


        ################## Finetuning baseline method (last layer of the MDN is finetuned)
        model_dir_curr="${curr_dir}/finetune_last_${n_adapt}/${type_autoenc}"
        python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_curr -d $data_dir_curr --last

        # Evaluate performance of the finetuned autoencoder on the test data
        cmf_finetuned="${model_dir_curr}/channel_model/channel"
        amf_finetuned="${model_dir_curr}/autoencoder/autoencoder"
        python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_test --cmf $cmf_finetuned --amf $amf_finetuned -o $model_dir_curr
    done

done
