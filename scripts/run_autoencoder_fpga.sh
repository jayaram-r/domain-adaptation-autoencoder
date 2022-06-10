#!/bin/bash
# Run autoencoder training and adaptation experiments with FPGA data
base_dir='/Users/jr/Documents/research/code/domain_adaptation/expts_fpga'

# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4

# Number of CPU cores to use for adaptation
n_jobs=8

# Number of adaptation samples per symbol
n_adapt_str='5,10,20,30,40,50'

# Number of trials to repeat the adaptation and fine-tuning runs. Averaged results are reported
n_trials=20
seed=123

# Number of test samples to evaluate the decoding performance
n_test=300000

# Number of epochs for training the autoencoder
n_epochs=20

# Set to 'standard'. Does not support 'symbol_estimation_map' autoencoder
type_autoenc='standard'

# All the BER mat files are saved here
results_dir="${base_dir}/metrics"
if [[ ! -d "$results_dir" ]]; then
    mkdir -p "$results_dir"
fi

# Data from the source distribution used to train the MDN channel and the autoencoder. This uses M-QAM constellation
data_source_train="${base_dir}/data_source_train"
if [[ ! -d "$data_source_train" ]]; then
    mkdir -p "$data_source_train"
fi

############# Training data from the source distribution should be collected and saved to the directory "$data_source_train". This uses M-QAM constellation


# Train the MDN channel and autoencoder on the source domain training dataset
model_dir="${base_dir}/models_source"
output_dir="${base_dir}/outputs_source"
python train_autoencoder.py --n-bits $n_bits --tdf "${data_source_train}/tx_symbols.mat" --rdf "${data_source_train}/rx_symbols.mat" --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs

# Saved channel model file and autoencoder model file from the source domain
cmf="${model_dir}/channel_model/channel"
if [[ $type_autoenc = 'standard' ]]; then
    amf="${model_dir}/autoencoder/autoencoder"
else
    amf="${model_dir}/autoencoder_${type_autoenc}/autoencoder"
fi

# Constellation file of the autoencoder: to be used to generate the target domain data.
# Use `np.load(filename)` to load the constellation
cof="${model_dir}/constellation_autoencoder.npy"

# Test data from the source domain to evaluate the symbol error rate (SER) of the autoencoder. This should use the autoencoder's learned constellation.
data_source_test="${base_dir}/data_source_test"
if [[ ! -d "$data_source_test" ]]; then
    mkdir -p "$data_source_test"
fi

############# Read the autoencoder's constellation file, and use it to collect test data from the source distribution (i.e. no distribution change).
# This data should be saved to the directory "$data_source_test"


# Evaluate the autoencoder's performance (without adaptation) on test data from the source distribution
python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_source_test --cmf $cmf --amf $amf --ta $type_autoenc -o $output_dir --shuffle
# Rename the BER file and copy it to the results directory
mv "${output_dir}/ber.mat" "${output_dir}/ber_source.mat"
cp "${output_dir}/ber_source.mat" "${results_dir}/ber_source.mat"

# Data from the target distribution that is based on the autoencoder's learned constellation
data_target="${base_dir}/data_target"
if [[ ! -d "$data_target" ]]; then
    mkdir -p "$data_target"
fi

############# Read the autoencoder's constellation file, and use it to collect data from the target distribution (i.e. with distribution change such as IQ imbalance).
# This data should be saved to the directory "$data_target"


# Data from the target distribution will be split up equally (50/50) for adaptation and testing.
# The first data split is used for adaptation or finetuning or retraining (depending on the method) of the MDN channel and autoencoder.
# The second data split will be used solely for testing/evaluation of performance on the target distribution.
# Different random subsets of the required target size (e.g 10 samples per symbol) are sub-sampled from the first data split, and used by the adaptation and fine-tuning methods.
data_target_train="${data_target}_train"
data_target_test="${data_target}_test"
python prepare_adaptation_data.py -d $data_target -b $base_dir --n-trials $n_trials --nad "$n_adapt_str"

# Evaluate the autoencoder's performance on test data from the target distribution
python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_target_test --cmf $cmf --amf $amf --ta $type_autoenc -o $output_dir --shuffle
# Rename the BER file and copy it to the results directory
mv "${output_dir}/ber.mat" "${output_dir}/ber_target.mat"
cp "${output_dir}/ber_target.mat" "${results_dir}/ber_target.mat"

# Train the autoencoder on the entire target domain training dataset (used as benchmark for adaptation performance).
model_dir_alt="${base_dir}/models_target"
output_dir_alt="${base_dir}/outputs_target"
# Try out both type of initializations below and use the better one:
# Random initialization of the MDN and autoencoder
python train_autoencoder.py --n-bits $n_bits --tdf "${data_target_train}/tx_symbols.mat" --rdf "${data_target_train}/rx_symbols.mat" --ta $type_autoenc -m $model_dir_alt -o $output_dir_alt --ne $n_epochs

# Initialization using the parameters of the source domain MDN and autoencoder
# python train_autoencoder.py --n-bits $n_bits --tdf "${data_target_train}/tx_symbols.mat" --rdf "${data_target_train}/rx_symbols.mat" --ta $type_autoenc -m $model_dir_alt -o $output_dir_alt --ne $n_epochs --cmf $cmf --amf $amf

# Model files of the retrained autoencoder
cmf_retr="${model_dir_alt}/channel_model/channel"
if [[ $type_autoenc = 'standard' ]]; then
    amf_retr="${model_dir_alt}/autoencoder/autoencoder"
else
    amf_retr="${model_dir_alt}/autoencoder_${type_autoenc}/autoencoder"
fi

# Test data for evaluating the retrained autoencoder is saved here
data_test_retr="${base_dir}/data_test_retrained"
if [[ ! -d "$data_test_retr" ]]; then
    mkdir -p "$data_test_retr"
fi
# Constellation file of the retrained autoencoder
cof_retr="${model_dir_alt}/constellation_autoencoder.npy"

############# Read the retrained autoencoder's constellation file, and use it to collect data from the target distribution (i.e. with distribution change such as IQ imbalance).
# This is used only for evaluating the retrained autoencoder, and should be saved to the directory "$data_test_retr".


# Evaluate the performance of the retrained autoencoder on test data from the target distribution
python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_test_retr --cmf $cmf_retr --amf $amf_retr --ta $type_autoenc -o $output_dir_alt --shuffle
# Rename the BER file and copy it to the results directory
mv "${output_dir_alt}/ber.mat" "${output_dir_alt}/ber_retrained_target.mat"
cp "${output_dir_alt}/ber_retrained_target.mat" "${results_dir}/ber_retrained_target.mat"


# Repeat the adaptation and fine-tuning for a number of trials
for t in {1..20}; do
    echo -e "\nRunning trial ${t} of adaptation/fine-tuning"

    # Sub-directories for this trial
    curr_dir="${base_dir}/trial$t"
    if [[ ! -d "$curr_dir" ]]; then
        mkdir -p "$curr_dir"
    fi
    data_adapt="${curr_dir}/data_adapt"

    results_dir_curr="${results_dir}/trial$t"
    if [[ ! -d "$results_dir_curr" ]]; then
        mkdir -p "$results_dir_curr"
    fi

    # Adaptation for different number of target samples
    for n_adapt in 5 10 20 30 40 50; do
        echo -e "\nRunning adaptation and fine-tuning for $n_adapt samples per symbol"
        data_dir_curr="${data_adapt}_$n_adapt"

        ################## Proposed adaptation method
        output_dir_curr="${curr_dir}/adapt_${n_adapt}"
        python adapt_autoencoder_measure.py --n-bits $n_bits --cmf $cmf --amf $amf --ta $type_autoenc -m $output_dir_curr -d $data_dir_curr --n-jobs $n_jobs

        # Evaluate performance of the proposed adaptation method on the test data
        apf="${output_dir_curr}/adaptation_params.npy"
        python adapt_autoencoder_decode.py --n-bits $n_bits -d $data_target_test --cmf $cmf --amf $amf --apf $apf --ta $type_autoenc -o $output_dir_curr --shuffle
        # Copy the BER file to the results directory
        cp "${output_dir_curr}/ber_adapted.mat" "${results_dir_curr}/ber_adapted_${n_adapt}.mat"


        ################## Finetuning baseline method (entire MDN is finetuned)
        model_dir_curr="${curr_dir}/finetune_${n_adapt}"
        python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_curr -d $data_dir_curr

        # Evaluate performance of the finetuned autoencoder on the test data
        cmf_finetuned="${model_dir_curr}/channel_model/channel"
        amf_finetuned="${model_dir_curr}/autoencoder/autoencoder"
        python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_target_test --cmf $cmf_finetuned --amf $amf_finetuned -o $model_dir_curr --shuffle
        # Copy the BER file to the results directory
        cp "${model_dir_curr}/ber_finetuned.mat" "${results_dir_curr}/ber_finetuned_${n_adapt}.mat"


        ################## Finetuning baseline method (last layer of the MDN is finetuned)
        model_dir_curr="${curr_dir}/finetune_last_${n_adapt}"
        python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_curr -d $data_dir_curr --last

        # Evaluate performance of the finetuned autoencoder on the test data
        cmf_finetuned="${model_dir_curr}/channel_model/channel"
        amf_finetuned="${model_dir_curr}/autoencoder/autoencoder"
        python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_target_test --cmf $cmf_finetuned --amf $amf_finetuned -o $model_dir_curr --shuffle
        # Copy the BER file to the results directory
        cp "${model_dir_curr}/ber_finetuned.mat" "${results_dir_curr}/ber_finetuned_last_${n_adapt}.mat"
    done

done
