#!/bin/bash
# Main script for the baseline method of finetuning the autoencoder.
base_dir='/Users/jr/Documents/research/code/domain_adaptation'
# base_dir='/nobackup/jr/domain_adapt/repo_jr/domain_adaptation'
results_dir="$base_dir/results_finetune"

# Specify comma seperated list (no spaces) of GPUs to be visible. Set to empty string to run on CPU only.
export CUDA_VISIBLE_DEVICES=''
# export CUDA_VISIBLE_DEVICES=0

# Number of bits
n_bits=4
# Number of epochs for training and adapting the autoencoder
n_epochs=20

# Regularization strength (or coefficient) multiplying the regularization term for the encoder layer activations in
# the loss function. This can help the autoencoder learn a more balanced or symmetric constellation.
l2_reg_strength='0.0'

# Channel data files
tx_data_file="${base_dir}/data_train/tx_send.mat"
rx_data_file="${base_dir}/data_train/rx_channel.mat"

# Only 'standard' is supported for this method
type_autoenc='standard'

# Train the autoencoder
model_dir="${results_dir}/models"
output_dir="${results_dir}/outputs"
python train_autoencoder.py --n-bits $n_bits --tdf $tx_data_file --rdf $rx_data_file --ta $type_autoenc -m $model_dir -o $output_dir --ne $n_epochs --reg $l2_reg_strength

# Assuming the adaptation and test data files have already been placed in the data directories below
data_dir_adapt="${base_dir}/data_adapt"
data_dir_test="${base_dir}/data_test"

###################### Baseline variation where the entire MDN is adapted
cmf="${model_dir}/channel_model/channel"
amf="${model_dir}/autoencoder/autoencoder"
model_dir_adapt="${results_dir}/models_finetuned"

# Finetune the autoencoder
python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_adapt -d $data_dir_adapt

# Evaluate performance
cmf_adapt="${model_dir_adapt}/channel_model/channel"
amf_adapt="${model_dir_adapt}/autoencoder/autoencoder"
python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_dir_test --cmf $cmf_adapt --amf $amf_adapt -o $model_dir_adapt


###################### Baseline variation where only the last layer of the MDN is adapted
cmf="${model_dir}/channel_model/channel"
amf="${model_dir}/autoencoder/autoencoder"
model_dir_adapt="${results_dir}/models_finetuned_last"

# Finetune the autoencoder
python finetune_autoencoder.py --n-bits $n_bits --cmf $cmf --amf $amf --n-epochs $n_epochs -m $model_dir_adapt -d $data_dir_adapt --last

# Evaluate performance
cmf_adapt="${model_dir_adapt}/channel_model/channel"
amf_adapt="${model_dir_adapt}/autoencoder/autoencoder"
python finetune_autoencoder_decode.py --n-bits $n_bits -d $data_dir_test --cmf $cmf_adapt --amf $amf_adapt -o $model_dir_adapt
