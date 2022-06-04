# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import sys
sys.path.append('../../')
from third_party.Grad_TTS.model.utils import fix_len_compatibility


# data parameters
train_filelist_path = 'C:/Users/zador/Desktop/WitchDoc'
valid_filelist_path = 'C:/Users/zador/Desktop/WitchDoc'
# test_filelist_path = '/media/alexey/HDDDataDIsk/datasets/common_voice/unprocessed_librispeech_360/test-clean-360/'
cmudict_path = 'resources/cmu_dictionary'
n_feats = 80
add_blank = True

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4
embedding_size = 256

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = 'logs/new_exp'
pretrain = 'checkpts/grad_424.pt'
test_size = 4
n_epochs = 200
batch_size = 4
learning_rate = 1e-4
seed = 37
save_every = 1
out_size = fix_len_compatibility(2*22050//256)
