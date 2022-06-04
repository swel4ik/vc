# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import os
import datetime as dt
from timeit import default_timer
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse, save_plot
#
# from third_party.Grad_TTS.hifi_gan_legacy.env import AttrDict
# from third_party.Grad_TTS.hifi_gan_legacy.models import Generator as HiFiGAN

from third_party.Grad_TTS.hifi_gan.env import AttrDict
from third_party.Grad_TTS.hifi_gan.models import Generator as HiFiGAN
HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

# HIFIGAN_CONFIG = 'third_party/Grad_TTS/checkpts/config.json'
# HIFIGAN_CHECKPT = 'third_party/Grad_TTS/checkpts/g_02175000'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-o', '--output', type=str, required=False, default='./out/', help='path to a result folder')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-e', '--embedding', type=str, required=True, help='path to a file with speaker embedding')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10, help='number of timesteps of reverse diffusion')

    parser.add_argument('-d', '--device', type=str, required=False, choices=['cuda', 'cpu'], default='cuda', help='Device for inference')

    parser.add_argument('-i', '--hifi_gan', type=str, required=False, default='./checkpts/g_02175000',
                        help='path to a checkpoint of HiFi-GAN')
    parser.add_argument('-g', '--hifi_config', type=str, required=False, default='./checkpts/config.json')
    parser.add_argument(
        '-r', '--resources', type=str, default='./resources/cmu_dictionary',
        required=False
    )

    args = parser.parse_args()

    HIFIGAN_CONFIG = args.hifi_config if args.hifi_config != 'None' else HIFIGAN_CONFIG
    HIFIGAN_CHECKPT = args.hifi_gan if args.hifi_gan != 'None' else HIFIGAN_CHECKPT
    
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.embedding_size, params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    generator = generator.to(args.device)
    generator.eval()
    print(f'Number of parameters: {generator.nparams}')
    
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))

    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    vocoder = vocoder.to(args.device)
    vocoder.eval()
    vocoder.remove_weight_norm()

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict(args.resources)

    speaker_name = os.path.splitext(os.path.basename(args.embedding))[0]
    emb = np.load(args.embedding).astype(np.float32)
    emb = torch.FloatTensor(emb).unsqueeze(0).to(args.device)
    time_series = []
        
    with torch.no_grad():
        for i, text in enumerate(texts):
            start_inference_time = default_timer()

            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).to(args.device)[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).to(args.device)
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, length_scale=0.91, emb=emb)
            t = (dt.datetime.now() - t).total_seconds()
            save_plot(y_dec.squeeze().cpu(), f'C:/Users/zador/Desktop/plotted/1.png')
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            if 'embedding_size' in h:
                audio = (vocoder.forward(y_dec, emb=emb).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            else:
                audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            # audio = (vocoder.forward(y_dec).to('cpu').squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'{args.output}/{speaker_name}_{args.timesteps}_sample_{i}.wav', 22050, audio)

            finish_inference_time = default_timer()

            time_series.append(finish_inference_time - start_inference_time)

    print('Done. Check out `out` folder for samples.')
    print(
        'Average inference time: {:.2f} sec'.format(
            np.array(time_series).mean()
        )
    )
