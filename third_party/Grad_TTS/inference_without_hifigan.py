import argparse
import json
import os
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys

sys.path.append('./hifi_gan/')
from env import AttrDict
from models import Generator as HiFiGAN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=False, default='/home/user/ardvk/ardvrk_voice_cloning/temp_dir/p230/p230_018.txt', help='path to a file with texts to synthesize')
    parser.add_argument('-o', '--output', type=str, required=False, default='./out/', help='path to a result folder')
    parser.add_argument('-c', '--checkpoint', type=str, required=False, default='/home/user/ardvk/ardvrk_voice_cloning/third_party/Grad-TTS/checkpts/model_ckpt/grad_75.pt', help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-e', '--embedding', type=str, required=False, default='/home/user/ardvk/ardvrk_voice_cloning/temp_dir/p230/p230_018.npy', help='path to a file with speaker embedding')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    args = parser.parse_args()

    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.embedding_size, params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc))
    generator = generator.to('cuda')
    generator.eval()
    print(f'Number of parameters: {generator.nparams}')

    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    speaker_name = os.path.splitext(os.path.basename(args.embedding))[0]
    emb = np.load(args.embedding).astype(np.float32)
    emb = torch.FloatTensor(emb).unsqueeze(0).cuda()
    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
                                                   stoc=False, length_scale=0.91, emb=emb)
            name, ext = os.path.splitext(os.path.basename(args.embedding))
            print(name)
            np.save(f'{args.output}/{name}.npy', y_dec.detach().cpu().numpy())

    print('Done. Check out `out` folder for samples.')
