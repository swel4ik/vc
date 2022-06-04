from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator as HiFiGAN
from hifi_inference import load_checkpoint, get_mel, scan_checkpoint
import numpy as np
import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="HifiGAN inference on single file")
    parser.add_argument("--config", required=True, help="path to HifiGAN config")
    parser.add_argument("--checkpoint", required=True, help="path to HifiGAN model weights")
    parser.add_argument('--mel_spectrogram', required=True, help="path to mel_spectrogram file")
    parser.add_argument("--voice_embedding", required=True, help="path to voice embedding file")
    parser.add_argument("--output", required=True, help="path to output file")
    parser.add_argument("--device", required=False, default="cuda")
    return parser.parse_args()


def inference(a):
    with open(a.config) as f:
        h = AttrDict(json.load(f))

    generator = HiFiGAN(h)
    generator.load_state_dict(torch.load(a.checkpoint, map_location=lambda loc, storage: loc)['generator'])
    generator = generator.to(a.device)
    generator.eval()
    generator.remove_weight_norm()

    emb = np.load(a.voice_embedding).astype(np.float32)
    emb = torch.FloatTensor(emb).unsqueeze(0).to(a.device)
    print(emb.shape)

    y_dec = np.load(a.mel_spectrogram).astype(np.float32)
    y_dec = torch.FloatTensor(y_dec).unsqueeze(0).to(a.device)
    print(y_dec.shape)

    audio = (generator.forward(y_dec, emb=emb).cpu().squeeze().clamp(-1, 1).detach().numpy() * 32768).astype(np.int16)
    print(audio.shape)
    output_file = os.path.join(a.output, os.path.splitext(a.mel_spectrogram)[0] + '_generated.wav')
    print(output_file)
    write(output_file, 22050, audio)


if __name__ == '__main__':
    a = parse_args()

    with open(a.config) as f:
        h = AttrDict(json.load(f))
    inference(a)
