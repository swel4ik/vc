from third_party.wavegrad2.lightning_model import Wavegrad2
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
from timeit import default_timer
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import matplotlib.pyplot as plt

from third_party.wavegrad2.text import Language
from third_party.wavegrad2.utils.stft import STFTMag
import numpy as np
from g2p_en import G2p
from pypinyin import pinyin, Style
import re

# from dataloader import TextAudioDataset


def save_stft_mag(wav, fname):
    fig = plt.figure(figsize=(9, 3))
    plt.imshow(rosa.amplitude_to_db(stft(wav[0].detach().cpu()).numpy(),
               ref=np.max, top_db = 80.),
               aspect='auto',
               origin='lower',
               interpolation='none')
    plt.colorbar()
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()
    fig.savefig(fname, format='png')
    plt.close()
    return


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

# def preprocess_eng(hparams, text):
#     lexicon = read_lexicon(hparams.data.lexicon_path)
#
#     g2p = G2p()
#     phones = []
#     words = re.split(r"([,;.\-\?\!\s+])", text)
#     for w in words:
#         if w.lower() in lexicon:
#             phones += lexicon[w.lower()]
#         else:
#             phones += list(filter(lambda p: p != " ", g2p(w)))
#     phones = "{" + "}{".join(phones) + "}"
#     phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
#     print('g2p: ', phones)
#
#     trainset = TextAudioDataset(hparams, hparams.data.train_dir, hparams.data.train_meta, train=False)
#
#     text = trainset.get_text(phones)
#     text = text.unsqueeze(0)
#     return text


def preprocess_eng(hparams, text, lexicon_path=None):
    if not lexicon_path:
        lexicon = read_lexicon(hparams.data.lexicon_path)
    else:
        lexicon = read_lexicon(lexicon_path)
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    lang = Language(hparams.data.lang, hparams.data.text_cleaners)
    text_norm = torch.LongTensor(lang.text_to_sequence(phones, hparams.data.text_cleaners))
    text_norm = text_norm.unsqueeze(0)
    return text_norm


# def preprocess_mandarin(hparams, text):
#     lexicon = read_lexicon(hparams.data.lexicon_path)
#
#     phones = []
#     pinyins = [
#         p[0]
#         for p in pinyin(
#             text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
#         )
#     ]
#     for p in pinyins:
#         if p in lexicon:
#             phones += lexicon[p]
#         else:
#             phones.append("sp")
#
#     phones = "{" + " ".join(phones) + "}"
#     print('g2p: ', phones)
#
#     trainset = TextAudioDataset(hparams, hparams.data.train_dir, hparams.data.train_meta, train=False)
#
#     text = trainset.get_text(phones)
#     text = text.unsqueeze(0)
#
#     return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument('--text',
                        type=str,
                        default=None,
                        help="raw text to synthesize, for single-sentence mode only")
    parser.add_argument('--speaker_emb',
                        type=str,
                        required=True,
                        help="Path to speacker embedding")
    parser.add_argument('--pace',
                        type=int,
                        default=1.0,
                        help="control the pace of the whole utterance")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument('--result_folder',
                        type=str,
                        required=False,
                        default='./results/',
                        help='Path to folder with generated files')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")

    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    hparams = OC.load('hparameter.yaml')
    os.makedirs(args.result_folder, exist_ok=True)
    if args.steps is not None:
        hparams.ddpm.max_step = args.steps
        if args.steps == 8:
            hparams.ddpm.noise_schedule = \
                "torch.tensor([1e-6,2e-6,1e-5,1e-4,1e-3,1e-2,1e-1,9e-1])"
    else:
        args.steps = hparams.ddpm.max_step
    print(args.steps)
    model = Wavegrad2(hparams).to(args.device)
    stft = STFTMag()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in args.checkpoint) else ckpt)

    start_inference_time = default_timer()
    if hparams.data.lang == 'eng':
        text = preprocess_eng(hparams, args.text)

    spk_emb = np.load(args.speaker_emb).astype(np.float32)

    text = text.to(args.device)
    spk_emb = torch.from_numpy(spk_emb).to(args.device).unsqueeze(0)

    wav_recon, align, *_ = model.inference(text, spk_emb, pace=args.pace)
    finish_inference_time = default_timer()

    text_to_save = str(args.text).replace('?', '.')
    save_name = os.path.splitext(os.path.basename(args.speaker_emb))[0]

    save_stft_mag(wav_recon, os.path.join(args.result_folder, f'{save_name}_{text_to_save}.png'))
    swrite(os.path.join(args.result_folder, f'{save_name}_{text_to_save}.wav'),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())

    print(
        'Average inference time: {:.2f} sec'.format(
            finish_inference_time - start_inference_time
        )
    )
