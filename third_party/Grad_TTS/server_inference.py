import json
import os
import datetime as dt
from timeit import default_timer
import numpy as np
from scipy.io.wavfile import write

import torch

from third_party.Grad_TTS import params
from third_party.Grad_TTS.model import GradTTS
from third_party.Grad_TTS.text import text_to_sequence, cmudict
from third_party.Grad_TTS.text.symbols import symbols
from third_party.Grad_TTS.utils import intersperse
from third_party.Grad_TTS.hifi_gan.env import AttrDict
from third_party.Grad_TTS.hifi_gan.models import Generator as HiFiGAN


class GradTTSInference(object):
    def __init__(self,
                 hifi_config: str,
                 hifi_weights: str,
                 grad_tts_weights: str,
                 corpus_dictionary_path: str,
                 device: str = 'cpu',
                 timesteps: int = 1000):
        """
        Class constructor
        Args:
            hifi_config: path to Hi-Fi config file
            hifi_weights: path to Hi-Fi model checkpoint
            grad_tts_weights: path to Grad-TTS model checkpoint
            corpus_dictionary_path: Path to CMU dictionary
            device: inference device (choice from ['cpu', 'cuda'])
            timesteps: Count of timestamps
        """
        self.generator = GradTTS(
            len(symbols) + 1,
            params.embedding_size,
            params.n_enc_channels,
            params.filter_channels,
            params.filter_channels_dp, params.n_heads,
            params.n_enc_layers,
            params.enc_kernel, params.enc_dropout,
            params.window_size,
            params.n_feats, params.dec_dim, params.beta_min,
            params.beta_max, params.pe_scale
        )
        self.generator.load_state_dict(
            torch.load(grad_tts_weights, map_location=lambda loc, storage: loc))
        self.generator = self.generator.to(device)
        self.generator.eval()

        with open(hifi_config) as f:
            h = AttrDict(json.load(f))
        self.vocoder = HiFiGAN(h)
        self.vocoder.load_state_dict(
            torch.load(hifi_weights, map_location=lambda loc, storage: loc)[
                'generator'])
        self.vocoder = self.vocoder.to(device)
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

        self.cmu = cmudict.CMUDict(corpus_dictionary_path)

        self.device = device
        self.timesteps = timesteps

    def __call__(self, texts: list, voice_embedding: np.ndarray) -> list:
        """

        Args:
            texts: List with strings with texts to generate speeches
            voice_embedding: voice embedding with target style

        Returns:
            List of wavefront samples
        """
        emb_batch = torch.stack(
            [torch.FloatTensor(voice_embedding).to(self.device)] * len(texts),
            dim=0
        )

        result_audios = []

        input_tensors = []
        with torch.no_grad():
            # for text in texts:
            #     x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=self.cmu), len(symbols))).to(self.device)[None]
            #     x_lengths = torch.LongTensor([x.shape[-1]]).to(self.device)
            #
            #     input_tensors.append([x, x_lengths])
            #
            # x_max_len = max([sample[0].shape[-1] for sample in input_tensors])
            #
            # x_tensor = torch.zeros(len(texts), x_max_len, dtype=torch.long)
            # for batch_sample_idx in range(len(texts)):
            #     _x = input_tensors[batch_sample_idx][0]
            #     x_tensor[batch_sample_idx, :_x.shape[-1]] = _x
            #
            # x_lengths = torch.cat(
            #     [input_tensor[1] for input_tensor in input_tensors],
            #     dim=0
            # )
            #
            # y_enc, y_dec, attn = self.generator.forward(
            #    x_tensor, x_lengths,
            #    n_timesteps=self.timesteps,
            #    temperature=1.5,
            #    stoc=False, length_scale=0.91,
            #    emb=emb_batch
            # )
            #
            # vocoded = self.vocoder.forward(y_dec).to('cpu')
            #
            # result_audios = [
            #     (vocoded[batch_idx].clamp(-1, 1).numpy().copy() * 32768).astype(np.int16)
            #     for batch_idx in range(len(texts))
            # ]

            for i, text in enumerate(texts):
                x = torch.LongTensor(
                    intersperse(text_to_sequence(text, dictionary=self.cmu),
                                len(symbols))).to(self.device)[None]
                x_lengths = torch.LongTensor([x.shape[-1]]).to(self.device)

                t = dt.datetime.now()
                y_enc, y_dec, attn = self.generator.forward(
                    x, x_lengths,
                    n_timesteps=self.timesteps,
                    temperature=1.5,
                    stoc=False,
                    length_scale=0.91,
                    emb=emb_batch[i]
                )
                hifigan_emb = emb_batch[i].unsqueeze(0)
                audio = (self.vocoder.forward(y_dec, emb=hifigan_emb).to('cpu').squeeze().clamp(
                    -1, 1).numpy() * 32768).astype(
                    np.int16)
                result_audios.append(audio)

        return result_audios
