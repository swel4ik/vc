from third_party.wavegrad2.lightning_model import Wavegrad2
from omegaconf import OmegaConf as OC
from third_party.wavegrad2.inference import read_lexicon, preprocess_eng
from timeit import default_timer
import torch
from third_party.wavegrad2.utils.stft import STFTMag
import numpy as np


class WaveGrad2Inference(object):
    def __init__(self,
                 hparams_config: str,
                 wavegrad2_weights: str,
                 lexicon_path: str,
                 device: str = 'cpu',
                 timesteps: int = 1000):

        self.hparams = OC.load(hparams_config)
        self.timesteps = timesteps
        self.device = device
        self.hparams.ddpm.max_step = self.timesteps
        self.lexicon = lexicon_path
        self.model = Wavegrad2(self.hparams).to(self.device)
        self.stft = STFTMag()
        self.ckpt = torch.load(wavegrad2_weights, map_location='cpu')
        self.model.load_state_dict(self.ckpt['state_dict'] if not ('EMA' in wavegrad2_weights) else self.ckpt)

    def __call__(self, texts: list, voice_embedding: np.ndarray) -> list:
        """
        Args:
            texts: List with strings with texts to generate speeches
            voice_embedding: voice embedding with target style

        Returns:
            List of wavefront samples
        """
        result_audios = []
        with torch.no_grad():
            for i, text in enumerate(texts):

                if self.hparams.data.lang == 'eng':
                    text = preprocess_eng(self.hparams, text, lexicon_path=self.lexicon)

                text = text.to(self.device)
                spk_emb = torch.from_numpy(voice_embedding).to(self.device).unsqueeze(0)
                wav_recon, align, *_ = self.model.inference(text, spk_emb, pace=1.0)
                result_wav = wav_recon[0].detach().cpu().numpy()
                result_audios.append(result_wav)

        return result_audios
