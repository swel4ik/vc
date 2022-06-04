from enum import Enum
from pathlib import Path
import numpy as np

from third_party.Grad_TTS.server_inference import GradTTSInference
from third_party.RTVC.encoder import inference as encoder
from third_party.wavegrad2.server_inference import WaveGrad2Inference


class ErrorState(Enum):
    SUCCESSFULLY = 0


class InferenceEngine(object):
    EMBEDDING_SIZE = 256

    def __init__(self, server_config: dict):
        self.config = server_config

        self.speaker_encoder = encoder
        self.speaker_encoder.load_model(Path(self.config['VoiceEncoder']['checkpoint_path']))

        self.synthesis_model = None
        model_name = self.config['Global_parameters']['used_model']

        if model_name == 'Grad-TTS':
            self.synthesis_model = GradTTSInference(
                hifi_config=self.config['Hi-Fi_Gan']['config_path'],
                hifi_weights=self.config['Hi-Fi_Gan']['checkpoint_path'],
                grad_tts_weights=self.config['Grad-TTS']['checkpoint_path'],
                corpus_dictionary_path=self.config['Grad-TTS']['corpus_dictionary_path'],
                device=self.config['Global_parameters']['inference_device'],
                timesteps=self.config['Grad-TTS']['timesteps']
            )
        elif model_name == 'WaveGrad2':
            self.synthesis_model = WaveGrad2Inference(
                hparams_config=self.config['WaveGrad2']['hparams_config_path'],
                wavegrad2_weights=self.config['WaveGrad2']['checkpoint_path'],
                lexicon_path=self.config['WaveGrad2']['lexicon_path'],
                device=self.config['Global_parameters']['inference_device'],
                timesteps=self.config['WaveGrad2']['timesteps']
            )
        else:
            raise RuntimeError(
                'Model architecture \'{}\' isn\'t supported'.format(model_name)
            )

        self.batch_size = self.config['Global_parameters']['batch_size']

    def _make_embedding(self, record: np.ndarray, source_sampling_rate: int) -> np.ndarray:
        """
        Make voice embedding
        Args:
            record: wavefront to extract voice style

        Returns:
            Embedding float vector
        """
        processed_record = self.speaker_encoder.preprocess_wav(record, source_sr=source_sampling_rate)
        embedding = self.speaker_encoder.embed_utterance(processed_record)
        return embedding

    def __call__(self, texts: list, target_voice: np.ndarray, source_sampling_rate: int) -> list:
        """
        Inference method
        Args:
            texts: list of texts to make inference
            target_voice: target wavefront to extract voice style

        Returns:
            List of wavefront samples
        """
        batches_count = len(texts) // self.batch_size + (
                len(texts) % self.batch_size > 0)
        batches = np.array_split(texts, batches_count)
        embedding = self._make_embedding(target_voice, source_sampling_rate)

        synthesised_records = [
            rec
            for text_batch in batches
            for rec in self.synthesis_model(text_batch, embedding)
        ]

        return synthesised_records
