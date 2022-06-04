#This code is adopted from
#https://github.com/ivanvovk/WaveGrad
import numpy as np

import torch

from third_party.wavegrad2.model.base import BaseModule
from third_party.wavegrad2.model.layers import Conv1dWithInitialization
from third_party.wavegrad2.model.upsampling import UpsamplingBlock as UBlock
from third_party.wavegrad2.model.downsampling import DownsamplingBlock as DBlock
from third_party.wavegrad2.model.linear_modulation import FeatureWiseLinearModulation as FiLM


class WaveGradNN(BaseModule):
    """
    WaveGrad is a fully-convolutional mel-spectrogram conditional
    vocoder model for waveform generation introduced in
    "WaveGrad: Estimating Gradients for Waveform Generation" paper (link: https://arxiv.org/pdf/2009.00713.pdf).
    The concept is built on the prior work on score matching and diffusion probabilistic models.
    Current implementation follows described architecture in the paper.
    """
    def __init__(self, hparams):
        super(WaveGradNN, self).__init__()
        # Building upsampling branch (mels -> signal)
        self.hparams = hparams
        self.ublock_preconv = Conv1dWithInitialization(
            in_channels=hparams.wavegrad.encode_channel,
            out_channels=hparams.wavegrad.upsample.preconv_channel,
            kernel_size=3,
            stride=1,
            padding=1
        )
        upsampling_in_sizes = [hparams.wavegrad.upsample.preconv_channel] \
            + hparams.wavegrad.upsample.out_channels[:-1]
        self.ublocks = torch.nn.ModuleList([
            UBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                upsampling_in_sizes,
                hparams.wavegrad.upsample.out_channels,
                hparams.wavegrad.scale_factors,
                hparams.wavegrad.upsample.dilations
            )
        ])
        self.ublock_postconv = Conv1dWithInitialization(
            in_channels=hparams.wavegrad.upsample.out_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Building downsampling branch (starting from signal)
        self.dblock_preconv = Conv1dWithInitialization(
            in_channels=1,
            out_channels=hparams.wavegrad.downsample.preconv_channel,
            kernel_size=5,
            stride=1,
            padding=2
        )
        downsampling_in_sizes = [hparams.wavegrad.downsample.preconv_channel] \
            + hparams.wavegrad.downsample.out_channels[:-1]
        self.dblocks = torch.nn.ModuleList([
            DBlock(
                in_channels=in_size,
                out_channels=out_size,
                factor=factor,
                dilations=dilations
            ) for in_size, out_size, factor, dilations in zip(
                downsampling_in_sizes,
                hparams.wavegrad.downsample.out_channels,
                hparams.wavegrad.scale_factors[1:][::-1],
                hparams.wavegrad.downsample.dilations
            )
        ])

        # Building FiLM connections (in order of downscaling stream)
        film_in_sizes = [hparams.wavegrad.downsample.preconv_channel] \
                + list(hparams.wavegrad.downsample.out_channels)
        film_out_sizes = hparams.wavegrad.upsample.out_channels[::-1]
        film_factors = [1] + hparams.wavegrad.scale_factors[1:][::-1]
        self.films = torch.nn.ModuleList([
            FiLM(
                in_channels=in_size,
                out_channels=out_size,
                input_dscaled_by=np.product(film_factors[:i+1]),  # for proper positional encodings initialization
                linear_scale=hparams.ddpm.pos_emb_scale
            ) for i, (in_size, out_size) in enumerate(
                zip(film_in_sizes, film_out_sizes)
            )
        ])

    def forward(self, yn, hidden_rep, noise_level):
        """
        Computes forward pass of neural network.
        :param mels (torch.Tensor): mel-spectrogram acoustic features of shape [B, n_mels, T//hop_length]
        :param yn (torch.Tensor): noised signal `y_n` of shape [B, T]
        :param noise_level (float): level of noise added by diffusion
        :return (torch.Tensor): epsilon noise
        """
        # Prepare inputs
        assert len(hidden_rep.shape) == 3  # B, n_mels, T
        yn = yn.unsqueeze(1)
        assert len(yn.shape) == 3  # B, 1, T
        # Downsampling stream + Linear Modulation statistics calculation
        statistics = []
        dblock_outputs = self.dblock_preconv(yn)
        scale, shift = self.films[0](x=dblock_outputs, noise_level=noise_level)
        statistics.append([scale, shift])
        for dblock, film in zip(self.dblocks, self.films[1:]):
            dblock_outputs = dblock(dblock_outputs)
            scale, shift = film(x=dblock_outputs, noise_level=noise_level)
            statistics.append([scale, shift])
        statistics = statistics[::-1]

        # Upsampling stream
        ublock_outputs = self.ublock_preconv(hidden_rep)
        for i, ublock in enumerate(self.ublocks):
            scale, shift = statistics[i]
            ublock_outputs = ublock(x=ublock_outputs, scale=scale, shift=shift)
        outputs = self.ublock_postconv(ublock_outputs)
        return outputs.squeeze(1)
