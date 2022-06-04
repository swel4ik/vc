# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np
import torch
import os

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from params import seed as random_seed


def extract_filepaths_and_texts_from_root_folder(rt_path: str) -> list:
    result_list = []    # Tuples(MEL_PATH, TEXT)

    for _folder_name in os.listdir(rt_path):
        folder_path = os.path.join(rt_path, _folder_name)
        files_basename_set = []
        for fname in os.listdir(folder_path):
            if len(fname.split('.')) > 2:
                continue
            basename = os.path.splitext(fname)[0]
            if basename not in files_basename_set:
                files_basename_set.append(basename)

        for basename in files_basename_set:
            mel_path = os.path.join(
                folder_path,
                basename + '.mel'
            )
            text_path = os.path.join(
                folder_path,
                basename + '.txt'
            )
            emb_path = os.path.join(
                folder_path,
                basename + '.npy'
            )
            with open(text_path, 'r') as text_f:
                text_lines = [fline.rstrip() for fline in text_f]

            result_list.append(
                (
                    mel_path,
                    emb_path,
                    text_lines[0]
                )
            )

    return result_list


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dataset_path, cmudict_path, add_blank=True):
        self.filepaths_and_text = extract_filepaths_and_texts_from_root_folder(
            root_dataset_path
        )
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, emb_filepath, text = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        emb = self.get_emb(emb_filepath)
        return (text, mel, emb)

    def get_emb(self, filepath):
        emb = torch.from_numpy(np.load(filepath)).float()
        return emb

    def get_mel(self, filepath):
        mel = torch.from_numpy(np.load(filepath)).float()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel, emb = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text, 'embedding': emb}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]
        emb_size = batch[0]['embedding'].shape[-1]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        emb = torch.zeros((B, emb_size), dtype=torch.float32)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_, emb_ = item['y'], item['x'], item['embedding']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            emb[i] = emb_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'embedding': emb}
