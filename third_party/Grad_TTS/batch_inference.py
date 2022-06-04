import os
import argparse

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse
import torch
import numpy as np
import datetime
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Batch inference of Grad_TTS model")
    parser.add_argument("--src", required=True, help="Source directory in LibriTTS format")
    parser.add_argument("--result", required=True, help="Output directory")
    parser.add_argument("--embs_dir", required=True, help="Embedding directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the model")

    return parser.parse_args()


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, root_dataset_path, output_folder, embs_dir):
        self.filepaths_and_text = []
        self.cmudict = cmudict.CMUDict('./resources/cmu_dictionary')
        for actor_folder in os.listdir(root_dataset_path):
            all_pairs = []
            for sub_actor_folder in os.listdir(os.path.join(root_dataset_path, actor_folder)):
                sub_actor_folder_path = os.path.join(root_dataset_path, actor_folder, sub_actor_folder)
                if '.DS_Store' in actor_folder:
                    continue

                all_samples = [
                    s for
                    s in sorted(os.listdir(sub_actor_folder_path))
                    if os.path.splitext(s)[1] == '.wav'
                ]

                res_folder = os.path.join(output_folder)
                all_samples = [
                    f for
                    f in all_samples
                    if not os.path.isfile(os.path.join(res_folder, os.path.splitext(f)[0] + '.npy'))
                ]

                all_samples = [
                    f for
                    f in all_samples
                    if os.path.isfile(os.path.join(embs_dir, os.path.splitext(f)[0] + '.npy'))
                ]

                for sample_name in all_samples:
                    basename, ext = os.path.splitext(sample_name)

                    embedding_path = os.path.join(embs_dir, basename + '.npy')

                    with open(os.path.join(sub_actor_folder_path, basename + '.original.txt')) as f:
                        text = f.read().strip()
                    all_pairs.append((embedding_path, text, os.path.join(res_folder, basename + '.npy')))
            self.filepaths_and_text.extend(all_pairs)

    def get_pair(self, filepath_and_text):
        emb_filepath, text, filename = filepath_and_text[0], filepath_and_text[1], filepath_and_text[2]
        text = self.get_text(text)
        emb = self.get_emb(emb_filepath)
        return (text, emb, filename)

    def get_emb(self, filepath):
        emb = torch.from_numpy(np.load(filepath)).float()
        return emb

    def get_text(self, text):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if True:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, emb, filename = self.get_pair(self.filepaths_and_text[index])
        item = {'x': text, 'embedding': emb, 'filename': filename}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        emb_size = batch[0]['embedding'].shape[-1]

        x = torch.zeros((B, x_max_length), dtype=torch.long)
        emb = torch.zeros((B, emb_size), dtype=torch.float32)
        x_lengths = []
        filenames = []

        for i, item in enumerate(batch):
            x_, emb_, filename = item['x'], item['embedding'], item['filename']
            x_lengths.append(x_.shape[-1])
            x[i, :x_.shape[-1]] = x_
            emb[i] = emb_
            filenames.append(filename)

        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'embedding': emb,
                'filename': filenames}


## currently working with directories in LibriTTS
if __name__ == '__main__':
    a = parse_args()
    dataset = TextMelDataset(a.src, a.result, a.embs_dir)
    batch_size = 8
    batch_collate = TextMelBatchCollate()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=batch_collate,
                                         num_workers=4, shuffle=False)

    generator = GradTTS(len(symbols) + 1, params.embedding_size, params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(a.checkpoint, map_location=lambda loc, storage: loc))
    generator = generator.to('cuda')
    generator.eval()
    begin_time = datetime.datetime.now()
    with tqdm(loader, total=len(dataset) // batch_size) as progress_bar:
        for batch_idx, batch in enumerate(progress_bar):
            x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
            emb = batch['embedding'].cuda()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=10, temperature=1.5,
                                                   stoc=False, length_scale=0.91, emb=emb)
            for i, filename in enumerate(batch['filename']):
                np.save(filename, y_dec[i].detach().cpu().numpy())
    print('Average execution time: ', (datetime.datetime.now() - begin_time) / len(dataset))
