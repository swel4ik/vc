import argparse
import itertools
import tqdm
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Remove last padding zeros from mel_spectrograms")
    parser.add_argument('--dir', required=True, help="Path directory with grad_tts melspectrograms")

    return parser.parse_args()


def trim_mels_on_dir(dir):
    for f in tqdm.tqdm(os.listdir(dir)):
        filepath = os.path.join(dir, f)
        data = np.load(filepath, allow_pickle=True)
        if data.ndim == 3:
            continue
        group_list = list(map(lambda x: (x[0], len(list(x[1]))), itertools.groupby(data[0])))
        if group_list[-1][0] == -0.0:
            np.save(filepath, np.expand_dims(data[:, :-group_list[-1][1]], axis=0))


if __name__ == '__main__':
    args = parse_args()
    trim_mels_on_dir(args.dir)
