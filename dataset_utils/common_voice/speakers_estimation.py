from os import path
import argparse
import glob
import sys
import os
from tqdm import tqdm
import shutil


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    parser.add_argument('-t', '--samples_thresh')

    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_main_dir = namespace.path_to_speakers_dir
    threshold = namespace.samples_thresh

    count = 0
    for speaker in tqdm(glob.glob(f"{path_to_main_dir}/common_voice_*")):
        if len(os.listdir(speaker)) - 1 < threshold:
            count += 1
    print(f'{count} speakers with less than {threshold} samples')
