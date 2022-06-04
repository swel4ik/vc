import os
from tqdm import tqdm
from multiprocess.pool import ThreadPool
import argparse
import shutil
import sys

'''
Process LibriTTS structure (or similar) to WaveGrad 2 pipeline requirements
'''


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    return parser


def move_speakers(current_speakers, full_data_path):
    dir_with_speakers = full_data_path

    def move_speaker(path_to_speaker, full_data_path=dir_with_speakers):
        for sub_speaker in os.listdir(os.path.join(full_data_path, path_to_speaker)):
            for sample in os.listdir(os.path.join(full_data_path, path_to_speaker, sub_speaker)):
                if sample[-3:] == 'tsv':
                    os.remove(os.path.join(full_data_path, path_to_speaker, sub_speaker, sample))
                    continue
                shutil.move(os.path.join(full_data_path, path_to_speaker, sub_speaker, sample),
                            os.path.join(full_data_path, path_to_speaker, sample))
            os.rmdir(os.path.join(full_data_path, path_to_speaker, sub_speaker))
                # if 'original' in sample or 'alignment' in sample:
                #     os.remove(os.path.join(in_dir, path_to_speaker, sub_speaker, sample))
                #     continue
                # if sample[-3:] == 'txt':
                #     parts = sample.split('.')
                #     os.rename(os.path.join(in_dir, path_to_speaker, sub_speaker, sample),
                #               os.path.join(in_dir, path_to_speaker, sub_speaker, parts[0] + '.txt'))
                # else:
                #     continue

    with ThreadPool(6) as pool:
        list(tqdm(pool.imap(move_speaker, current_speakers),
                  'LibriTTS', len(current_speakers),
                  unit="speakers"))


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_speakers_dir = namespace.path_to_speakers_dir
    speakers = os.listdir(path_to_speakers_dir)

    move_speakers(speakers, path_to_speakers_dir)
