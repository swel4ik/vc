import os
from tqdm import tqdm
from multiprocess.pool import ThreadPool
import argparse
import shutil
import sys


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    return parser


#in_dir = '/media/peacock/WD4VOICE/datasets/text_to_speech/LibriTTS/LibriTTS-360/train-clean-360'




def rename_speaker_dirs(current_speakers: list, full_data_path: str):
    in_dir = full_data_path

    def rename_speaker(path_to_speaker):
        for sub_speaker in os.listdir(os.path.join(in_dir, path_to_speaker)):
            for sample in os.listdir(os.path.join(in_dir, path_to_speaker, sub_speaker)):
                if 'original' in sample or 'alignment' in sample:
                    os.remove(os.path.join(in_dir, path_to_speaker, sub_speaker, sample))
                    continue
                if sample[-3:] == 'txt':
                    parts = sample.split('.')
                    os.rename(os.path.join(in_dir, path_to_speaker, sub_speaker, sample),
                              os.path.join(in_dir, path_to_speaker, sub_speaker, parts[0] + '.txt'))
                else:
                    continue

    with ThreadPool(6) as pool:
        list(tqdm(pool.imap(rename_speaker, speakers),
                  'LibriTTS', len(speakers),
                  unit="speakers"))



if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_speakers_dir = namespace.path_to_speakers_dir
    speakers = os.listdir(path_to_speakers_dir)

    rename_speaker_dirs(speakers, path_to_speakers_dir)