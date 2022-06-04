import os
from tqdm import tqdm
import argparse
import shutil
import sys
from multiprocess.pool import ThreadPool


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    parser.add_argument('-g', '--path_to_grids_dir')

    return parser



# in_dir = '/media/peacock/WD4VOICE/datasets/text_to_speech/LibriTTS/LibriTTS-100/train-clean-100'
# grid_dir = '/media/peacock/WD4VOICE/datasets/text_to_speech/LibriTTS (align)/textgrid/train-clean-100'
# speakers = os.listdir(in_dir)


def mv_grids(speaker_dirs, full_data_path: str, grids_dir: str):
    in_dir = full_data_path
    grid_dir = grids_dir

    def move_one_grid(speaker):
        for sub_speaker in os.listdir(os.path.join(grid_dir, speaker)):
            if os.path.isdir(os.path.join(grid_dir, speaker, sub_speaker)):
                for sample in os.listdir(os.path.join(grid_dir, speaker, sub_speaker)):
                    if '-' in sample:
                        new_name = sample.replace('-', '_')
                        os.rename(os.path.join(grid_dir, speaker, sub_speaker, sample),
                                  os.path.join(grid_dir, speaker, sub_speaker, new_name))
                        sample = new_name

                    shutil.copy(os.path.join(grid_dir, speaker, sub_speaker, sample),
                                os.path.join(in_dir, speaker, sub_speaker, sample))

    with ThreadPool(6) as pool:
        list(tqdm(pool.imap(move_one_grid, speakers),
                  'LibriTTS', len(speakers),
                  unit="speakers"))


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_speakers_dir = namespace.path_to_speakers_dir
    grid_dir = namespace.path_to_grids_dir
    speakers = os.listdir(path_to_speakers_dir)

    mv_grids(speakers, path_to_speakers_dir, grid_dir)



