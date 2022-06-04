import sys
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import tgt
from multiprocess.pool import ThreadPool

'''
Convert .TextGrid files to alignments.txt for synthesizer train (only for SV2TTS)
'''


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    parser.add_argument('-g', '--path_to_grids_dir')

    return parser

# dataset_path = Path('/media/peacock/WD4VOICE/datasets/text_to_speech/LibriTTS/LibriTTS-100/train-clean-100')
# # base_path = Path('/output/montreal-aligned/{}'.format(DATASET))
# base_path = Path('/media/peacock/WD4VOICE/datasets/text_to_speech/LibriTTS (align)/textgrid/train-clean-100')


def grid_to_text_speakers(speaker_dirs):

    def preprocess_speaker(speaker_dir):
        book_dirs = [f for f in speaker_dir.glob("*") if f.is_dir()]
        for book_dir in book_dirs:
            alignment_file = dataset_path.joinpath(
                speaker_dir.stem,
                book_dir.stem,
                "{0}_{1}.alignment.txt".format(speaker_dir.stem, book_dir.stem)
            )

            with open(alignment_file, 'w', encoding='utf-8') as out_file:
                # find our textgrid files
                textgrid_files = sorted([f for f in book_dir.glob("*.TextGrid") if f.is_file()])

                # process each grid file and add to our output
                for textgrid_file in textgrid_files:
                    # read the raw transcript as well
                    transcript_file = dataset_path.joinpath(
                        speaker_dir.stem,
                        book_dir.stem,
                        "{0}.txt".format(textgrid_file.stem)
                    )
                    try:
                        with open(transcript_file, 'r', encoding='utf-8') as in_file:
                            transcript = in_file.read()
                    except FileNotFoundError:
                        new_end = textgrid_file.stem.replace('-', '_')
                        transcript_file = dataset_path.joinpath(
                            speaker_dir.stem,
                            book_dir.stem,
                            "{0}.txt".format(new_end)
                        )
                        with open(transcript_file, 'r', encoding='utf-8') as in_file:
                            transcript = in_file.read()


                    # read the grid
                    input = tgt.io.read_textgrid(textgrid_file)
                    # sys.exit(1)

                    # get all the word tiers
                    word_tier = input.get_tier_by_name('words')

                    out_file.write("{0} \"{1}\" \"{2}\" {3}\n".format(
                        textgrid_file.stem,
                        ",".join(list(map(lambda interval: interval.text, word_tier.intervals))),
                        ",".join(list(map(lambda interval: str(interval.end_time), word_tier.intervals))),
                        transcript
                    ))
    with ThreadPool(6) as pool:
        list(tqdm(pool.imap(preprocess_speaker, speaker_dirs),
                  'LibriTTS', len(speaker_dirs),
                  unit="speakers"))

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_speakers_dir = namespace.path_to_speakers_dir
    path_to_grids_dir = namespace.path_to_grids_dir

    speaker_dirs = [f for f in base_path.glob("*") if f.is_dir()]
    grid_to_text_speakers(speaker_dirs=speaker_dirs)