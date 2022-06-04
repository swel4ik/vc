import argparse
import sys
from encoder import inference as encoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


def embed_utterance_new(wav_path, encoder_model_fpath):
    if not encoder.is_loaded():
        encoder.load_model(encoder_model_fpath)

    # Compute the speaker embedding of the utterance
    wav_fpath = wav_path
    wav = np.load(wav_fpath)
    wav = encoder.preprocess_wav(wav)
    embed = encoder.embed_utterance(wav)
    np.save(wav_path, embed, allow_pickle=False)


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_speakers_dir')
    parser.add_argument('-m', '--model_path')

    return parser


# def create_embeddings(synthesizer_root: Path, encoder_model_fpath: Path, n_processes: int):
#     wav_dir = synthesizer_root.joinpath("audio")
#     metadata_fpath = synthesizer_root.joinpath("train.txt")
#     assert wav_dir.exists() and metadata_fpath.exists()
#     embed_dir = synthesizer_root.joinpath("embeds")
#     embed_dir.mkdir(exist_ok=True)
#
#     # Gather the input wave filepath and the target output embed filepath
#     with metadata_fpath.open("r") as metadata_file:
#         metadata = [line.split("|") for line in metadata_file]
#         fpaths = [(wav_dir.joinpath(m[0]), embed_dir.joinpath(m[2])) for m in metadata]
#
#     # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
#     # Embed the utterances in separate threads
#     func = partial(embed_utterance, encoder_model_fpath=encoder_model_fpath)
#     job = Pool(n_processes).imap(func, fpaths)
#     list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))

if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    in_dir = namespace.path_to_speakers_dir
    encoder_model_fpath = Path(namespace.model_path)



    speakers = os.listdir(in_dir)

    for path_to_speaker in tqdm(speakers):
        if not encoder.is_loaded():
            encoder.load_model(encoder_model_fpath)
        # for sub_speaker in os.listdir(os.path.join(in_dir, path_to_speaker)):
        for sample in os.listdir(os.path.join(in_dir, path_to_speaker)):
            if sample[-3:] == 'wav':
                # wav = np.load(os.path.join(in_dir, path_to_speaker, sub_speaker, sample))
                wav = encoder.preprocess_wav(os.path.join(in_dir, path_to_speaker, sample))
                embed = encoder.embed_utterance(wav)
                np.save(os.path.join(in_dir, path_to_speaker, sample[:-4]), embed, allow_pickle=False)
            else:
                continue
