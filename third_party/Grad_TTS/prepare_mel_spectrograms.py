from argparse import ArgumentParser, Namespace
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import torch

from third_party.Grad_TTS.hifi_gan_legacy.meldataset import mel_spectrogram, load_wav, MAX_WAV_VALUE


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description='Predict mel-spectrograms for LibriTTS dataset.'
    )
    parser.add_argument(
        '-s', required=True, type=str,
        help='Path to processed LibriTTS dataset.'
    )
    parser.add_argument(
        '--njobs', required=False, type=int, default=4,
        help='Parallel processes.'
    )
    return parser.parse_args()


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def worker_task(folders_list_and_worker_id: tuple):
    folders_list, wid = folders_list_and_worker_id
    loop_generator = tqdm(folders_list) if wid == 0 else folders_list
    for folder_path in loop_generator:
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

            if os.path.isfile(mel_path):
                continue

            wav_path = os.path.join(
                folder_path,
                basename + '.wav'
            )

            audio, sampling_rate = load_wav(wav_path)
            audio = audio / MAX_WAV_VALUE
            audio = torch.FloatTensor(audio).unsqueeze(0)

            # (y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False)
            mel_spec = mel_spectrogram(
                audio,
                1024,
                80,
                22050,
                256,
                1024,
                0.0,
                8000.0
            ).squeeze(0).to('cpu').numpy()

            np.save(mel_path, mel_spec)
            os.rename(
                mel_path + '.npy',
                mel_path
            )


if __name__ == '__main__':
    args = parse_args()
    ROOT_FOLDER = args.s
    NJOBS = args.njobs

    folders = [os.path.join(ROOT_FOLDER, f) for f in os.listdir(ROOT_FOLDER)]

    chunks_folders = np.array_split(folders, NJOBS)
    chunks_folders_and_ids = [
        (cf, i)
        for i, cf in enumerate(chunks_folders)
    ]

    imap_unordered_bar(worker_task, chunks_folders_and_ids, NJOBS)
