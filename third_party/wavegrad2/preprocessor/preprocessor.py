#This code is adopted from
#https://github.com/ming024/FastSpeech2
import os
import random
import json

import tgt
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from shutil import copy


class Preprocessor:
    def __init__(self, hparams):
        self.hparams = hparams
        self.in_dir = hparams.path.raw_path
        self.out_dir = hparams.path.preprocessed_path
        self.val_size = hparams.preprocessing.val_size
        self.sampling_rate = hparams.preprocessing.audio.sampling_rate
        self.scale = hparams.preprocessing.audio.scale

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "embeddings")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0

        # Compute duration
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            if speaker == '.DS_Store':
                continue

            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if wav_name == '.DS_Store':
                    continue

                if os.path.splitext(wav_name)[1] != '.wav':
                    continue

                basename = str(os.path.splitext(wav_name)[0])
                tg_path = os.path.join(self.in_dir, speaker, basename + '.TextGrid')
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, n = ret
                    out.append(info)

                n_frames += n

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        print(
            "Total time: {} hours".format(
                n_frames / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.txt".format(basename))
        emb_path = os.path.join(self.in_dir, speaker, "{}.npy".format(basename))
        # tg_path = os.path.join(
        #     self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        # )
        tg_path = os.path.join(self.in_dir, speaker, basename + '.TextGrid')

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        wav_filename = "{}-wav-{}.wav".format(speaker, basename)
        wavfile.write(
            os.path.join(self.out_dir, "wav", wav_filename),
            self.sampling_rate,
            wav,
        )

        emb_filename = "{}-embedding-{}.npy".format(speaker, basename)
        copy(
            emb_path,
            os.path.join(self.out_dir, 'embeddings', emb_filename)
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            wav.shape[0],
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.scale)
                    - np.round(s * self.sampling_rate / self.scale)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time
