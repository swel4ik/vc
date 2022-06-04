from argparse import ArgumentParser, Namespace
import os
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Check all files in voice dataset')
    parser.add_argument(
        '-p', '--dataset-path', required=True, type=str
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dpath = args.dataset_path

    for person_folder in tqdm(os.listdir(dpath)):
        person_folder_path = os.path.join(dpath, person_folder)

        files_basename_set = []
        for fname in os.listdir(person_folder_path):
            if len(fname.split('.')) > 2:
                # print('Multiple dots in: {}'.format(fname))
                continue
            basename = os.path.splitext(fname)[0]
            if basename not in files_basename_set:
                files_basename_set.append(basename)

        for basename in files_basename_set:
            wav_path = os.path.join(
                person_folder_path,
                basename + '.wav'
            )

            emb_path = os.path.join(
                person_folder_path,
                basename + '.npy'
            )

            textdrid = os.path.join(
                person_folder_path,
                basename + '.TextGrid'
            )

            text = os.path.join(
                person_folder_path,
                basename + '.txt'
            )

            for file in [wav_path, emb_path, textdrid, text]:
                if not os.path.isfile(file):
                    print('Can\'t see file: {}'.format(file))
