import os
import shutil
import numpy as np


def write_file(filename, lines):
    """Write lines to file with specified filename"""
    with open(filename, 'w') as f:
        for item in lines:
            f.write("%s" % item)


def copy_only_files(source_dir, output_dir):
    """Copy files from source_dir to output_dir"""
    for dir in os.listdir(source_dir):
        for file in os.listdir(os.path.join(source_dir, dir)):
            if os.path.isdir(file):
                continue
            else:
                shutil.move(os.path.join(source_dir, dir, file), os.path.join(output_dir, file))


def check_correct_files(source_dir):
    """Check that on the second nested level of source_dir: there are only files"""
    incorrected = []
    for dir in os.listdir(source_dir):
        for file in os.listdir(os.path.join(source_dir, dir)):
            if not os.path.isfile(os.path.join(source_dir, dir, file)):
                incorrected.append(os.path.join(source_dir, dir, file))


def validate_with_npy(file, validate_folder):
    """
    Validate that all filepaths listed in file matched with .npy file from validate_folder
    """
    unvalidated = []
    with open(file, 'r') as f:
        filelist = f.readlines()
    for filename in filelist:
        if not os.path.isfile(os.path.join(validate_folder, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy')):
            unvalidated.append(os.path.join(validate_folder, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
    return unvalidated


def remove_files_without_npy(file, validate_folder):
    """
    Removes unmatched with .npy filepaths from file
    """
    with open(file, 'r') as f:
        filelist = f.readlines()
    for filename in filelist:
        if not os.path.isfile(os.path.join(validate_folder, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy')):
            filelist.remove(filename)
    write_file(file, filelist)


def npy_files_to_dir(source_dir, emb_dir):
    "From dataset folder with .wav and .npy files, split .npy files to the emb_dir"
    for d in os.listdir(source_dir):
        for sub_dir in os.listdir(os.path.join(source_dir, d)):
            for pname in os.listdir(os.path.join(source_dir, d, sub_dir)):
                basename, ext = os.path.splitext(pname)
                if ext == '.wav':
                    if os.path.isfile(os.path.join(emb_dir, basename + '.npy')):
                        shutil.copyfile(os.path.join(emb_dir, basename + '.npy'),
                                        os.path.join(source_dir, d, sub_dir, basename + '.npy'))


def validate_npy_paths(source_dir):
    """Validate that all files in source_dir have .npy extension"""
    unvalidated = []
    for d in os.listdir(source_dir):
        for f in os.listdir(os.path.join(source_dir, d)):
            basename, ext = os.path.splitext(f)
            if ext != '.npy':
                unvalidated.append(os.path.join(source_dir, d, f))
    return unvalidated


def create_evaluate_dataset(source_dir, output_dir):
    """Create dataset for validation from wav48_silence_trimmed(VCTK)"""
    # wav48_silence_trimmed
    texts = os.path.join(source_dir, 'txt')
    wavs = os.path.join(source_dir, 'wav48_silence_trimmed')
    for speaker_dir in os.listdir(wavs):

        if speaker_dir[0] == 'p':
            wavs_files = np.array([f for f in os.listdir(os.path.join(wavs, speaker_dir))])
            indices = np.random.choice(len(wavs_files), 2)
            f1, f2 = wavs_files[indices]
            chanks = f2.split('_')
            txt = f'{speaker_dir}_{chanks[1]}.txt'
            if os.path.isfile(os.path.join(texts, speaker_dir, txt)):
                os.mkdir(os.path.join(output_dir, speaker_dir))
                shutil.copyfile(os.path.join(texts, speaker_dir, txt), os.path.join(output_dir, speaker_dir, txt))
                shutil.copyfile(os.path.join(wavs, speaker_dir, f1), os.path.join(output_dir, speaker_dir, f1))
                shutil.copyfile(os.path.join(wavs, speaker_dir, f2), os.path.join(output_dir, speaker_dir, f2))


def change_name(source_dir):
    """Used for change name"""
    for speaker_dir in os.listdir(source_dir):
        for f in os.listdir(os.path.join(source_dir, speaker_dir)):
            basename, ext = os.path.splitext(f)
            if ext == '.wav':
                chunks = basename.split('_')
                os.rename(os.path.join(source_dir, speaker_dir, f),
                          os.path.join(source_dir, speaker_dir, '_'.join(chunks[:-1]) + '.wav'))
