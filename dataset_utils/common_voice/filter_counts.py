import pandas as pd
import os
import numpy as np
import shutil
from tqdm import tqdm
import sys
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_tsv')
    parser.add_argument('-d', '--dst_dir')
    parser.add_argument('-w', '--path_to_wavs')


    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_tsv = namespace.path_to_tsv

    data = pd.read_csv(path_to_tsv, sep='\t')

    # value_counts = data['client_id'].value_counts()
    # # преобразование в df и присвоение новых имен колонкам
    # df_value_counts = pd.DataFrame(value_counts)
    # df_value_counts = df_value_counts.reset_index()
    # df_value_counts.columns = ['client_id', 'Audio samples']
    # df_value_counts = df_value_counts[df_value_counts['Audio samples'] > 3]
    #
    # unique_users = df_value_counts.client_id.values

    value_counts = data['client_id'].value_counts()
    # converting to DataFrame and assign new names to columns
    df_value_counts = pd.DataFrame(value_counts)
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = ['client_id', 'Audio samples']
    #
    data_with_counts = pd.merge(data, df_value_counts, how='inner', on='client_id')
    data_with_counts = data_with_counts[data_with_counts['Audio samples'] > 3]

    # current_users = os.listdir(namespace.dst_dir)
    # print(len(unique_users))
    # for user_id in unique_users:
    #     if user_id not in current_users:
    #         print('oh_yeah')
    #         os.mkdir(f'{os.path.join(namespace.dst_dir, user_id)}')

    path_to_wavs = namespace.path_to_wavs
    path_to_dst = namespace.dst_dir

    for ind, sample in tqdm(data_with_counts.iterrows()):
        try:
            shutil.move(os.path.join(path_to_wavs, sample.path), os.path.join(path_to_dst, sample.client_id, sample.path))
        except FileNotFoundError:
            continue

