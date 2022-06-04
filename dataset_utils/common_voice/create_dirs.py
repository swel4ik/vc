import pandas as pd
import os
import numpy as np
import shutil

import sys
import argparse


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_tsv')
    parser.add_argument('-d', '--dst_dir')

    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    path_to_tsv = namespace.path_to_tsv

    data = pd.read_csv(path_to_tsv, sep='\t')

    value_counts = data['client_id'].value_counts()
    # converting to DataFrame and assign new names to columns
    df_value_counts = pd.DataFrame(value_counts)
    df_value_counts = df_value_counts.reset_index()
    df_value_counts.columns = ['client_id', 'Audio samples']
    df_value_counts = df_value_counts[df_value_counts['Audio samples'] > 3]

    unique_users = df_value_counts.client_id.values

    for user_id in unique_users:
        os.mkdir(f'{os.path.join(namespace.dst_dir, user_id)}')
