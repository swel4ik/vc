import argparse
import shutil
import sys
import os


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_src_dir')
    parser.add_argument('-d', '--path_to_dst_dir')
    return parser


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    src_path = namespace.path_to_src_dir
    dst_path = namespace.path_to_dst_dir

    shutil.copytree(src_path, dst_path, ignore=ig_f)