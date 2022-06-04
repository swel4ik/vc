import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
from enhance_onnx import NSnet2Enhancer


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--path_to_src_dir')
    parser.add_argument('-d', '--path_to_dst_dir')
    parser.add_argument('-m', '--model_path', default='nsnet2-20ms-baseline.onnx')
    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    src_path = namespace.path_to_src_dir
    dst_path = namespace.path_to_dst_dir

    enhancer = NSnet2Enhancer(modelfile=namespace.model_path)
    modelname = Path(namespace.model_path).stem

    for speaker in tqdm(os.listdir(src_path)):
        for sub_speaker in os.listdir(os.path.join(src_path, speaker)):
            # for sample in Path(os.path.join(src_path, speaker, sub_speaker)).glob('*wav'):
            for sample in os.listdir(os.path.join(src_path, speaker, sub_speaker)):
                sigIn, fs = sf.read(os.path.join(src_path, speaker, sub_speaker, sample))

                if len(sigIn.shape) > 1:
                    sigIn = sigIn[:, 0]

                outSig = enhancer(sigIn, fs)

                outpath = os.path.join(dst_path, speaker, sub_speaker, sample)
                sf.write(outpath, outSig, fs)
