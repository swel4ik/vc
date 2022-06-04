from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
import numpy as np
import os
import tempfile
from timeit import default_timer as time
from tqdm import tqdm
import subprocess


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        '--model', required=True, type=str,
        choices=['wavegrad2', 'gradtts'],
        help='Model type to make inference'
    )
    parser.add_argument(
        '--weights', required=True, type=str,
        help='Path to model weights file'
    )
    parser.add_argument(
        '--text', required=True, type=str,
        help='Text for inference'
    )
    parser.add_argument(
        '--src', required=True, type=str,
        help='Path to folder with reference voices'
    )
    parser.add_argument(
        '--result', required=True, type=str,
        help='Path to result folder'
    )
    parser.add_argument(
        '--gpus', required=False, type=int, default=1,
        help='Count of usage GPUs'
    )
    return parser.parse_args()


def imap_unordered(func, args, n_processes=2):
    p = Pool(n_processes)
    res_list = []
    for res in p.imap_unordered(func, args):
        res_list.append(res)
    p.close()
    p.join()
    return res_list


def run_wavegrad2_inference(
        model_weights_path: str,
        text: str,
        emb_path: str,
        res_path: str,
        device_id: int = 0) -> float:
    run_command = 'cd third_party/wavegrad2/;' \
                  'CUDA_VISIBLE_DEVICES={} python3 inference.py ' \
                  '-c {} --text "{}" --speaker_emb {} --result_folder {}'
    run_command = run_command.format(
        device_id,
        model_weights_path,
        text,
        emb_path,
        res_path
    )

    start_time = time()
    process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    finish_time = time()

    return finish_time - start_time


def run_gradtts_inference(
        model_weights_path: str,
        text: str,
        emb_path: str,
        res_path: str,
        device_id: int = 0) -> float:
    run_command = 'cd third_party/Grad-TTS/;' \
                  'CUDA_VISIBLE_DEVICES={} python3 inference.py ' \
                  '-f {} -o {} -c {} -e {}'

    fd, tmp_file_path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(text)

        run_command = run_command.format(
            device_id,
            tmp_file_path,
            res_path,
            model_weights_path,
            emb_path
        )

        start_time = time()
        process = subprocess.Popen(
            run_command, shell=True, stdout=subprocess.PIPE
        )
        process.wait()
        finish_time = time()
    finally:
        os.remove(tmp_file_path)

    return finish_time - start_time


def inference_worker(input_tuple: tuple) -> list:
    inp_data, device_id, run_func = input_tuple
    loop_generator = tqdm(inp_data) if device_id == 0 else inp_data
    inference_times = []
    for sample in loop_generator:
        embedding_path, res_folder, gen_text, weights = sample
        os.makedirs(res_folder, exist_ok=True)

        inference_times.append(
            run_func(
                weights, gen_text, embedding_path, res_folder, device_id
            )
        )

    return inference_times


def main():
    args = parse_args()

    run_function = run_wavegrad2_inference \
        if args.model == 'wavegrad2' \
        else run_gradtts_inference

    print('Model: {}'.format(args.model))

    to_process_data = []

    for actor_folder in os.listdir(args.src):
        actor_folder_path = os.path.join(args.src, actor_folder)
        if '.DS_Store' in actor_folder:
            continue

        all_samples = [
            s for
            s in sorted(os.listdir(actor_folder_path))
            if os.path.splitext(s)[1] == '.npy'
        ]
        first_and_last_samples = [all_samples[0], all_samples[-1]]

        for sample_name in first_and_last_samples:
            basename, ext = os.path.splitext(sample_name)
            if ext != '.npy':
                continue

            embedding_path = os.path.join(actor_folder_path, sample_name)
            res_folder = os.path.join(args.result, actor_folder, basename)

            to_process_data.append(
                [embedding_path, res_folder, args.text, args.weights]
            )

    if args.gpus > 1:
        chunks_data = np.array_split(to_process_data, args.gpus)
        chunks_data_and_ids = [
            (cf, i, run_function)
            for i, cf in enumerate(chunks_data)
        ]

        res_times = imap_unordered(inference_worker, chunks_data_and_ids)
        res_times = [sub_t for t in res_times for sub_t in t]
    else:
        res_times = inference_worker((to_process_data, 0, run_function))

    res_times = np.array(res_times)
    print('Average inference time: {:.2f} sec'.format(res_times.mean()))


if __name__ == '__main__':
    main()
