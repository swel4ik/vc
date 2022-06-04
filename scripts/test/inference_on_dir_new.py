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
        '-m', '--model', required=True, type=str,
        choices=['encoder', 'wavegrad2', 'gradtts', 'gradtts_only'],
        help='Model type to make inference'
    )
    parser.add_argument(
        '-w', '--weights', required=True, type=str,
        help='Path to model weights file'
    )
    parser.add_argument(
        '-t', '--text', required=False, type=str,
        help='Text for inference'
    )
    parser.add_argument(
        '-s', '--src', required=True, type=str,
        help='Path to folder with reference voices'
    )
    parser.add_argument(
        '-r', '--result', required=False, type=str,
        help='Path to result folder'
    )
    parser.add_argument(
        '-i', '--hifi_model', required=False, type=str,
        help='Path to hifi-gan model chekpoint, if not provided used default'
    )
    parser.add_argument(
        '-c', '--hifi_config', required=False, type=str,
        help='Path to hifi-gan config, if not provided used default'
    )
    parser.add_argument(
        '-g', '--gpus', required=False, type=int, default=1,
        help='Count of usage GPUs'
    )
    parser.add_argument(
        '-v', '--val_data', required=False, type=str, default=None,
        help='Path to validation data, if we use this function to validate'
    )

    return parser.parse_args()


def create_params_dict(args):
    return {
        'model': args.model,
        'weights': args.weights,
        'text': args.text,
        'src': args.src,
        'result': args.result,
        'hifi_model': args.hifi_model,
        'hifi_config': args.hifi_config,
        'gpus': args.gpus,
        'val_data': args.val_data
    }


def imap_unordered(func, args, n_processes=2):
    p = Pool(n_processes)
    res_list = []
    for res in p.imap_unordered(func, args):
        res_list.append(res)
    p.close()
    p.join()
    return res_list


def run_voice_encoder_inference(
        model_weights_path: str,
        speakers_path: str,
        device_id: int = 0
) -> float:
    run_command = 'cd third_party/RTVC/;' \
                  'python3 prepare_test_embeds.py ' \
                  '-m {} -s {}'
    run_command = run_command.format(
        model_weights_path,
        speakers_path
    )
    start_time = time()
    process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    finish_time = time()

    return finish_time - start_time


def run_wavegrad2_inference(
        device_id: int = 0, **kwargs) -> float:
    run_command = 'cd third_party/wavegrad2/;' \
                  'CUDA_VISIBLE_DEVICES={} python3 inference.py ' \
                  '-c {} --text "{}" --speaker_emb {} --result_folder {}'
    run_command = run_command.format(
        device_id,
        kwargs['model_weights_path'],
        kwargs['text'],
        kwargs['emb_path'],
        kwargs['res_path']
    )

    start_time = time()
    process = subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    finish_time = time()

    return finish_time - start_time


def run_gradtts_inference(device_id=0, **kwargs) -> float:
    fd, tmp_file_path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(kwargs['text'])
        run_command = 'cd third_party/Grad_TTS/;' \
                      'CUDA_VISIBLE_DEVICES={} python3 inference.py ' \
                      '-f {} -o {} -c {} -e {} -i {} -g {}'
        run_command = run_command.format(
            device_id,
            tmp_file_path,
            kwargs['res_path'],
            kwargs['model_weights_path'],
            kwargs['emb_path'],
            kwargs['hifi_model_weights_path'],
            kwargs['hifi_config']
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


def run_gradtts_only_inference(device_id=0, **kwargs):
    fd, tmp_file_path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write(kwargs['text'])
        run_command = 'cd third_party/Grad_TTS/;' \
                      'CUDA_VISIBLE_DEVICES={} python3 inference_without_hifigan.py ' \
                      '-f {} -o {} -c {} -e {}'
        run_command = run_command.format(
            device_id,
            tmp_file_path,
            kwargs['res_path'],
            kwargs['model_weights_path'],
            kwargs['emb_path'],
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
        os.makedirs(sample['res_path'], exist_ok=True)

        inference_times.append(
            run_func(
                **sample
            )
        )

    return inference_times


def inference_on_dir(args):
    model_funcs = {
        'wavegrad': run_wavegrad2_inference,
        'gradtts': run_gradtts_inference,
        'gradtts_only': run_gradtts_only_inference
    }
    if args['model'] == 'encoder':
        run_voice_encoder_inference(args['weights'], args['src'])
    else:
        run_function = model_funcs[args['model']]
        print('Model: {}'.format(args['model']))

        to_process_data = []

        for actor_folder in os.listdir(args['src']):
            actor_folder_path = os.path.join(args['src'], actor_folder)
            if '.DS_Store' in actor_folder:
                continue

            all_samples = [
                s for
                s in sorted(os.listdir(actor_folder_path))
                if os.path.splitext(s)[1] == '.npy'
            ]
            # if all_samples == []:
            #     break
            # first_and_last_samples = [all_samples[0], all_samples[-1]]

            for sample_name in all_samples:
                basename, ext = os.path.splitext(sample_name)
                if ext != '.npy':
                    continue

                embedding_path = os.path.join(actor_folder_path, sample_name)
                res_folder = os.path.join(args['result'], actor_folder)

                if args['text'] is None:
                    for pname in os.listdir(os.path.join(args['val_data'], actor_folder)):
                        name, ext_ = os.path.splitext(pname)
                        if ext_ == '.txt':
                            with open(os.path.join(args['val_data'], actor_folder, pname), 'r') as f:
                                text = f.read().strip()
                else:
                    text = args['text']
                params = {'emb_path': embedding_path,
                          'model_weights_path': args['weights'],
                          'res_path': res_folder,
                          'text': text,
                          'hifi_model_weights_path': args['hifi_model'],
                          'hifi_config': args['hifi_config']
                          }
                to_process_data.append(
                    params
                )

        if args['gpus'] > 1:
            chunks_data = np.array_split(to_process_data, args['gpus'])
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
    args = parse_args()
    params = create_params_dict(args)
    inference_on_dir(params)
