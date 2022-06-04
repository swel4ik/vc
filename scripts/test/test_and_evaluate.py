import argparse

from pathlib import Path
import shutil
import os

from inference_on_dir_new import inference_on_dir
from parse_utility import parse_config, parse_data, parse_data_val


def parse_args():
    parser = argparse.ArgumentParser(
        "Test whole pipeline(voice encoder, model(synthesizer, vocoder) or wavegrad with different checkpoints")
    parser.add_argument('--config', required=True,
                        help="Path to config with model paths")
    parser.add_argument('-m', '--mode', required=True, type=str,
                        choices=['test', 'test_and_evaluate'],
                        help='Mode evaluate or test')
    parser.add_argument('-s', '--save_intermediate', required=False, type=bool,
                        help='Save or not intermediate results')
    return parser.parse_args()


def encoder_command(encoder, data):
    params = {
        'model': 'encoder',
        'weights': encoder,
        'src': data['data']
    }
    inference_on_dir(params)


def wavegrad_command(wavegrad, data, output_name, hifi=None, hifi_config=None):
    params = {
        'model': 'wavegrad',
        'weights': wavegrad,
        'text': data['output_text'] if 'output_text' in data else None,
        'src': data['data'],
        'result': os.path.join(data["output_path"], output_name),
        'gpus': 1,
        'hifi_model': hifi,
        'hifi_config': hifi_config
    }
    if 'val_data' in data:
        params['val_data'] = data['val_data']
    print(params)
    inference_on_dir(params)


def grad_tts_command(grad_tts, data, output_name, hifi=None, hifi_config=None):
    print('data', data)
    params = {
        'model': 'gradtts',
        'weights': grad_tts,
        'text': data['output_text'] if 'output_text' in data else None,
        'src': data['data'],
        'result': os.path.join(data["output_path"], output_name),
        'hifi_model': hifi,
        'hifi_config': hifi_config,
        'gpus': 1,
    }
    if 'val_data' in data:
        params['val_data'] = data['val_data']
    print(params)
    inference_on_dir(params)


def grad_tts_only_command(grad_tts, data, output_name, hifi=None, hifi_config=None):
    params = {
        'model': 'gradtts',
        'weights': grad_tts,
        'text': data['output_text'] if 'output_text' in data else None,
        'src': data['data'],
        'result': os.path.join(data["output_path"], output_name),
        'gpus': 1,
        'hifi': hifi,
        'hifi_config': hifi_config
    }
    if 'val_data' in data:
        params['val_data'] = data['val_data']
    inference_on_dir(params)


def run_command_on_list(data, model_name, output_prefix, hifi_model=None, hifi_config=None):
    models_dict = {'grad_tts': grad_tts_command,
                   'wavegrad': wavegrad_command}
    command = models_dict[model_name]
    ckpts_name = '%s_ckpts' % model_name
    for model in data[ckpts_name]:
        name = Path(model).stem
        if hifi_model is not None:
            command(model, data, "%s_%s" % (output_prefix, name),
                    hifi_model, hifi_config)
        else:
            command(model, data, "%s_%s" % (output_prefix, name))


def clean_embs(source_dir):
    for speaker in os.listdir(source_dir):
        for s in os.listdir(os.path.join(source_dir, speaker)):
            if s[-3:] == 'npy':
                os.remove(os.path.join(source_dir, speaker, s))


def test(data, save_mode):
    if data["encoder_ckpts"] is not None:
        for encoder in data["encoder_ckpts"]:
            encoder_name = Path(encoder).stem
            encoder_command(encoder, data)
            if data["hifi_gan"]['ckpts'] is not None:
                for hifi_model, hifi_config in zip(data["hifi_gan"]["ckpts"],
                                                   data["hifi_gan"]["configs"]):
                    hifi_name = Path(hifi_model).stem
                    run_command_on_list(data, 'grad_tts', output_prefix="%s_%s" % (encoder_name, hifi_name),
                                        hifi_model=hifi_model, hifi_config=hifi_config)
            elif data["grad_tts_ckpts"] is not None:
                run_command_on_list(data, 'grad_tts', output_prefix="%s" % encoder_name)
            if data['wavegrad_ckpts'] is not None:
                run_command_on_list(data, 'wavegrad', output_prefix="%s" % encoder_name)
            # clean_embs(data["data"])
    else:
        if data["hifi_gan"]['ckpts'] is not None:
            for hifi_model, hifi_config in zip(data["hifi_gan"]["ckpts"],
                                               data["hifi_gan"]["configs"]):
                hifi_name = Path(hifi_model).stem
                run_command_on_list(data, 'grad_tts', output_prefix="%s" % hifi_name,
                                    hifi_model=hifi_model, hifi_config=hifi_config)
        elif data["grad_tts_ckpts"] is not None:
            run_command_on_list(data, 'grad_tts', output_prefix='')
        if data['wavegrad_ckpts'] is not None:
            run_command_on_list(data, 'wavegrad', output_prefix='')
    if not save_mode:
        shutil.rmtree(data["data"])


if __name__ == '__main__':
    args = parse_args()
    config = args.config
    mode = args.mode
    save_mode = args.save_intermediate
    if mode == 'test':
        data = parse_config(config, parse_data)
        test(data, save_mode)
    else:
        data = parse_config(config, parse_data_val)
        test(data, save_mode)
