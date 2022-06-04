import os
import shutil
import yaml


def parse_data(data):
    if isinstance(data["data"], list):
        os.makedirs(os.path.join('temp_dir', 'temp_speaker'), exist_ok=True)
        for f in data["data"]:
            shutil.copyfile(f, os.path.join('temp_dir', 'temp_speaker', os.path.basename(f)))
    else:
        for speaker in os.listdir(data["data"]):
            os.makedirs(os.path.join('temp_dir', speaker), exist_ok=True)
            for s in os.listdir(os.path.join(data["data"], speaker)):
                if s[-3:] == 'wav' or data["encoder_ckpts"] is None:
                    shutil.copyfile(os.path.join(data["data"], speaker, s), os.path.join("temp_dir", speaker, s))
    data["data"] = os.path.abspath("temp_dir")
    os.makedirs(data["output_path"], exist_ok=True)


def parse_data_val(data):
    if 'val_data' in data:
        return 
    for speaker in os.listdir(data["data"]):
        os.makedirs(os.path.join('temp_dir', speaker), exist_ok=True)
        os.makedirs(os.path.join('val_temp_dir', speaker), exist_ok=True)
        for s in os.listdir(os.path.join(data["data"], speaker)):
            basename, ext = os.path.splitext(s)
            if ext == '.txt':
                shutil.copyfile(os.path.join(data["data"], speaker, s), os.path.join("val_temp_dir", speaker, s))
                shutil.copyfile(os.path.join(data["data"], speaker, basename + '.wav'),
                                os.path.join("val_temp_dir", speaker, basename + '.wav'))
        for s in os.listdir(os.path.join(data["data"], speaker)):
            basename, ext = os.path.splitext(s)
            if not os.path.isfile(os.path.join(data['data'], speaker, basename + '.txt')):
                shutil.copyfile(os.path.join(data["data"], speaker, s), os.path.join("temp_dir", speaker, s))

    data["data"] = os.path.abspath("temp_dir")
    data["val_data"] = os.path.abspath("val_temp_dir")
    os.makedirs(data["output_path"], exist_ok=True)


def parse_config(config, parse_data_func):
    with open(config, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if isinstance(data["wavegrad_ckpts"], str):
        data["wavegrad_ckpts"] = [data["wavegrad_ckpts"]]
    if isinstance(data["grad_tts_ckpts"], str):
        data["grad_tts_ckpts"] = [data["grad_tts_ckpts"]]
    if isinstance(data["hifi_gan"]["ckpts"], str):
        data["hifi_gan"]["ckpts"] = [data["hifi_gan"]["ckpts"]]
    if isinstance(data["hifi_gan"]["configs"], str):
        data["hifi_gan"]["configs"] = [data["hifi_gan"]["configs"]]
    if isinstance(data["encoder_ckpts"], str):
        data["encoder_ckpts"] = [data["encoder_ckpts"]]
    if isinstance(data["data"], str) and os.path.isfile(data["data"]):
        data["data"] = [data["data"]]
    os.makedirs('temp_dir', exist_ok=True)
    parse_data_func(data)
    return data
