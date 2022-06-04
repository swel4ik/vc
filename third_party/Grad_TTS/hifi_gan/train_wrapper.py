import argparse
import subprocess
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Function to setup multiple experiments')

    parser.add_argument('--config', required=False, help='Path to config')

    return parser.parse_args()


def parse_config(config):
    with open(config, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

if __name__ == '__main__':
    config = parse_config(parse_args().config)
    for conf in config['config']:
        run_command = 'python train.py ' \
                      '--input_wavs_dir {} --input_mels_dir {} --input_embeddings_dir {} ' \
                      '--input_training_file {} --input_validation_file {} --checkpoint_path {} ' \
                      '--config {} --training_epochs {} --training_steps {} ' \
                      '--stdout_interval {} --checkpoint_interval {} --summary_interval {} ' \
                      '--validation_interval {}'
        run_command = run_command.format(
            config['input_wavs_dir'],
            config['input_mels_dir'],
            config['input_embeddings_dir'],
            config['input_training_file'],
            config['input_validation_file'],
            config['checkpoint_path'],
            conf,
            config['training_epochs'],
            config['training_steps'],
            config['stdout_interval'],
            config['checkpoint_interval'],
            config['summary_interval'],
            config['validation_interval'],
        )
        if config['fine_tuning']:
            run_command += ' --fine_tuning True'
        print(run_command)
        with subprocess.Popen(run_command, shell=True, stdout=subprocess.PIPE) as process:
            for line in process.stdout:
                print(line)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode,
                                                process.args)
