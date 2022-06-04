from argparse import ArgumentParser, Namespace
import librosa
# from waitress import serve
from flask import Flask, request, Response
import soundfile as sf
import shutil
from flask_api import status
import logging
import io
import json
import os
import yaml
import tempfile
from time import gmtime, strftime
import torch
from requests_toolbelt import MultipartEncoder
from flask_server.inference_pipeline import InferenceEngine
from flask_server.model_serving import FunctionServingWrapper
import time
import noisereduce as nr


OUT_SAMPLE_RATE = 22050

with open(
        os.path.join(os.path.dirname(__file__), 'flask_server/server_config.yaml'),
        'r'
) as config_file:
    server_configuration_dict = yaml.safe_load(config_file)

app = Flask(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
app_log = logging.getLogger('ServerGlobalLog')
SYNTHESIZER_MODEL = server_configuration_dict['Global_parameters']['used_model']
synthesis_pipeline = FunctionServingWrapper(
    InferenceEngine(server_configuration_dict),
    1
)


@app.route('/api/inference', methods=['POST'])
def tts_inference_zip():
    data = json.load(request.files['text'])
    texts_dict = data['texts']
    audio_format = data['format']
    input_extension = request.files['voice'].filename.split('.')[-1]
    output_data_format = data['out']

    if input_extension != 'wav':
        logging.info(f' Input extension: ".{input_extension}" is not supported. Supported format: ".wav"')
        return f'Input extension: ".{input_extension}" is not supported. Supported format: ".wav"', \
               status.HTTP_400_BAD_REQUEST

    fb = io.BytesIO(request.files['voice'].read())
    voice_wav, sampling_rate = librosa.load(fb, sr=None)
    duration = librosa.get_duration(voice_wav, sampling_rate)
    available_formats = ['wav', 'flac']

    if audio_format not in available_formats:
        logging.info(f' Output format: ".{audio_format}" is not supported. Default: ".wav"')
        audio_format = 'wav'

    if duration < 3.5 or duration > 10.:
        logging.info(f" Record duration: {duration} sec\nThe duration must be more than 4 seconds and less than 10.")
        return f"Record duration: {duration} sec\nThe duration must be more than 4 seconds and less than 10.",\
               status.HTTP_400_BAD_REQUEST

    logging.info(' Texts to generating: \'{}\''.format(texts_dict))
    logging.info(' Source sampling rate: \'{}\''.format(sampling_rate))

    file_ordering_names = list(texts_dict.keys())
    texts_list = [
        texts_dict[record_file_name]
        for record_file_name in file_ordering_names
    ]

    start_time = time.time()
    records = synthesis_pipeline(texts_list, voice_wav, sampling_rate)
    logging.info(" ---Inference: %s seconds ---" % (time.time() - start_time))

    if output_data_format == 'zip':
        temp_dir = tempfile.mkdtemp()
        for i, wav_record in enumerate(records):
            record_file_name = '{}.{}'.format(file_ordering_names[i], audio_format)
            sf.write(os.path.join(temp_dir, record_file_name), wav_record, OUT_SAMPLE_RATE)
            noise_data, rate = sf.read(os.path.join(temp_dir, record_file_name))
            reduced_noise = nr.reduce_noise(y=noise_data, sr=rate,
                                            n_std_thresh_stationary=0.35,
                                            stationary=True)
            sf.write(os.path.join(temp_dir, record_file_name), reduced_noise, OUT_SAMPLE_RATE)


        _, tmp_file_path = tempfile.mkstemp()
        shutil.make_archive(os.path.join(tmp_file_path), 'zip', temp_dir)
        shutil.rmtree(temp_dir)
        logging.info('Done')

        with open(tmp_file_path + '.zip', 'rb') as f:
            data = f.readlines()

        os.remove(tmp_file_path + '.zip')
        return Response(data, headers={
            'Content-Type': 'application/zip',
            'Content-Disposition': 'attachment; filename=records.zip'
        })

    elif output_data_format == 'multipart':
        tmp_files = []
        response_fields = dict()
        for i, wav_record in enumerate(records):
            record_file_name = '{}.{}'.format(file_ordering_names[i], audio_format)

            fb, tmp_file_path = tempfile.mkstemp(suffix=f'.{audio_format}')
            # write(tmp_file_path, OUT_SAMPLE_RATE, wav_record)
            sf.write(tmp_file_path, wav_record, OUT_SAMPLE_RATE)
            noise_data, rate = sf.read(tmp_file_path)
            reduced_noise = nr.reduce_noise(y=noise_data, sr=rate,
                                            n_std_thresh_stationary=0.35,
                                            stationary=True)
            sf.write(tmp_file_path, reduced_noise, OUT_SAMPLE_RATE)

            fd = open(tmp_file_path, 'rb')

            response_fields[record_file_name] = (
                record_file_name,
                io.BytesIO(fd.read()),
                f'sound/{audio_format}'
            )
            fd.close()
            os.close(fb)

            tmp_files.append(tmp_file_path)

        for file in tmp_files:
            os.remove(file)

        multipart_encoded_records = MultipartEncoder(fields=response_fields)

        logging.info('Done')

        return Response(
            multipart_encoded_records.to_string(),
            mimetype=multipart_encoded_records.content_type
        )


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Voice cloning server')
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=5000)
    parser.add_argument(
        '--logfile', required=False, type=str, default='./log.txt',
        help='Path to log file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(filename=args.logfile, level=logging.INFO)
    app_log.info(
        '\n\n'
        'SERVER START TIME: {}'
        '\n'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    )
    app_log.info(' Inference device: {}'.format(DEVICE))
    app_log.info(f' Synthesizer model: {SYNTHESIZER_MODEL}')
    print(f'Current device: {DEVICE}')
    # serve(app, host=args.ip, port=args.port)
    app.run(host=args.ip, debug=False, port=args.port)

