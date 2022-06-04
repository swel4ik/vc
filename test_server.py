import json
from argparse import ArgumentParser, Namespace
import io
import requests
from requests_toolbelt.multipart import decoder
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os
import zipfile
import time


def parse_args() -> Namespace:
    parser = ArgumentParser('Test server TTS feature')
    parser.add_argument('--ip', required=False, type=str, default='localhost')
    parser.add_argument('--port', required=False, type=int, default=5000)
    parser.add_argument(
        '-t', '--text', required=True, type=str,
        help='Path to JSON file'
    )
    parser.add_argument(
        '-v', '--voice', required=True, type=str,
        help='Path to voice wav file'
    )
    parser.add_argument(
        '-o', '--output', required=True, type=str,
        help='Path to output folder'
    )
    parser.add_argument(
        '-f', '--format', required=False, type=str, default='wav',
        help='Extension for output audio files ("wav" or "flac")'
    )

    parser.add_argument(
        '-z', '--out_structure', required=False, type=str, default='zip',
        help='The structure how to process the output response ("zip" or "multipart")'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    audio_format = args.format
    mp_encoder = MultipartEncoder(
        fields={
            'text': (
                'text.json',
                open(args.text, 'rb'),
                'application/json'
            ),
            'voice': (
                'voice.wav',
                open(args.voice, 'rb'),
                'sound/wav'
            )
        }
    )

    print('http://{}:{}/api/inference'.format(args.ip, args.port))
    response = requests.post(
        url=f'http://{args.ip}:{args.port}/api/inference',
        data=mp_encoder,
        headers={'Content-Type': mp_encoder.content_type}
    )
    print('Status: {}'.format(response.status_code))
    if args.out_structure == 'zip':
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(args.output)
    else:
        multipart_data = decoder.MultipartDecoder.from_response(response)

        for part in multipart_data.parts:
            content_type = list(part.headers.items())[1][1].decode()
            file_name = \
                list(part.headers.items())[0][1].decode().split('filename=\"')[
                    1].split("\"")[0]

            if content_type == f'sound/{audio_format}':
                with open(os.path.join(args.output, file_name), 'wb') as f:
                    f.write(part.content)
