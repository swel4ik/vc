import json
import subprocess


def get_commit_hash():
    message = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return message.strip().decode('utf-8')


def get_speakers(json_path: str) -> list:
    with open(json_path, 'r') as jf:
        speakers_dict = json.load(jf)

    return speakers_dict


def get_speakers_names(json_path: str) -> list:
    with open(json_path, 'r') as jf:
        speakers_dict = json.load(jf)

    return list(speakers_dict.keys())
