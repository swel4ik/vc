librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test":
        ["LibriSpeech/test-clean", 'LibriSpeech/test-other']
    ,
    "dev":
        ["LibriSpeech/dev-clean", 'LibriSpeech/dev-other']
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
