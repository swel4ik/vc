# Ardvrk Voice Cloning
## Installation
#### conda
```pre
conda create --name VoiceCloning python=3.8
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install --file requirements.txt
conda activate VoiceCloning
```
## Docker
If you need to use GPU, follow the next [instruction](https://github.com/NVIDIA/nvidia-docker) to prepare your host machine.

### 1. Choose CPU/GPU Dockerfile
`cp ./docker_cpu/Dockerfile Dockerfile` - CPU  
`cp ./docker_gpu/Dockerfile Dockerfile` - GPU  
### 2. Edit server config file
CPU: in `flask_server/server_config.yaml` file choose `inference_device: 'cpu'`  
GPU: in `flask_server/server_config.yaml` file choose `inference_device: 'cuda'`    
### 3. Build docker image
`cd ardvrk_voice_cloning`  
`docker build -t ardvrk_voice_cloning .`
### 4. Run docker server
CPU: `docker run -p 5000:5000 --name VoiceCloning -d ardvrk_voice_cloning`  
GPU: `docker run -d --gpus all -p 5000:5000 --name VoiceCloning ardvrk_voice_cloning`  
### Optional
If you need to get server logs use the following command: `docker exec -i -t VoiceCloning cat log.txt`
### 5. Stop and delete docker container
`docker stop VoiceCloning`  
`docker rm VoiceCloning`



## Project structure
## Overview
```pre
.
├── dataset_utils
│   ├── common_voice
│   │   ├── ...
│   ├── denoising
│   │   └── NSNet2
│   │       ├──...
│   └── libritts
│       ├── ...
├── flask_server
│   ├── ...
├── README.md
├── requirements.txt
├── scripts
│   ├── ...
│   └── test
│       ├──...
└── third_party
    ├── Grad_TTS
    │   ├──...
    │   ├── hifi_gan
    │   │   ├── ...
    ├── RTVC
    │   ├── ...
    └── wavegrad2
        ├── ... 
```

## dataset_utils
Scripts used for dataset preprocessing, including common_voice,
libritts, and NSNet2 for denoising data
#### [common voice](./datset_utils/common_voice/README.md)
#### [denoising](./dataset_utils/denoising/NSNet2/README.md)
#### [libritts](./dataset_utils/libritts/README.md)

## server
Contain [server](./flask_server/README.md) backend
### Run
```
python server.py --ip <ip_address> --port <port> --logfile <path to log file>
```
### Test
```
python test_server.py --ip <ip_address> --port <port> --text <json file with text to be generated> --voice <voice .wav file> --output <path to output folder>
```
## scripts
[Scripts](./scripts/README.md) for testing and validation
## third_party

### [Grad TTS](./third_party/Grad_TTS/README.md)
#### Inference

### [Real-Time-Voice-Cloning-master](./third_party/Real-Time-Voice-Cloning-master/README.md)
#### Inference

### [wavegrad2](./third_party/wavegrad2/README.md)
####Inference

## license
###Network architectures:
#### HiFi-GAN: MIT
#### Grad_TTS: MIT
#### RTVC: MIT
#### Wavegrad2: BSD 3
###Datasets:
#### COMMON-VOICE: CC-0
#### LibriTTS: CC 4.0
#### VCTK: CC 4.0
#### LibriSpeech CC 4.0
#### VoxCeleb CC 4.0