FROM ubuntu:latest

# Install general dependencies

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y tzdata
RUN apt-get install -y screen curl unzip cmake git ffmpeg
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

# Project dependencies
ADD . /ardvrk_voice_clonning/
RUN pip3 install --default-timeout=100 torch==1.9.0
RUN pip3 install --default-timeout=100 torchvision==0.10.0
RUN pip3 install torchaudio==0.9.0

WORKDIR /ardvrk_voice_clonning/
RUN pip3 install --default-timeout=100 -r requirements.txt

WORKDIR /ardvrk_voice_clonning/third_party/Grad_TTS/model/monotonic_align
RUN python3 setup.py build_ext --inplace

WORKDIR /ardvrk_voice_clonning/

CMD python3 server.py



