FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /app

RUN apt-get update
RUN apt-get install -y libgeos-dev libgl1-mesa-glx

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . ./
