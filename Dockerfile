FROM ubuntu:22.04
USER root
COPY . /home/multi-level-rl-for-robotics
WORKDIR /home/multi-level-rl-for-robotics
RUN apt-get update
RUN apt-get install -y --no-install-recommends software-properties-common
RUN apt install -y python3-pip
RUN pip install -r requirements.txt &&\
    apt install python3.10-venv -y &&\
    pip install twine build &&\
    python3 -m build &&\
    twine check dist/* 

