FROM nvcr.io/nvidia/pytorch:20.06-py3

ENV DEBIAN_FRONTEND noninteractive 

# 方式1(GPGが古いパッケージリポジトリ登録情報を除外してwgetインストール)
# RUN rm -f /etc/apt/sources.list.d/cuda.list \
#     && apt-get update && apt-get install -y --no-install-recommends \
#         wget \
#     && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all.deb \
#     && rm -f cuda-keyring_1.0-1_all.deb


RUN apt-get update && apt-get install -y tzdata

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    python3 \
    python3-pip \
    graphviz \
    opencv-data \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools

RUN python3 -m pip install numpy
RUN python3 -m pip install opencv-python
# RUN pip3 install -U pip wheel setuptools japanize-matplotlib
RUN pip3 install jupyter click numpy matplotlib seaborn pandas tqdm
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install visdom jupyter torchviz torchinfo japanize_matplotlib
ENV USER_NAME=user
ENV USER_UID=1000
ARG wkdir=/home/${USER_NAME}/src

RUN echo "root:root" | chpasswd \
    && useradd -m -u ${USER_UID} --groups sudo,video ${USER_NAME} \
    && echo "${USER_NAME}:${USER_NAME}" | chpasswd \
    && echo "%${USER_NAME}    ALL=(ALL)    NOPASSWD:    ALL" >> /etc/sudoers.d/${USER_NAME} \
    && chmod 0440 /etc/sudoers.d/${USER_NAME}

USER ${USER_NAME}
WORKDIR ${wkdir}


COPY ./src/ .
RUN sudo chown -hR ${USER_NAME}:${USER_NAME} ${wkdir}
