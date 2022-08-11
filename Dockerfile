FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04


RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    graphviz \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo 

RUN pip3 install -U pip wheel setuptools japanize-matplotlib
RUN pip3 install jupyter click numpy matplotlib seaborn pandas tqdm
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install visdom jupyter torchviz
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
