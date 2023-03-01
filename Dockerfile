FROM docker.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED="1"

ENV WEBUI_REPO_URL="https://github.com/AUTOMATIC1111/stable-diffusion-webui.git"
ENV PYTHON_VERSION="3.10"

ENV WORKDIR="/workspace"
ENV VIRTUAL_ENV="${WORKDIR}/venv"

RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y \
  git \
  wget \
  libgl1 \
  python3-pip \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-venv \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth=1 ${WEBUI_REPO_URL} ${WORKDIR}
WORKDIR ${WORKDIR}

# install virtual env
RUN python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN wget -O ./models/Stable-diffusion/model.safetensors https://civitai.com/api/download/models/15959

# install dependencies
RUN pip install -U xformers
RUN pip install runpod
RUN python launch.py --exit --no-download-sd-model --skip-torch-cuda-test --ckpt ./models/Stable-diffusion/model.safetensors

COPY handler.py ./

ENTRYPOINT [ "python", "handler.py" ]