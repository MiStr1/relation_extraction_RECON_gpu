FROM nvidia/cuda:10.1-base-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update \
  && apt install python3.8 -y \
  && apt install python3-distutils -y \
  && apt install python3-apt -y \
  && apt install libpython3.8-dev -y\
  && apt-get install wget -y \
  && wget https://bootstrap.pypa.io/get-pip.py \
  && python3.8 get-pip.py \
  && apt-get remove wget -y \
  && rm get-pip.py \
  && apt install build-essential -y \
  && apt-get install python3.8-venv -y


RUN python3.8 -m venv /opt/venv1 \
	&& python3.8 -m venv /opt/venv2 \
	&& mkdir RECON \
	&& mkdir CLASSLA

COPY requirements.txt RECON/
COPY classla_service/requirements.txt classla_service/get_classla_models.py CLASSLA/

RUN . /opt/venv1/bin/activate \
	&& python -m pip install --upgrade pip \
	&& python -m pip install --no-cache-dir --upgrade Cython \
	&& python -m pip --no-cache-dir install -r RECON/requirements.txt -f https://download.pytorch.org/whl/cu101/torch_stable.html \
	&& deactivate \
	&& . /opt/venv2/bin/activate \
	&& python -m pip install --upgrade pip \
	&& python -m pip install --no-cache-dir --upgrade Cython \
    && python -m pip --no-cache-dir install -r CLASSLA/requirements.txt -f https://download.pytorch.org/whl/cu101/torch_stable.html \
    && python CLASSLA/get_classla_models.py \
	&& deactivate 

COPY relation_extraction RECON/
COPY classla_service/wiki.sl.small classla_service/main.py classla_service/mark_entities.py CLASSLA/
COPY run.sh run_CLASSLA.sh ./


CMD ["sh", "run.sh"]
