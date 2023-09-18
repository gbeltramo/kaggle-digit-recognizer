FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS ka_digrec_build

# Update `apt` and install `curl`
RUN apt-get update && \
    apt-get --only-upgrade install -y apt && \
    apt-get install -y --no-install-recommends curl

# Install `conda`/`mamba`
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" && \
    bash Mambaforge-$(uname)-$(uname -m).sh -b -p "/mambaforge"

# Upgrade `mamba` in `base` env
RUN /mambaforge/bin/mamba update -y mamba

# Create `ka-pytorch` environment
RUN /mambaforge/bin/mamba create -c conda-forge -y \
    --name ka-pytorch \
    python=3.11 \
    numpy black pytest

# Make RUN commands use the new environment:
SHELL ["/mambaforge/bin/conda", "run", "-n", "ka-pytorch", "/bin/bash", "-c"]

RUN mamba install -y pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia

# Install conda-pack to shrink the environment
RUN mamba install -c conda-forge -y conda-pack

# Create standalone env
RUN conda-pack -n ka-pytorch -o /tmp/env.tar && \
    mkdir /venv && \
    cd /venv && \
    tar xf /tmp/env.tar && \
    rm /tmp/env.tar

RUN /venv/bin/conda-unpack
