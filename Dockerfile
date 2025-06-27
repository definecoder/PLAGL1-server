FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    gnupg \
    build-essential \
    ca-certificates \
    lsb-release \
    libssl-dev \
    libcurl4-openssl-dev \
    libreadline-dev \
    libbz2-dev \
    libsqlite3-dev \
    libffi-dev \
    zlib1g-dev \
    liblzma-dev \
    libncurses5-dev \
    libxml2-dev \
    locales \
    cmake \
    libgit2-dev \
    libglpk-dev \
    libgmp-dev \
    libgsl-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfontconfig1-dev \
    pkg-config \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# ---------------- Install Python 3.11 ----------------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 -m pip install --upgrade pip

# Add CRAN GPG key and repository
RUN curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor -o /usr/share/keyrings/cran.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/cran.list

# Install latest R (e.g., 4.5.0)
RUN apt-get update && \
    apt-get install -y r-base && \
    apt-mark hold r-base

# ---------------- Install R packages ----------------
RUN R -e "install.packages('lme4', dependencies=TRUE)"
RUN R -e "install.packages(c('ggplot2','tibble','tidyr','readr','purrr','dplyr','stringr','forcats','httr','gargle','googledrive','googlesheets4','rvest','covr'), repos='https://cran.r-project.org', dependencies=TRUE)"

RUN R -e "install.packages('remotes', repos='https://cran.r-project.org'); \
    remotes::install_version('BiocManager', version = '1.30.25', repos = 'https://cran.r-project.org'); \
    BiocManager::install(version = '3.21'); \
    BiocManager::install(c('WGCNA', 'DESeq2', 'limma', 'biomaRt', 'sva', 'STRINGdb')); \
    install.packages(c('tidyverse', 'Rtsne', 'umap', 'ggplot2', \
        'readr', 'ape', 'mice', 'dplyr', 'gplots', \
        'ggVennDiagram', 'pheatmap', 'RColorBrewer', \
        'stringr'), repos='https://cran.r-project.org', dependencies=TRUE);"



# ---------------- FastAPI Setup ----------------
WORKDIR /code

COPY ./api/requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir -r /code/requirements.txt
RUN pip install -r /code/requirements.txt

COPY ./api /code/api
WORKDIR /code/api

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker rm -f plgl-server && docker build -t wonderful_rubin . && docker run --name plgl-server -p 8000:8000 -d wonderful_rubin:latest
