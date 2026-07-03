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
    libwebp-dev \
    libfontconfig1-dev \
    libuv1-dev \
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
# Install BiocManager first
RUN R -e "install.packages('BiocManager', repos='https://cran.r-project.org')"

# Set Bioconductor version to 3.21 - this will manage compatible package versions
RUN R -e "BiocManager::install(version = '3.23', ask = FALSE, update = TRUE)"

# Install Bioconductor packages - BiocManager ensures version compatibility within Bioc 3.21
RUN R -e "BiocManager::install(c('WGCNA', 'DESeq2', 'limma', 'biomaRt', 'sva', 'STRINGdb', 'apeglm', 'impute'), ask = FALSE, update = FALSE)"

# Install CRAN packages - let BiocManager handle compatibility with Bioconductor
RUN R -e "BiocManager::install(c('tidyverse', 'Rtsne', 'umap', 'ggplot2', 'readr', 'ape', 'mice', 'dplyr', 'gplots', 'ggVennDiagram', 'pheatmap', 'RColorBrewer', 'stringr', 'here', 'lme4'), ask = FALSE)"

# WGCNA is used at runtime by analyze.R and needs an explicit install check.
RUN R -e "install.packages('WGCNA', repos='https://cran.r-project.org', dependencies = TRUE)"
RUN R -e "stopifnot(requireNamespace('WGCNA', quietly = TRUE))"



# ---------------- FastAPI Setup ----------------
WORKDIR /code

COPY ./api/requirements.txt /code/requirements.txt
# RUN pip install --no-cache-dir -r /code/requirements.txt
RUN pip install -r /code/requirements.txt
RUN pip install gunicorn uvicorn

COPY ./api /code/api
WORKDIR /code/api

EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:8000", "--timeout", "6000", "--worker-connections", "30"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker rm -f plgl-server && docker build -t wonderful_rubin . && docker run --name plgl-server -p 8000:8000 -d wonderful_rubin:latest
