FROM python:3.11-slim AS builder


# Install system dependencies
RUN apt-get update -y && apt-get install -y \
    r-base \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Bioconductor version based on R version
RUN R -e "install.packages('BiocManager', repos='http://cran.us.r-project.org'); \
    bioc_version <- ifelse(getRversion() >= '4.3', '3.18', '3.16'); \
    BiocManager::install(version = bioc_version); \
    BiocManager::install(c('WGCNA', 'DESeq2', 'limma', 'biomaRt', 'sva', 'STRINGdb')); \
    install.packages(c('tidyverse', 'Rtsne', 'umap', 'ggplot2', \
    'readr', 'ape', 'mice', 'dplyr', 'gplots', \
    'ggVennDiagram', 'pheatmap', 'RColorBrewer', \
    'stringr'), repos='http://cran.us.r-project.org', dependencies=TRUE);"

# Set working directory
WORKDIR /code

# Copy requirements and install Python dependencies
COPY ./api/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application code
COPY ./api /code/api

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
