if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos = "http://cran.us.r-project.org")
}

if (!requireNamespace("apeglm", quietly = TRUE)) {
    BiocManager::install("apeglm", ask = FALSE)
}

if (!requireNamespace("impute", quietly = TRUE)) {
    BiocManager::install("impute", ask = FALSE)
}

if (!requireNamespace("DESeq2", quietly = TRUE)) {
    BiocManager::install("DESeq2", ask = FALSE)
    library(DESeq2)
}

if (!requireNamespace("Bioconductor", quietly = TRUE)) {
    install.packages("Bioconductor", repos = "http://cran.us.r-project.org")
}

if (!requireNamespace("WGCNA", quietly = TRUE)) {
    BiocManager::install("WGCNA", ask = FALSE)
    library(WGCNA)
}
