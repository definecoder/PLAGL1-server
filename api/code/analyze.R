args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)


print(id)

# Ensure the number of samples match


# setwd(here("api", "code", id))

count_data <- readRDS("rds/count_data.rds")
sample_info <- readRDS("rds/sample_info.rds")

if (ncol(count_data) != nrow(sample_info)) {
    stop("Number of samples in count_data and sample_info do not match!")
}

# Convert the Col name of user to "Treatment"
colnames(sample_info)[colnames(sample_info) == colnames(sample_info)] <- "Treatment"

# Convert Condition to factor
sample_info$Treatment <- factor(sample_info$Treatment)

##############################################################################################################

########################################### Screen Data Quality | NODE 2 ##############################################

# Dimension Reduction
# Data Quality
# Remove Outlier Samples

##############################################################################################################

# Quality Control: detect outlier genes
library(WGCNA)
gsg <- goodSamplesGenes(t(count_data)) # check the data format
summary(gsg)

# Remove outlier genes
data <- count_data[gsg$goodGenes == TRUE, ]


################################################## Data Manipulation | NODE 2.1 | NODE 2.2 | NODE 2.3 (Yet to be included) #######################

# if there is any outlier samples exclude them from the both expression
# data and metadata

# samples_to_exclude <- c(1, 2, 5)
#
# count_data <- count_data[, -samples_to_exclude]
# sample_info <- sample_info [-samples_to_exclude,]

############################################################################################################

########################################### DGE Analysis ###################################################

########################################## Normalization | NODE 3 ###################################################

# Expression Data
# load the count data
# count_data <- read.csv("count_data.csv", header=TRUE,row.names = 1)

# Metadata
# load the sample info
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Ensure the number of samples match
# if (ncol(count_data) != nrow(sample_info)) {
#   stop("Number of samples in count_data and sample_info do not match!")
# }

# Convert Condition to factor
# sample_info$Treatment <- factor(sample_info$Treatment)
#
# Quality Control: detect outlier genes
# library(WGCNA)
#
# gsg <- goodSamplesGenes(t(count_data))
# summary(gsg)
#
# Remove outlier genes
# data <- count_data[gsg$goodGenes == TRUE,]

# Convert non-integer values to integers in count data
data <- round(data)
head(data)

# Create a new count data object
new_data <- as.matrix(data)
head(new_data)

# Display dimensions for verification
cat("Dimensions of data:", dim(data), "\n")
cat("Dimensions of new_data:", dim(new_data), "\n")
cat("Dimensions of sample_info:", dim(sample_info), "\n")

# Start Normalization

library(DESeq2)


# Generate the DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = new_data, colData = sample_info, design = ~Treatment)

# Set the factor levels for the Treatment column based on unique values
condition <- unique(sample_info$Treatment)
dds$Treatment <- factor(dds$Treatment, levels = condition)

# Input all factor from metadata # TO do

# Filter genes with low counts (less than 75% of sample number)
threshold <- round(dim(sample_info)[1] * 0.70)
keep <- rowSums(counts(dds)) >= threshold
dds <- dds[keep, ]

# Perform DESeq2 analysis
dds <- DESeq(dds)

# save the normalized counts
normalize_counts <- counts(dds, normalized = TRUE)
head(normalize_counts)
dim(normalize_counts)
write.csv(normalize_counts, "files/Normalized_Count_Data.csv")

count_data_norm <- normalize_counts

##################################################################################################################################################

################################################## BoxPlot | NODE 4 ######################################################################################

# Log2 transformation for count data
count_matrix <- counts(dds) + 1 # Adding 1 to avoid log(0)
log2_count_matrix <- log2(count_matrix)

png("figures/Boxplot_denorm.png")
boxplot(log2_count_matrix,
    outline = FALSE, main = "Boxplot of Log2-transformed Count Data",
    cex.main = 0.9, # Make title size smaller
    ylab = "Log2-transformed Counts",
    cex.axis = 0.7, las = 1, font.axis = 1, xaxt = "n"
) # Set font.axis to 1 for normal

# Rotate x-axis labels to 45 degrees
text(
    x = 1:ncol(log2_count_matrix),
    y = par("usr")[3] - 0.5, # Adjust y position as needed
    labels = colnames(log2_count_matrix),
    srt = 45, adj = 1, xpd = TRUE, cex = 0.7
) # Removed font for normal

dev.off()

pdf("figures/Boxplot_denorm.pdf")
boxplot(log2_count_matrix,
    outline = FALSE, main = "Boxplot of Log2-transformed Count Data",
    cex.main = 0.9, # Make title size smaller
    ylab = "Log2-transformed Counts",
    cex.axis = 0.7, las = 1, font.axis = 1, xaxt = "n"
) # Set font.axis to 1 for normal

# Rotate x-axis labels to 45 degrees
text(
    x = 1:ncol(log2_count_matrix),
    y = par("usr")[3] - 0.5, # Adjust y position as needed
    labels = colnames(log2_count_matrix),
    srt = 45, adj = 1, xpd = TRUE, cex = 0.7
) # Removed font for normal

dev.off()

# Log2 transformation for normalized count data
normalized_counts <- counts(dds, normalized = TRUE)


log2_normalized_counts <- log2(normalized_counts + 1) # Adding 1 to avoid log(0)


png("figures/Boxplot_norm.png")
boxplot(log2_normalized_counts,
    outline = FALSE,
    main = "Boxplot of Log2-transformed Normalized Count Data",
    cex.main = 0.9, # Make title size smaller
    ylab = "Log2-transformed Counts",
    cex.axis = 0.7, las = 1, font.axis = 1, xaxt = "n"
) # Set font.axis to 1 for normal

# Rotate x-axis labels to 45 degrees
text(
    x = 1:ncol(log2_normalized_counts),
    y = par("usr")[3] - 0.5, # Adjust y position as needed
    labels = colnames(log2_normalized_counts),
    srt = 45, adj = 1, xpd = TRUE, cex = 0.7
) # Removed font for normal
dev.off()


pdf("figures/Boxplot_norm.pdf")
boxplot(log2_normalized_counts,
    outline = FALSE,
    main = "Boxplot of Log2-transformed Normalized Count Data",
    cex.main = 0.9, # Make title size smaller
    ylab = "Log2-transformed Counts",
    cex.axis = 0.7, las = 1, font.axis = 1, xaxt = "n"
) # Set font.axis to 1 for normal

# Rotate x-axis labels to 45 degrees
text(
    x = 1:ncol(log2_normalized_counts),
    y = par("usr")[3] - 0.5, # Adjust y position as needed
    labels = colnames(log2_normalized_counts),
    srt = 45, adj = 1, xpd = TRUE, cex = 0.7
) # Removed font for normal
dev.off()


saveRDS(condition, "rds/condition.rds")
saveRDS(dds, "rds/dds.rds")



################################################# PCA ########################################################

# Load Expression Data
# data <- read.csv("count_data.csv")

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

#***** Make A Loop *****#

# Check for NA or Infinite values
summary(data)
is.na(data)
# Replace NA and Infinite values with zero

data[is.na(data)] <- 0

# Replacing Infinite Values
data_list <- as.list(data)

for (name in names(data_list)) {
    data_list[[name]][is.infinite(data_list[[name]])] <- 0
}

data <- as.data.frame(data_list)

# Verify no NA or Infinite values remain
summary(data)

#***** Make A Loop *****#

# Remove non-numeric columns for PCA
# Remove the gene ID column
data_numeric <- data[, sapply(data, is.numeric)]
# data_numeric <- data[,1:12]

# Perform PCA
pca <- prcomp(t(data_numeric))

# View the PCA results
summary(pca)

# Prepare PCA data for plotting
pca.dat <- as.data.frame(pca$x)
pca.var <- pca$sdev^2
pca.var.percent <- round(pca.var / sum(pca.var) * 100, digits = 2)

# Merge PCA data with metadata
pca.dat <- cbind(pca.dat, sample_info)

library(ggplot2)
# Plot PCA with metadata groups
plot <- ggplot(pca.dat, aes(PC1, PC2, color = Treatment)) +
    geom_point(size = 3) + # Adjust point size to match t-SNE
    geom_text(aes(label = rownames(pca.dat)), hjust = 0.5, vjust = 1.5, size = 3, show.legend = FALSE) + # Bold, smaller labels
    labs(
        x = paste0("PC1: ", pca.var.percent[1], " %"),
        y = paste0("PC2: ", pca.var.percent[2], " %")
    ) +
    theme_minimal() +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows
# save pca plot as a png file
ggsave("figures/PCA_denorm.png", plot)
ggsave("figures/PCA_denorm.pdf", plot)

################################################# UMAP #####################################################

# Load libraries
library(umap)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for Umap
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Set random seed for reproducibility
set.seed(123)

# Perform UMAP dimensionality reduction
umap_result <- umap(t(data_numeric), n_neighbors = 5, min_dist = 0.5)

# Extract UMAP coordinates and combine with metadata
umap_df <- data.frame(
    X1 = umap_result$layout[, 1], # UMAP component 1
    X2 = umap_result$layout[, 2], # UMAP component 2
    sample_info
)

library(ggplot2)
# Plot using ggplot2
plot <- ggplot(umap_df, aes(x = X1, y = X2, color = Treatment)) +
    geom_point(size = 3) + # Adjust point size
    geom_text(aes(label = rownames(sample_info)), hjust = .5, vjust = 1.5, size = 3, show.legend = FALSE) + # Adjust label size and position
    labs(
        title = "UMAP",
        x = "UMAP 1", y = "UMAP 2"
    ) +
    theme_minimal() +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows

ggsave("figures/UMAP_denorm.png", plot)
ggsave("figures/UMAP_denorm.pdf", plot)

################################################# t-SNE ###############################################

library(Rtsne)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for t-SNE
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

set.seed(123)
# Perform t-SNE
tsne_result <- Rtsne(t(data_numeric), dims = 2, perplexity = 3, verbose = TRUE, max_iter = 500)

# Create a data frame for plotting
tsne_data <- data.frame(
    X = tsne_result$Y[, 1],
    Y = tsne_result$Y[, 2],
    sample_info
)

# Plot the t-SNE results using ggplot2
plot <- ggplot(tsne_data, aes(x = X, y = Y, color = Treatment)) +
    geom_point(size = 3) +
    geom_text(aes(label = rownames(sample_info)), hjust = 0.5, vjust = 1.5, size = 3, show.legend = FALSE) + # Reduced text size
    theme_minimal() +
    ggtitle("t-SNE Plot") +
    xlab("t-SNE 1") +
    ylab("t-SNE 2") +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows


ggsave("figures/tSNE_denorm.png", plot)
ggsave("figures/tSNE_denorm.pdf", plot)


########################################### Phylogenetic Tree ###############################################


set.seed(123)
library(ape)


# Detect outlier samples using hierarchical clustering
htree <- hclust(dist(t(data_numeric)), method = "average")

phylo_tree <- as.phylo(htree)


# Assuming 'htree' and 'metadata' are correctly defined

# Convert labels to factors or unique numeric indices for colors
group_colors <- as.factor(sample_info$Treatment)
tip_colors <- as.numeric(group_colors)


# Plot the dendrogram with colored labels

pdf("figures/htree_denorm.pdf")
plot.phylo(phylo_tree,
    type = "phylogram",
    tip.color = tip_colors,
    cex = 0.8,
    main = "Hierarchical Clustering Dendrogram"
)

# Add a legend for the treatment groups
legend("topleft",
    legend = levels(group_colors),
    col = 1:length(levels(group_colors)),
    pch = 19,
    cex = 0.8,
    bty = "n"
)

dev.off()

png("figures/htree_denorm.png")
plot.phylo(phylo_tree,
    type = "phylogram",
    tip.color = tip_colors,
    cex = 0.8,
    main = "Hierarchical Clustering Dendrogram"
)

# Add a legend for the treatment groups
legend("topleft",
    legend = levels(group_colors),
    col = 1:length(levels(group_colors)),
    pch = 19,
    cex = 0.8,
    bty = "n"
)

dev.off()


# ---------------------------- Done till here ---------------------------- #
# ---------------------------- Again do the operations ---------------------------- #





count_data <- read.csv("files/Normalized_Count_Data.csv", header = TRUE, row.names = 1)


if (ncol(count_data) != nrow(sample_info)) {
    stop("Number of samples in count_data and sample_info do not match!")
}

# Convert the Col name of user to "Treatment"
colnames(sample_info)[colnames(sample_info) == colnames(sample_info)] <- "Treatment"

# Convert Condition to factor
sample_info$Treatment <- factor(sample_info$Treatment)

##############################################################################################################

########################################### Screen Data Quality | NODE 2 ##############################################

# Dimension Reduction
# Data Quality
# Remove Outlier Samples

##############################################################################################################

# Quality Control: detect outlier genes
library(WGCNA)
gsg <- goodSamplesGenes(t(count_data)) # check the data format
summary(gsg)

# Remove outlier genes
data <- count_data[gsg$goodGenes == TRUE, ]

################################################# PCA ########################################################

# Load Expression Data
# data <- read.csv("count_data.csv")

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

#***** Make A Loop *****#

# Check for NA or Infinite values
summary(data)
is.na(data)
# Replace NA and Infinite values with zero

data[is.na(data)] <- 0

# Replacing Infinite Values
data_list <- as.list(data)

for (name in names(data_list)) {
    data_list[[name]][is.infinite(data_list[[name]])] <- 0
}

data <- as.data.frame(data_list)

# Verify no NA or Infinite values remain
summary(data)

#** Make A Loop **#

# Remove non-numeric columns for PCA
# Remove the gene ID column
data_numeric <- data[, sapply(data, is.numeric)]

# data_numeric <- data[,1:12]

# Perform PCA
pca <- prcomp(t(data_numeric))

# View the PCA results
summary(pca)

# Prepare PCA data for plotting
pca.dat <- as.data.frame(pca$x)
pca.var <- pca$sdev^2
pca.var.percent <- round(pca.var / sum(pca.var) * 100, digits = 2)

# Merge PCA data with metadata
pca.dat <- cbind(pca.dat, sample_info)

library(ggplot2)

plot <- ggplot(pca.dat, aes(PC1, PC2, color = Treatment)) +
    geom_point(size = 3) + # Adjust point size to match t-SNE
    geom_text(aes(label = rownames(pca.dat)), hjust = 0.5, vjust = 1.5, size = 3, show.legend = FALSE) + # Bold, smaller labels
    labs(
        x = paste0("PC1: ", pca.var.percent[1], " %"),
        y = paste0("PC2: ", pca.var.percent[2], " %")
    ) +
    theme_minimal() +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows
# save pca plot as a png file
ggsave("figures/PCA_norm.png", plot)
ggsave("figures/PCA_norm.pdf", plot)

################################################# UMAP #####################################################

# Load libraries
library(umap)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for Umap
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

# Set random seed for reproducibility
set.seed(123)

# Perform UMAP dimensionality reduction
umap_result <- umap(t(data_numeric), n_neighbors = 5, min_dist = 0.5)

# Extract UMAP coordinates and combine with metadata
umap_df <- data.frame(
    X1 = umap_result$layout[, 1], # UMAP component 1
    X2 = umap_result$layout[, 2], # UMAP component 2
    sample_info
)

library(ggplot2)
# Plot using ggplot2
plot <- ggplot(umap_df, aes(x = X1, y = X2, color = Treatment)) +
    geom_point(size = 3) + # Adjust point size
    geom_text(aes(label = rownames(sample_info)), hjust = .5, vjust = 1.5, size = 3, show.legend = FALSE) + # Adjust label size and position
    labs(
        title = "UMAP",
        x = "UMAP 1", y = "UMAP 2"
    ) +
    theme_minimal() +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows

ggsave("figures/UMAP_norm.png", plot)
ggsave("figures/UMAP_norm.pdf", plot)

################################################# t-SNE ###############################################

library(Rtsne)

# Load Data
# data <- read.csv("count_data.csv")
# Remove non-numeric columns for t-SNE
# Remove the gene ID column
# data_numeric <- data[, sapply(data, is.numeric)]

# load the metadata
# sample_info <- read.csv("meta_data.csv", header =TRUE,row.names = 1)

set.seed(123)
# Perform t-SNE
tsne_result <- Rtsne(t(data_numeric), dims = 2, perplexity = 3, verbose = TRUE, max_iter = 500)

# Create a data frame for plotting
tsne_data <- data.frame(
    X = tsne_result$Y[, 1],
    Y = tsne_result$Y[, 2],
    sample_info
)

# Plot the t-SNE results using ggplot2
plot <- ggplot(tsne_data, aes(x = X, y = Y, color = Treatment)) +
    geom_point(size = 3) +
    geom_text(aes(label = rownames(sample_info)), hjust = 0.5, vjust = 1.5, size = 3, show.legend = FALSE) + # Reduced text size
    theme_minimal() +
    ggtitle("t-SNE Plot") +
    xlab("t-SNE 1") +
    ylab("t-SNE 2") +

    # Adjust the legend
    theme(
        legend.position = "bottom", # Move the legend to the bottom
        legend.title = element_text(size = 10), # Legend title size
        legend.text = element_text(size = 8), # Legend text size
        legend.key.size = unit(0.5, "cm")
    ) + # Reduce legend key size
    guides(color = guide_legend(nrow = 2, byrow = TRUE)) # Organize legend into 2 rows


ggsave("figures/tSNE_norm.png", plot)
ggsave("figures/tSNE_norm.pdf", plot)


########################################### Phylogenetic Tree ###############################################


set.seed(123)
library(ape)


# Detect outlier samples using hierarchical clustering
htree <- hclust(dist(t(data_numeric)), method = "average")

phylo_tree <- as.phylo(htree)


# Assuming 'htree' and 'metadata' are correctly defined

# Convert labels to factors or unique numeric indices for colors
group_colors <- as.factor(sample_info$Treatment)
tip_colors <- as.numeric(group_colors)


# Plot the dendrogram with colored labels

pdf("figures/htree_norm.pdf")
plot.phylo(phylo_tree,
    type = "phylogram",
    tip.color = tip_colors,
    cex = 0.8,
    main = "Hierarchical Clustering Dendrogram"
)

# Add a legend for the treatment groups
legend("topleft",
    legend = levels(group_colors),
    col = 1:length(levels(group_colors)),
    pch = 19,
    cex = 0.8,
    bty = "n"
)

dev.off()

png("figures/htree_norm.png")
plot.phylo(phylo_tree,
    type = "phylogram",
    tip.color = tip_colors,
    cex = 0.8,
    main = "Hierarchical Clustering Dendrogram"
)

# Add a legend for the treatment groups
legend("topleft",
    legend = levels(group_colors),
    col = 1:length(levels(group_colors)),
    pch = 19,
    cex = 0.8,
    bty = "n"
)

dev.off()
