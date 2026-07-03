# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript script.R species_name gene_symbols_json clustering_method output_dir")
}

species_name <- args[1]
gene_symbols_json <- args[2]
clustering_method <- args[3]
output_dir <- args[4]
output_dir <- normalizePath(output_dir, winslash = "/", mustWork = FALSE)

# Load necessary package for JSON parsing
library(jsonlite)
gene_symbols <- fromJSON(gene_symbols_json)

script_path_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_path_arg[1]), winslash = "/", mustWork = TRUE)
script_dir <- dirname(script_path)

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Source all required R scripts
source(file.path(script_dir, "1_setup_string_env.R"))
source(file.path(script_dir, "2_map_n_plot_input_genes.R"))
source(file.path(script_dir, "3_single_gene_analysis.R"))
source(file.path(script_dir, "4_enrichment_analysis_n_plot_alt.R"))
source(file.path(script_dir, "5_query_genes_analysis.R"))
source(file.path(script_dir, "6_cluster_finder.R"))
source(file.path(script_dir, "7_single_gene_analysis_compiled.R"))
source(file.path(script_dir, "8_query_gene_network_analysis.R"))
source(file.path(script_dir, "9_query_gene_enrichment_analysis.R"))
source(file.path(script_dir, "11_input_genes_enrichment_analysis.R"))

################# Workflow #####################################
setup_stringdb_environment()

# Load organism list and initialize STRING database
organism_List <- read.csv(file.path(script_dir, "Organism_List.csv"))

taxon_id <- get_taxon_id(organism_List, species_name)
string_db <- initialize_stringdb(taxon_id)

# Input gene list provided by the user
# species_name and gene_symbols come from the command-line arguments
# Example: species_name <- 'Homo sapiens'
#          gene_symbols <- c("ZNF212", "ZNF451", "PLAGL1", "NFAT5", "ICAM5", "RRAD")

# Map genes to STRING IDs
mapped_genes <- map_genes_to_string_ids(string_db, gene_symbols)
write.csv(mapped_genes, file.path(output_dir, "string_ids_according_to_initial_input_genes.csv"))

mapped_gene_str_IDs <- mapped_genes$STRING_id
mapped_gene_str_IDs <- mapped_gene_str_IDs[!is.na(mapped_gene_str_IDs) & mapped_gene_str_IDs != ""]

if (length(mapped_gene_str_IDs) == 0) {
  stop("No valid STRING IDs were found for the provided genes and organism.")
}

# Find neighbors of the input genes and map to gene symbols
input_genes_neighbors <- find_unique_neighbors(string_db, mapped_gene_str_IDs)
neighbor_genes_symbols <- map_neighbors_to_symbols(string_db, input_genes_neighbors)
write.csv(neighbor_genes_symbols, file.path(output_dir, "all_neighbor_genes_of_initial_input_genes.csv"))

# Plot the PPI network of the input genes and save as PDF
gene_identity1 <- "Input Genes"
pdf(file.path(output_dir, "ppi_network_of_input_genes.pdf"))
plot_network(string_db, mapped_gene_str_IDs, gene_identity1)
dev.off()

####################### Cluster Finder #######################
# clustering_methods <- c("fastgreedy", "walktrap", "spinglass", "edge.betweenness")
# print(clustering_methods)

# INPUT: Use clustering_method for clustering the input gene list
find_all_clusters(gene_symbols, clustering_method, output_dir)

####################### Enrichment Analysis for Input Genes #########
input_genes_enrichment_analysis(mapped_gene_str_IDs, output_dir)

############################# Single Gene Analysis ##########################
single_gene_lists <- mapped_gene_str_IDs
single_gene_analysis(single_gene_lists, output_dir)

##############################################################################
############################# Query Gene Analysis ############################

load_query_gene <- file.path(script_dir, "Fibro_UP_genes.csv")
if (file.exists(load_query_gene)) {
  matched_gene_ids <- query_gene_network_analysis(load_query_gene, mapped_gene_str_IDs, neighbor_genes_symbols, output_dir)
  if (!is.null(matched_gene_ids) && length(matched_gene_ids) > 0) {
    query_genes_enrichment_analysis(matched_gene_ids, output_dir)
  } else {
    message("Skipping query gene enrichment because no matched gene IDs were found.")
  }
} else {
  message("Skipping query gene analysis because Fibro_UP_genes.csv is not available.")
}
