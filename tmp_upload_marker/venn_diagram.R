args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
if (is.na(id) || id == "") {
  stop("Missing user id argument")
}

script_path_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_path_arg[1]), winslash = "/", mustWork = TRUE)
script_dir <- dirname(script_path)
venn_dir <- file.path(script_dir, id, "venn")

if (!dir.exists(venn_dir)) {
  stop(sprintf("Venn workspace not found: %s", venn_dir))
}

setwd(venn_dir)

source(file.path(script_dir, "1_init_venn_m.R"))
source(file.path(script_dir, "2_gene_list_m.R"))
source(file.path(script_dir, "3_plot_venn_m.R"))
source(file.path(script_dir, "4_wide_frame_venn_m.R"))


reg <- readRDS(file.path("rds", "reg.rds"))
input_data <- readRDS(file.path("rds", "input_data.rds"))


input_data

gene_lists <- take_user_inputs(input_data)

plot_and_save_venn(gene_lists, reg) # This will save files with "down" in their names


# Call the function to create the Venn data frame
venn_result <- create_venn_dataframe(gene_lists)

# Concatenate to create the output file name
output_file <- file.path("files", paste0(reg, "_venn_result.csv"))
write.csv(venn_result, file = output_file, row.names = FALSE)
###############################################################################
