args <- commandArgs(trailingOnly = TRUE)
id <- args[1]
library(here)


print(id)

# Ensure the number of samples match


setwd(here("api", "code", id))


dds <- readRDS("rds/dds.rds")
condition <- readRDS("rds/condition.rds")
