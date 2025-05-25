
batch_effect_correction <- function(input_file, output_dir, user_id) {
  library(jsonlite)
  library(sva) # For batch effect correction
  
  tryCatch(
    {
      # Read and preprocess data - preserve exact feature names
      merged_df_data <- read.csv(input_file, header = TRUE, row.names = 1, check.names = FALSE)
      merged_df_data <- na.omit(merged_df_data)
      
      # Ensure unique column names
      colnames(merged_df_data) <- make.unique(colnames(merged_df_data))
      
      # Extract condition and expression matrix
      condition_info <- merged_df_data$condition
      data_t <- t(merged_df_data[, !(colnames(merged_df_data) %in% c("condition", "batch"))])
      
      # Save original feature names
      feature_names <- rownames(data_t)
      sample_names <- colnames(data_t)
      
      # Batch effect correction with ComBat
      batch_info <- merged_df_data$batch
      data_combat <- ComBat(dat = as.matrix(data_t), batch = batch_info, par.prior = TRUE, prior.plots = FALSE)
      
      # Restore original feature names
      rownames(data_combat) <- feature_names
      
      # Save corrected data
      output_file <- file.path(output_dir, paste0("batch_", basename(input_file)))
      data_corrected <- t(data_combat)
      data_corrected_with_condition <- cbind(condition = condition_info, data_corrected)
      
      # Write CSV with proper quoting to preserve commas/spaces in feature names
      write.csv(
        data_corrected_with_condition,
        output_file,
        row.names = TRUE,
        quote = TRUE,
        na = "",
        fileEncoding = "UTF-8")
      
      # Create boxplots in PDF and PNG formats only
      plot_formats <- c("pdf", "png")
      for (fmt in plot_formats) {
        file_name <- file.path(output_dir, paste0("batch_correction_boxplots.", fmt))
        
        # Set up the plotting device
        if (fmt == "png") {
          png(file_name, width = 1200, height = 600, res = 300)
        } else {
          pdf(file_name, width = 12, height = 6)
        }
        
        # Create the plots
        par(mfrow = c(1, 2), mar = c(10, 5, 4, 2))
        
        # Pre-correction plot
        boxplot(data_t,
                main = "Before Batch Correction", 
                las = 2, 
                col = "lightblue", 
                outline = FALSE,
                ylab = "Expression Levels", 
                cex.axis = 0.7, 
                names = sample_names)
        
        # Post-correction plot
        boxplot(data_combat,
                main = "After Batch Correction", 
                las = 2, 
                col = "lightgreen",
                outline = FALSE, 
                ylab = "Expression Levels", 
                cex.axis = 0.7, 
                names = sample_names)
        
        dev.off()
      }
      
      # Output completion message
      cat("Batch effect correction completed. Corrected data saved to:", output_file, "\n")
      cat("Boxplots saved in PDF and PNG formats.\n")
    },
    error = function(e) {
      # Handle errors gracefully
      cat("Error: ", e$message, "\n")
    }
  )
}

# Example command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_dir <- args[2]
user_id <- args[3]

batch_effect_correction(input_file, output_dir, user_id)
