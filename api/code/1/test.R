cdata <- readRDS("rds/count_data.rds")
mdata <- readRDS("rds/sample_info.rds")

cdata <- cdata[, !colnames(cdata) %in% c('GSM6765254', 'GSM6765268')]

head(mdata)

mdata <- mdata[!rownames(mdata) %in% c('GSM6765254', 'GSM6765268'), , drop = FALSE]

head(cdata)
