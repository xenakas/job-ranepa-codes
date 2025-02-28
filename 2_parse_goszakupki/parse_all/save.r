df <-  readRDS('small_df.rds')

options(java.parameters = "-Xmx8000m")

library(readxl)
library(xlsx)

write.xlsx(df, 'small_goszak_no_duplicates.xlsx')
