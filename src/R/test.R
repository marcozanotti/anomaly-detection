reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(weird)

reticulate::source_python('src/Python/utils/utilities.py')
source('src/R/utils.R')

data = get_data(path_list = c('data', 'nab'), name_list = c('nab', 'prep'))
data = data |> 
    filter(unique_id == "speed_7578")
plot(x = data$ds, y = data$y, type = "l")


weird::






