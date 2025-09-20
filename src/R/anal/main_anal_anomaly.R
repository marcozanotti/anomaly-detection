
# install.packages('tidyverse')
# install.packages('greybox')
# install.packages('DT')
# install.packages('patchwork')
# install.packages('reticulate')

reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')

config = get_config('config/anal/anal_anomaly_config.yaml')

analyze_anomaly_results(config)
