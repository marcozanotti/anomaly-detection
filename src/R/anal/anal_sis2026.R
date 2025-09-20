# Empirical Analysis of Results
reticulate::use_condaenv('global_retrain')

library(tidyverse)
library(greybox)
library(DT)
library(patchwork)
library(reticulate)

source('src/R/utils.R')
reticulate::source_python('src/Python/utils/utilities.py')

# NOTE:
# plot andamenti 1120x525
# plot test 730x635
# plot test doppio 1120x525
# plot overall 800x800 or 900x800


# Load & prepare data -----------------------------------------------------

# run twice, one for absolute and one for relative
analysis_file_name <- 'docs/sis2026/anomevaltime_20250920_175046.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Parameters --------------------------------------------------------------

dataset_name <- 'nab'



# Analysis ----------------------------------------------------------------

# =========================================================================
# * Anomaly ---------------------------------------------------------------
# =========================================================================
anom_res <- res[[dataset_name]][['anomaly']][['results']]
anom_metrics <- c('accuracy', 'precision', 'recall', 'f1', 'auc')

# ** Tables ---------------------------------------------------------------
print(anom_res$tables, include.rownames = FALSE)

# ** Plots ----------------------------------------------------------------
anom_res$plots

# ** Tests ----------------------------------------------------------------
for (am in anom_metrics) {
	g <- anom_res$tests |> 
		plot_test_anomaly_results(
			.metric = am, 
			facet = TRUE,
			metric_label = toupper(am),
			title = paste0(toupper(dataset_name), " - ", toupper(am), " Test Comparison")
		)
	print(g)
}


# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================
eval_res <- res[[dataset_name]][['evaluation']][['results']]

# ** Tables ---------------------------------------------------------------
print(eval_res$tables, include.rownames = FALSE)


# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================
time_res <- res[[dataset_name]][['time']][['results']]

# ** Tables ---------------------------------------------------------------
print(time_res$tables, include.rownames = FALSE)


# =========================================================================
# * Comparison ------------------------------------------------------------
# =========================================================================

# anom_data <- res[[dataset_name]][['anomaly']][['data']]
# eval_data <- res[[dataset_name]][['evaluation']][['data']]
# plot_scatter_results()