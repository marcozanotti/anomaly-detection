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
analysis_file_name <- 'docs/global_ensemble/absolute_evaltimecost_overlap_20250708_180356.RData'
# analysis_file_name <- 'docs/global_ensemble/relative_evaltimecost_overlap_20250708_180211.RData'

res <- load(analysis_file_name)
res <- analysis_results
rm(analysis_results)



# Parameters --------------------------------------------------------------

dataset_name1 <- 'm5_daily'
dataset_name2 <- 'vn1_weekly'

# run twice, one for ML_DL and one for ENSACC_ENSTIME
models_type <- 'ML_DL'
models_type <- 'ENSACC_ENSTIME'



# Analysis ----------------------------------------------------------------

# =========================================================================
# * Evaluation ------------------------------------------------------------
# =========================================================================

eval_res1 <- res[[dataset_name1]][['evaluation']][['results']][[models_type]]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']][[models_type]]
eval_metrics <- c('rmsse', 'scaled_mqloss')

# ** Tables ---------------------------------------------------------------
for (m in eval_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(eval_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(eval_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
for (m in eval_metrics) {
	print(
		eval_res1$plots[[m]] + eval_res2$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# ** Tests -----------------------------------------------------------------
for (m in eval_metrics) {
	g1 <- plot_test_results_facet(
		data = eval_res1$tests[[m]],
		.metric = m, 
		by = "retrain_window", 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", ""), "- Nemenyi Test"))
	)
	print(g1)
	g2 <- plot_test_results_facet(
		data = eval_res2$tests[[m]],
		.metric = m, 
		by = "retrain_window", 
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", ""), "- Nemenyi Test"))
	)
	print(g2)
}

# =========================================================================
# * Time ------------------------------------------------------------------
# =========================================================================

time_res1 <- res[[dataset_name1]][['time']][['results']][[models_type]]
time_res2 <- res[[dataset_name2]][['time']][['results']][[models_type]]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------
for (m in time_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(time_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(time_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

# ** Plots -----------------------------------------------------------------
for (m in time_metrics) {
	print(
		time_res1$plots[[m]] + time_res2$plots[[m]] +
			patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	)
}

# =========================================================================
# * Cost ------------------------------------------------------------------
# =========================================================================

cost_res1 <- res[[dataset_name1]][['cost']][['results']][[models_type]]
cost_res2 <- res[[dataset_name2]][['cost']][['results']][[models_type]]
cost_metrics <- c('cost', 'savings_perc')

# ** Tables ---------------------------------------------------------------
for (m in cost_metrics) {
	cat(paste(dataset_name1, m, "\n\n"))
	print(xtable::xtable(cost_res1$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
	cat(paste(dataset_name2, m, "\n\n"))
	print(xtable::xtable(cost_res2$tables[[m]]$x$data, digits = 3), include.rownames = FALSE)
	cat("\n\n")
}

res[[dataset_name2]][['cost']]$data |> 
	filter(type %in% c('ENSACC', 'ENSTIME')) |> 
	group_by(retrain_window) |> 
	summarize(cost = mean(cost)) |> 
	mutate(method = 'Average') |> 
	pivot_wider(names_from = retrain_window, values_from = cost) |>  
	xtable::xtable(digits = 3) |> 
	print(include.rownames = FALSE)

# =========================================================================
# * Ensemble Comparisons --------------------------------------------------
# =========================================================================

models_types <- c('ML_DL', 'ENSACC_ENSTIME')

# Overall Results ---------------------------------------------------------

# * Evaluation ------------------------------------------------------------
eval_res1 <- res[[dataset_name1]][['evaluation']][['results']]
eval_res2 <- res[[dataset_name2]][['evaluation']][['results']]
eval_metrics <- c('rmsse', 'scaled_mqloss')

# ** Tables ---------------------------------------------------------------
for (m in eval_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				eval_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				eval_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						eval_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						eval_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 3
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# ** Tests ----------------------------------------------------------------
for (m in eval_metrics) {
	g1 <- plot_test_results(
		data = dplyr::bind_rows(
			eval_res1[[models_types[[1]]]]$tests[[m]], 
			eval_res1[[models_types[[2]]]]$tests[[m]]
		),
		.metric = m, 
		by = "method", 
		.retrain_window = 7,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name1), "_.*", " "), "- Nemenyi Test"))
	) + 
		ggplot2::labs(x = "") +
		ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90))
	g2 <- plot_test_results(
		data = dplyr::bind_rows(
			eval_res2[[models_types[[1]]]]$tests[[m]], 
			eval_res2[[models_types[[2]]]]$tests[[m]]
		),
		.metric = m, 
		by = "method", 
		.retrain_window = 1,
		metric_label = toupper(gsub("_", " ", m)),
		title = toupper(paste(stringr::str_replace_all(toupper(dataset_name2), "_.*", " "), "- Nemenyi Test"))
	) + 
		ggplot2::labs(x = "") +
		ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90))
	g3 <- g1 / g2 +	patchwork::plot_layout(guides = "collect")
	print(g3)
}

# * Time ------------------------------------------------------------------
time_res1 <- res[[dataset_name1]][['time']][['results']]
time_res2 <- res[[dataset_name2]][['time']][['results']]
time_metrics <- c('total_sample_time')

# ** Tables ---------------------------------------------------------------
for (m in time_metrics) {
	cat(paste(m, "\n\n"))
	print(
		xtable::xtable(
			dplyr::bind_rows(
				time_res2[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
				time_res2[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
			) |>
				dplyr::left_join(
					dplyr::bind_rows(
						time_res1[[models_types[[1]]]]$tables[[m]]$x$data |> dplyr::select(1:2),
						time_res1[[models_types[[2]]]]$tables[[m]]$x$data |> dplyr::select(1:2)
					),
					by = 'Method'
				) |> 
				dplyr::rename('M5' = `7`, 'VN1' = `1`) |>
				dplyr::relocate("M5", .after = "Method"),	
			digits = 0
		), 
		include.rownames = FALSE
	)
	cat("\n\n")
}

# * Cost ------------------------------------------------------------------
cost_res1 <- res[[dataset_name1]][['cost']][['results']]
cost_res2 <- res[[dataset_name2]][['cost']][['results']]
cost_metrics <- c('cost')

# ** Plots ----------------------------------------------------------------
for (m in cost_metrics) {
	g1 <- plot_compared_results(
		dplyr::bind_rows(
			cost_res1[[models_types[[1]]]]$plots[[m]]$data,
			cost_res1[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 7, title = "M5"
	)
	g2 <- plot_compared_results(
		dplyr::bind_rows(
			cost_res2[[models_types[[1]]]]$plots[[m]]$data,
			cost_res2[[models_types[[2]]]]$plots[[m]]$data
		), 
		metric = m, .retrain_window = 1, title = "VN1"
	)
	g3 <- g1 / g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	print(g3)
}

# * Evaluation - Cost -----------------------------------------------------
metrics <- list(c('rmsse', 'cost'), c('scaled_mqloss', 'cost'))
for (m in metrics) {
	g1 <- dplyr::bind_rows(
		eval_res1[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res1[[models_types[[2]]]]$plots[[m[1]]]$data,
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				cost_res1[[models_types[[1]]]]$plots[[m[2]]]$data,
				cost_res1[[models_types[[2]]]]$plots[[m[2]]]$data,
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 7, title = "M5",
		)
	g2 <- dplyr::bind_rows(
		eval_res2[[models_types[[1]]]]$plots[[m[1]]]$data,
		eval_res2[[models_types[[2]]]]$plots[[m[1]]]$data,
	) |> 
		dplyr::left_join(
			dplyr::bind_rows(
				cost_res2[[models_types[[1]]]]$plots[[m[2]]]$data,
				cost_res2[[models_types[[2]]]]$plots[[m[2]]]$data,
			),
			by = c('type', 'method', 'retrain_window')
		) |> 
		plot_scatter_results(
			metrics = m, .retrain_window = 1, title = "VN1",
		)
	g3 <- g1 + g2 + 
		patchwork::plot_layout(guides = "collect") & ggplot2::theme(legend.position = "bottom")
	print(g3)
}


# Retraining Results ------------------------------------------------------

# * Cost ------------------------------------------------------------------
cost_res1 <- res[[dataset_name1]][['cost']][['results']]
cost_res2 <- res[[dataset_name2]][['cost']][['results']]
cost_metrics <- c('cost')

scaling_fun <- function(x) { scales::dollar(x, big.mark = ",", decimal.mark = '.', accuracy = 1) }

for (m in cost_metrics) {
	print(
		cost_res1[[models_types[1]]]$plots[[m]] + 
			ggplot2::scale_y_continuous(limits = c(0, 4000000), labels = scaling_fun) +
			ggplot2::theme(legend.position = "bottom") +
		cost_res1[[models_types[2]]]$plots[[m]] + 
			ggplot2::scale_y_continuous(limits = c(0, 4000000), labels = scaling_fun) +
			ggplot2::theme(legend.position = "bottom")
	)
	print(
		cost_res2[[models_types[1]]]$plots[[m]] + 
			ggplot2::scale_y_continuous(limits = c(0, 2000000), labels = scaling_fun) +
			ggplot2::theme(legend.position = "bottom") +
		cost_res2[[models_types[2]]]$plots[[m]] + 
			ggplot2::scale_y_continuous(limits = c(0, 2000000), labels = scaling_fun) +
			ggplot2::theme(legend.position = "bottom")
	)
}
