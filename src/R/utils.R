
remove_series <- function(data, min_series_length) {
  
    # Compute series length
    series_length <- data |> 
      dplyr::group_by(unique_id) |> 
      dplyr::count()
    
    message(sprintf("Removing series shorter than %d observations...", min_series_length))
    
    # IDs to remove
    remove_ids <- series_length |> 
      dplyr::filter(n < min_series_length) |> 
      dplyr::pull(unique_id)
    
    # Filter dataset
    res_df <- data |> filter(!unique_id %in% remove_ids)
    
    # Summary stats
    n_series <- length(unique(data$unique_id))
    n_series_final <- length(unique(res_df$unique_id))
    n_series_to_remove <- n_series - n_series_final
    p_series_to_remove <- n_series_to_remove / n_series * 100
    
    message(sprintf(
      "Removed %d series out of %d (%.1f%%).", 
      n_series_to_remove, n_series, p_series_to_remove
    ))
    message(sprintf("The final dataset contains %d series.", n_series_final))
    
    return(res_df)
}

filter_series <- function(data, max_series_length) {
  
    message(sprintf("Filtering series longer than %d observations...", max_series_length))
    
    res_df <- data |> 
      dplyr::arrange(unique_id, ds) |> 
      dplyr::group_by(unique_id) |> 
      dplyr::slice_tail(n = max_series_length) |> 
      dplyr::ungroup()
    
    # Summary stats
    n_obs <- nrow(data)
    n_obs_final <- nrow(res_df)
    n_obs_to_remove <- n_obs - n_obs_final
    p_obs_to_remove <- n_obs_to_remove / n_obs * 100
    
    message(sprintf(
      "Removed %d observations out of %d (%.1f%%).", 
      n_obs_to_remove, n_obs, p_obs_to_remove
    ))
    message(sprintf("The final dataset contains %d observations.", n_obs_final))
    
    return(res_df)
}

sampling_data <- function(data, samples = 1000) {
  
    message(sprintf("Sampling %d series from data...", samples))
    
    ids <- unique(data$unique_id)
    sample_ids <- sample(ids, size = samples, replace = FALSE)
    
    res_df <- data |>
      dplyr::filter(unique_id %in% sample_ids)
    
    return(res_df)
}

get_data <- function(
    path_list, 
    name_list, 
    min_series_length = NULL, 
    max_series_length = NULL, 
    samples = NULL, 
    ext = '.parquet'
) {

    res_df = load_data(path_list, name_list, ext)

    if (!is.null(min_series_length)) {
      res_df = remove_series(res_df, min_series_length)
    }
  
    if (!is.null(max_series_length)) {
      res_df = filter_series(res_df, max_series_length)
    }

    if (!is.null(samples)) {
      res_df = sampling_data(res_df, samples)
    }

    res_df = res_df |> dplyr::arrange('unique_id', 'ds')
  
    return(res_df)

}

get_aggregate_function <- function(function_name) {

    if (function_name == 'mean') {
      return(mean)
    } else if (function_name == 'median') {
      return(median)
    } else if(function_name == 'std') {
      return(sd)
    } else if (function_name == 'max') {
      return(max)
    } else if(function_name =='min') {
      return(min)
    } else if(function_name == 'sum') {
      return(sum)
    } else {
      stop('Invalid aggregate function')
    }
    
}

aggregate_data <- function(
    data, 
    group_columns, 
    drop_columns = NULL, 
    function_name = 'mean',
    adjust_metrics = FALSE
) {

    data_agg = data

    if (!is.null(drop_columns)) {
      data_agg <- data_agg |> 
        dplyr::select(-dplyr::any_of(drop_columns))
    }

    agg_fun <- get_aggregate_function(function_name)
  
    data_agg <- data_agg |> 
      dplyr::mutate(
        dplyr::across(
          where(is.numeric), 
          ~ ifelse(is.nan(.x) | is.infinite(.x), NA_real_, .x)
        )
      )

    data_agg <- data_agg |> 
      dplyr::group_by(!!!rlang::syms(group_columns)) |> 
      dplyr::summarise(dplyr::across(where(is.numeric), agg_fun, na.rm = TRUE), .groups = 'drop')

    if (adjust_metrics) {

      if ('mse' %in% names(data_agg))
        data_agg[['rmse']] = sqrt(data_agg[['mse']])

      if ('msse' %in% names(data_agg))
        data_agg[['rmsse']] = sqrt(data_agg[['msse']])
        
      if ('total_fit_time' %in% names(data_agg)) {
        model_names <- unique(data_agg[['method']])
        retrain_scenarios <- unique(data_agg[['retrain_window']])
        total_fit_time <- c()
        for (m in model_names) {
          for (rs in retrain_scenarios) {
            data_tmp <- data |> dplyr::filter(method == m & retrain_window == rs)
            ids_tmp <- get_retrain_ids(
              data_tmp[['test_window']][1], 
              data_tmp[['horizon']][1], 
              data_tmp[['retrain_window']][1]
            )
            tot_fit_time_tmp <- data_tmp[['total_fit_time']][ids_tmp]
            total_fit_time <- c(total_fit_time, agg_fun(tot_fit_time_tmp))
          }
        }
        data_agg[['total_fit_time']] = total_fit_time
      }

    }

    return(data_agg)

}

get_model_type <- function(model_name) {

    sf <- c('ETS', 'ARIMA', 'Theta', 'TBATS', 'MFLES', 'MSTL', 'CES')
    ens <- c(
      'EnsembleMean1', 'EnsembleMean2', 'EnsembleMean3', 'EnsembleMean4',
      'EnsembleMean5', 'EnsembleMean6', 'EnsembleMean7', 'EnsembleMean8',
      'EnsembleMean9', 'EnsembleMean10', 'EnsembleMean11', 'EnsembleMean12'
    )

    model_type <- dplyr::case_when(
      model_name %in% sf ~ 'SF',
      model_name %in% ens ~ 'ENS',
      TRUE ~ NA_character_
    )

    return(model_type)

}

get_model_name_abbr <- function(model_name) {

  model_name_abbr <- dplyr::case_when(
    model_name == 'ETS' ~ 'ETS',
    model_name == 'ARIMA' ~ 'ARIMA',
    model_name == 'Theta' ~ 'Theta',
    model_name == 'TBATS' ~ 'TBATS',
    model_name == 'MFLES' ~ 'MFLES',
    model_name == 'MSTL' ~ 'MSTL',
    model_name == 'CES' ~ 'CES',
    model_name == 'EnsembleMean1' ~ 'Ens1',
    model_name == 'EnsembleMean2' ~ 'Ens2',
    model_name == 'EnsembleMean3' ~ 'Ens3',
    model_name == 'EnsembleMean4' ~ 'Ens4',
    model_name == 'EnsembleMean5' ~ 'Ens5',
    model_name == 'EnsembleMean6' ~ 'Ens6',
    model_name == 'EnsembleMean7' ~ 'Ens7',
    model_name == 'EnsembleMean8' ~ 'Ens8',
    model_name == 'EnsembleMean9' ~ 'Ens9',
    model_name == 'EnsembleMean10' ~ 'Ens10',
    model_name == 'EnsembleMean11' ~ 'Ens11',
    model_name == 'EnsembleMean12' ~ 'Ens12',
    TRUE ~ model_name
  )
  return(model_name_abbr)

}

recode_data <- function(data, model_type_levels, model_names_abbr) {

    data_recoded <- data |>  
      dplyr::mutate(
        type = get_model_type(method),
        type = factor(type, levels = model_type_levels, ordered = TRUE),
        .before = 'method',
      ) |> 
      dplyr::mutate(
        method = factor(get_model_name_abbr(method), levels = model_names_abbr, ordered = TRUE)
      ) |> 
      dplyr::arrange(type, method)
    return(data_recoded)

}

dt_table <- function(data, title = "", caption = "", rownames = FALSE, digits = 2, format = 'numeric') {
	
    p_len <- nrow(data)
    
    if (format == 'dollar') {
      res_data <- data |>
        dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits))) |>  
        dplyr::mutate(dplyr::across(where(is.numeric), ~ scales::dollar(.x, big.mark = ",", decimal.mark = '.')))
    } else if (format == 'percent') {
      res_data <- data |>
        dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits))) |>  
        dplyr::mutate(dplyr::across(where(is.numeric), ~ scales::percent(.x, scale = 1)))
    } else {
      res_data <- data |>
        dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, digits = digits)))
    }
    
    res <- res_data |> 
      DT::datatable(
        extensions = "Buttons",
        # filter = "top", # for filtering enable searching and lenghtChange
        options = list(
          pageLength = p_len,
          paging = FALSE,
          searching = FALSE,
          ordering = FALSE,
          lenghtChange = FALSE,
          autoWidth = FALSE,
          dom = "Bfrtip",
          buttons = c("copy", "print", "csv", "excel", "pdf"),
          drawCallback = DT::JS(
            c(
              "function(settings){",
              "  var datatable = settings.oInstance.api();",
              "  var table = datatable.table().node();",
              paste0("  var caption = '", caption, "'"),
              "  $(table).append('<caption style=\"caption-side: bottom\">' + caption + '</caption>');",
              "}"
            )
          )
        ),
        rownames = rownames,
        caption = title
      )
    return(res)
	
}

table_results <- function(data, metrics, digits = 3) {

  res <- data |> 
    dplyr::select(dplyr::all_of(c('method', metrics))) |> 
    dplyr::rename_with(toupper) 

  if ('scaled_mqloss' %in% metrics) {
    res <- res |> dplyr::rename('SMQL' = 'SCALED_MQLOSS')
  }

  if ('total_sample_time' %in% metrics) {
    res <- res |> dplyr::rename('CT' = 'TOTAL_SAMPLE_TIME')
  }
  
  res <- xtable::xtable(res, digits = digits)
  return(res)

}

test_differences <- function(
  data, 
  .metric, 
  by = 'anomaly_type', 
  .level = 99
) {

  cat(paste0("Testing differences in ", .metric, "...\n"))
  data_test <- data |> dplyr::filter(type == 'ENS')

  data_test <- data_test |> 
    dplyr::filter(level == .level) |> 
    dplyr::select(dplyr::all_of(c('unique_id', 'method', 'anomaly_type', .metric))) |> 
    tidyr::pivot_wider(names_from = 'anomaly_type', values_from = .metric) |> 
    dplyr::select(-unique_id, -method) |>
    as.matrix()

  test_res <- greybox::rmcb(data = data_test, level = 0.95, outplot = "none")
  data_test <- tibble::tibble(
    'anomaly_type' = names(test_res$mean),
    'metric' = .metric, 
    'mean' = test_res$mean,
    'lower' = test_res$interval[, 1],
    'upper' = test_res$interval[, 2],
    'pvalue' = test_res$p.value
  )
  
  return(data_test)
    
}

extract_significance <- function(p_value) {
	
  cat("Extracting significance...\n")
	res <- dplyr::case_when(
		p_value < 0.001 ~ "***", 
		p_value >= 0.001 & p_value < 0.01 ~ "**",
		p_value >= 0.01 & p_value < 0.05 ~ "*",
		p_value >= 0.05 & p_value < 0.1 ~ ".",
		TRUE ~ ""
	)
	return(res)
	
}

plot_test_anomaly_results <- function(
  data, 
  .metric, 
  by = "anomaly_type", 
  facet = FALSE, 
  metric_label = "", 
  title = ""
) {
	
	cat("Creating plot...\n")
  data_plot <- data |> 
    dplyr::filter(testing == by) |> 
    dplyr::filter(metric == .metric) |> 
    dplyr::mutate(anomaly_type = factor(anomaly_type, ordered = FALSE)) |> 
    dplyr::mutate(level = factor(level, ordered = TRUE))
  data_min <- data_plot |> 
    dplyr::group_by(level) |> 
    dplyr::slice_min(mean) |> 
    dplyr::ungroup()
  
  g <- data_plot |> 
    ggplot2::ggplot(ggplot2::aes(x = anomaly_type, y = mean)) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper), 
      col = 'lightblue', width = 0.1, linewidth = 1
    ) +
    ggplot2::geom_point(size = 4, col = 'lightblue') +
    # ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = lower), col = 'gray', linetype = 2) +
    # ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = upper), col = 'gray', linetype = 2) +
    ggplot2::labs(title = title, x = 'Anomaly Type', y = 'Rank') + 
    ggplot2::theme_bw() +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  if (facet) {
    g <- g +
      ggplot2::facet_wrap(~ level, ncol = 2, scales = 'free_y')
  }

	return(g)
	
}

plot_distribution_results <- function(data, metric, metric_label = "", title = "") {
	
	cat("Creating plot...\n")	

  data_plot <- data |> dplyr::filter(type == 'ENS')
  anomaly_types <- unique(data_plot$anomaly_type)

	g <- data_plot |> 
		ggplot2::ggplot(
			ggplot2::aes(
				x = .data[[metric]], 
				color = .data[['anomaly_type']] |> factor(levels = anomaly_types, ordered = FALSE)
			)
		) +
		ggplot2::geom_density() +
		ggplot2::facet_wrap(~ .data[['level']], nrow = 1, ncol = 2) + 
		# ggplot2::scale_x_continuous(breaks = retrain_scenario) +
		ggplot2::labs(
			title = title, 
			x = metric_label, y = 'Density',
			color = 'Anomaly Type'
		) + 
		ggplot2::theme_bw() +
		ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5), legend.position = "bottom")
	
	return(g)
	
}

clean_outliers <- function(data, .metric, q = c(0.003, 0.997)) {
	
	cat("Removing outliers...\n")
	
	q_funs <- list(
		'_qlow' = function(x) { round(quantile(x, q[1], na.rm = TRUE), 3) },
		'_qhigh' = function(x) { round(quantile(x, q[2], na.rm = TRUE), 3) }
	)
	qs_df <- data |> 
		dplyr::summarise(dplyr::across(.metric, .fns = q_funs)) |> 
		tidyr::pivot_longer(cols = dplyr::everything()) |>
		tidyr::separate(name, into = c('metric', 'q'), sep = "__") |>
		tidyr::pivot_wider(names_from = q, values_from = value)
	
	data_cln <- data
	for (i in 1:nrow(qs_df)) {
		m <- qs_df[['metric']][i]
		q_low <- qs_df[['qlow']][i]
		q_high <- qs_df[['qhigh']][i]
		cat(paste0("Metric: ", m, ", Q Low: ", q_low, ", Q High: ", q_high, "\n"))
		data_cln <- data_cln |> 
			dplyr::filter(dplyr::between(.data[[m]], q_low, q_high))
	}
	
	n_full <- nrow(data)
	n_filtered <- nrow(data_cln)
	cat(
		paste0(
			"Removed ", n_full - n_filtered, " (", 
			round((n_full - n_filtered) / n_full * 100, 1), 
			"%) of results\n"
		)
	)
	
	return(data_cln)
	
}

flatten_list <- function(list) {
  return(purrr::map_chr(list, ~ paste0(.x, collapse = "_")))
}

create_results_list <- function(dataset_names, analysis_types, data_types, final_analyses) {

  dataset_names <- flatten_list(dataset_names)
  analysis_types <- flatten_list(analysis_types)
  data_types <- flatten_list(data_types)
  final_analyses <- flatten_list(final_analyses)

  # level 1 list of dataset names 
  res_list <- vector("list", length(dataset_names)) |> 
    purrr::set_names(dataset_names)

  for (i in seq_along(res_list)) {
    res_list[[i]] <- vector("list", length(analysis_types)) |> 
      purrr::set_names(analysis_types)
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      res_list[[i]][[j]] <- vector("list", length(data_types)) |> 
        purrr::set_names(data_types)
    }
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      res_list[[i]][[j]][['results']] <- vector("list", length(final_analyses)) |> 
        purrr::set_names(final_analyses)
    }
  }
  # for (i in seq_along(res_list)) {
  #   for (j in seq_along(res_list[[i]])) {
  #     for (k in seq_along(res_list[[i]][[j]][['results']])) {
  #       res_list[[i]][[j]][['results']][[k]] <- vector("list", length(final_analyses)) |> 
  #         purrr::set_names(final_analyses)
  #     }
  #   }
  # }

  gc()
  return(res_list)

}

get_table_plot_params <- function(analysis_metric, analysis_method) {

	# default values
	digits <- 3
	format <- 'numeric'
	label <- toupper(analysis_metric)
	add_average <- FALSE
	scaling_fun <- function(x) { scales::number(x, accuracy = 0.001) }

  if (analysis_metric == 'mqloss') {
    label <- 'MQL'
  } else if (analysis_metric == 'scaled_mqloss') {
    label <- 'SMQL'
  } else if (analysis_metric == 'total_sample_time') {
    label <- 'Computing Time'
    if (analysis_method == 'absolute') {
    	digits <- 0
    	scaling_fun <- function(x) { scales::number(x, accuracy = 1) }
    }
  } else if (analysis_metric == 'cost') {
  	digits <- 0
  	format <- 'dollar'
    label <- 'Cost ($)'
    add_average <- TRUE
    scaling_fun <- function(x) { scales::dollar(x, big.mark = ",", decimal.mark = '.', accuracy = 1) }
  } else if (analysis_metric == 'savings') {
  	digits <- 0
  	format <- 'dollar'
    label <- 'Savings ($)'
    add_average <- TRUE
    scaling_fun <- function(x) { scales::dollar(x, big.mark = ",", decimal.mark = '.', accuracy = 1) }
  } else if (analysis_metric == 'savings_perc') {
  	digits <- 0
  	format <- 'percent'
    label <- 'Savings (%)'
    add_average <- TRUE
    scaling_fun <- function(x) { scales::percent(x, scale = 1, accuracy = 1) }
  } else {
  	x <- 'Keep default'
  }
	
  res <- list(
    'digits' = digits,
    'format' = format,
    'label' = label,
    'add_average' = add_average,
    'scaling_fun' = scaling_fun
  )
  return(res)

}

analyze_anomaly_results <- function(config) {

  # analysis config
  analysis_name <- config$analysis$name
  analysis_types <- config$analysis$types
  # dataset config
  dataset_names <- config$dataset$dataset_names
  ext <- config$dataset$ext
  # model config
  model_types <- config$models$types
  model_names <- config$models$model_names
  model_names_abbr <- config$models$model_names_abbr
  model_type_levels <- unlist(model_types) 
  # anomaly, evaluation, and time config
  anom_metrics <- config$anomaly_params$metrics
  eval_metrics <- config$evaluation_params$metrics
  eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
  eval_outlier_cleaning_quantiles <- config$evaluation_params$outlier_cleaning_quantiles
  time_metrics <- config$time_params$metrics
  
  # final analysis names
  data_types <- c('data', 'results')
  final_analyses <- c('tables', 'plots', 'tests')  
  analysis_results <- create_results_list(
    dataset_names, analysis_types, data_types, final_analyses
  )
  
  for (dn in dataset_names) {

    cat(paste0("*************** Analysing ", dn, " ***************\n"))
    dataset_name_tmp <- dn

    cat("Loading and preparing the anomaly data...\n")
    anom_df_tmp = load_data(
      path_list = c('results', dataset_name_tmp, 'evaluation'),
      name_list = c(dataset_name_tmp, 'eval', 'anomaly'),
      ext = ext
    ) |> 
      tibble::as_tibble() |> 
      dplyr::filter(method %in% model_names) |> 
      dplyr::rename('anomaly_type' = 'type') |> 
      recode_data(model_type_levels, model_names_abbr)
    anom_df_agg_tmp <- anom_df_tmp |> 
      aggregate_data(
        group_columns = c('type', 'method', 'anomaly_type', 'level'),
        drop_columns = c('unique_id', 'test_window', 'horizon', 'retrain_window')
      )

    cat("Loading and preparing the evaluation data...\n")
    eval_df_tmp = load_data(
      path_list = c('results', dataset_name_tmp, 'evaluation'),
      name_list = c(dataset_name_tmp, 'eval'),
      ext = ext
    ) |> 
      tibble::as_tibble() |> 
      dplyr::filter(method %in% model_names) |> 
      recode_data(model_type_levels, model_names_abbr) |> 
      clean_outliers(eval_outlier_cleaning_metrics, eval_outlier_cleaning_quantiles)
    eval_df_agg_tmp <- eval_df_tmp |> 
      aggregate_data(
        group_columns = c('type', 'method'),
        drop_columns = c('unique_id', 'test_window', 'horizon', 'retrain_window'),
        adjust_metrics = TRUE
      )

    cat("Loading and preparing the time data...\n")
    time_df_tmp = load_data(
      path_list = c('results', dataset_name_tmp, 'evaluation'),
      name_list = c(dataset_name_tmp, 'time'),
      ext = ext
    ) |> 
      tibble::as_tibble() |> 
      dplyr::filter(method %in% model_names) |> 
      recode_data(model_type_levels, model_names_abbr)
    time_df_agg_tmp <- time_df_tmp |> 
      aggregate_data(
        group_columns = c('type', 'method'),
        drop_columns = c('unique_id', 'test_window', 'horizon', 'retrain_window'),
        function_name = 'sum',
        adjust_metrics = TRUE
      )
    
    # ANALYSIS ---------------------------------------------------------------
    cat("Creating anomaly table, plot and tests...\n")
    analysis_results[[dn]][['anomaly']][['data']] <- anom_df_agg_tmp
    anom_types <- unique(anom_df_tmp$anomaly_type)
    anom_levels <- unique(anom_df_tmp$level)
    anom_tab <- vector('list', length(anom_types) * length(anom_levels))
    anom_names <- expand.grid(anom_types, anom_levels) |> 
      tidyr::unite('name', Var1, Var2) |> 
      dplyr::pull('name') |> 
      sort()
    names(anom_tab) <- anom_names    
    # tables
    for (nm in anom_names) {
      anom_type_tmp <- gsub("_.*", "", nm)
      anom_lvl_tmp <- as.numeric(gsub(".*_", "", nm))
      anom_tab[[nm]] <- anom_df_agg_tmp |> 
        dplyr::filter(anomaly_type == anom_type_tmp & level == anom_lvl_tmp) |> 
        table_results(anom_metrics, digits = 4)
    }
    analysis_results[[dn]][['anomaly']][['results']][['tables']] <- anom_tab
    # plots
    anom_plots <- vector('list', length(anom_metrics))
    names(anom_plots) <- anom_metrics
    for (am in anom_metrics) {
      anom_plots[[am]] <- plot_distribution_results(
        anom_df_tmp, metric = am, metric_label = toupper(am), 
        title = paste0(toupper(dn), " - ", toupper(am), " Distribution Comparison")
      )
    }
    analysis_results[[dn]][['anomaly']][['results']][['plots']] <- anom_plots
    # tests
    anom_test <- NULL
    for (lvl in anom_levels) {
      cat(paste0("Testing level ", lvl, "...\n"))
      anom_test_tmp <- purrr::map(
        anom_metrics,
        ~ test_differences(anom_df_tmp, .metric = .x, by = "anomaly_type", .level = lvl)
      ) |>
      dplyr::bind_rows() |> 
      dplyr::mutate("testing" = "anomaly_type", "level" = lvl, .before = 1)
      anom_test <- dplyr::bind_rows(anom_test, anom_test_tmp)
    }
    analysis_results[[dn]][['anomaly']][['results']][['tests']] <- anom_test

    cat("Creating evaluation table, plot and tests...\n")
    analysis_results[[dn]][['evaluation']][['data']] <- eval_df_agg_tmp
    eval_tab <- table_results(eval_df_agg_tmp, eval_metrics)
    analysis_results[[dn]][['evaluation']][['results']][['tables']] <- eval_tab

    cat("Creating time table, plot and tests...\n")
    analysis_results[[dn]][['time']][['data']] <- time_df_agg_tmp
    time_tab <- table_results(time_df_agg_tmp, time_metrics, digits = 0)
    analysis_results[[dn]][['time']][['results']][['tables']] <- time_tab

  }

  cat("Saving results...\n")
  file_name <- paste0(
    stringr::str_sub_all(analysis_types, start = 1, end = 4) |> 
      unlist() |> 
      paste0(collapse = ""),  
    "_", 
    Sys.time() |> 
      as.character() |> 
      stringr::str_remove_all("\\..*") |> 
      stringr::str_replace_all("(-)|(:)", "") |> 
      stringr::str_replace_all(" ", "_"),
    ".RData"
  )
  save(analysis_results, file = paste0('docs/', analysis_name, '/', file_name))
  cat("Done!\n")

  return(invisible(NULL))

}

plot_scatter_results <- function(
		data, 
		metrics, 
		.retrain_window = NULL,
		analysis_method = 'absolute',
		title = ""
) {
	
	cat("Creating plot...\n")
	
	colors_lbls_2 <- c("SF" = "#17BECF", "ENS" = "#FFA500")
	
	params <- purrr::map(metrics, ~ get_table_plot_params(.x, analysis_method))
	names(params) <- c('x', 'y')
	
	if (is.null(.retrain_window)) {
		data_plot <- data |> 
			dplyr::select(dplyr::all_of(c('type', 'method', metrics)))
	} else {
		data_plot <- data |> 
			dplyr::filter(retrain_window == .retrain_window) |> 
			dplyr::select(dplyr::all_of(c('type', 'method', metrics)))
	}

	g <- data_plot |> 
		ggplot2::ggplot(
			ggplot2::aes(
				x = .data[[metrics[1]]], 
				y = .data[[metrics[2]]], 
				col = .data[['type']]
			)
		)
	
	g <- g + ggplot2::geom_point()
	
	g <- g +
		ggrepel::geom_text_repel(
			ggplot2::aes(label = .data[['method']]), 
			col = 'black', vjust = -0.25
		) + 
		ggplot2::scale_x_continuous(labels = params$x$scaling_fun) +
		ggplot2::scale_y_continuous(labels = params$y$scaling_fun) +
		ggplot2::scale_color_manual(values = colors_lbls_2) +
		ggplot2::labs(
			title = title, 
			x = params$x$label, y = params$y$label, col = 'Method Type'
		) + 
		ggplot2::theme_minimal() +
		ggplot2::theme(
			plot.title = ggplot2::element_text(hjust = 0.5),
			legend.position = "bottom",
			axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
		)
	
	return(g)
	
}