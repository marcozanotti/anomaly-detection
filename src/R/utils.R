
get_retrain_ids <- function(test_window, horizon, retrain_window = 1) {
  res = seq.int(from = 0, to = (test_window - horizon + 1), by = retrain_window)
  return(res)
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
  function_name = 'median', 
  adjust_metrics = False
) {

  data_agg = data

  if (!is.null(drop_columns)) {
    data_agg <- data_agg |> 
      dplyr::select(-dplyr::any_of(drop_columns))
  }

  agg_fun <- get_aggregate_function(function_name)

  data_agg <- data_agg |> 
    dplyr::group_by(!!!rlang::syms(group_columns)) |> 
    dplyr::summarise(dplyr::across(where(is.numeric), agg_fun), .groups = 'drop')

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

  sf <- c('ETS', 'ARIMA')
  ml <- c(
    'LinearRegression', 'Lasso', 'Ridge', 
    'RandomForestRegressor', 
    'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor' 
  )
  dl <- c('MLP', 'LSTM', 'TCN', 'NBEATSx', 'NHITS')
  ens_acc <- c('EnsembleMean2A', 'EnsembleMean3A', 'EnsembleMean4A', 'EnsembleMean5A')
  ens_time <- c('EnsembleMean2T', 'EnsembleMean3T', 'EnsembleMean4T', 'EnsembleMean5T')

  model_type <- dplyr::case_when(
    model_name %in% sf ~ 'SF',
    model_name %in% ml ~ 'ML',
    model_name %in% dl ~ 'DL',
    model_name %in% ens_acc ~ 'ENSACC',
    model_name %in% ens_time ~ 'ENSTIME',
    TRUE ~ NA_character_
  )

  return(model_type)

}

get_model_name_abbr <- function(model_name) {

  model_name_abbr <- dplyr::case_when(
    model_name == 'LinearRegression' ~ 'LR',
    model_name == 'RandomForestRegressor' ~ 'RF',
    model_name == 'XGBRegressor' ~ 'XGBoost',
    model_name == 'LGBMRegressor' ~ 'LGBM',
    model_name == 'CatBoostRegressor' ~ 'CatBoost',
    model_name == 'MLP' ~ 'MLP',
    model_name == 'LSTM' ~ 'LSTM',
    model_name == 'TCN' ~ 'TCN',
    model_name == 'NBEATSx' ~ 'NBEATSx',
    model_name == 'NHITS' ~ 'NHITS',
    model_name == 'EnsembleMean2A' ~ 'Ens2A',
    model_name == 'EnsembleMean3A' ~ 'Ens3A',
    model_name == 'EnsembleMean4A' ~ 'Ens4A',
    model_name == 'EnsembleMean5A' ~ 'Ens5A',
    model_name == 'EnsembleMean2T' ~ 'Ens2T',
    model_name == 'EnsembleMean3T' ~ 'Ens3T',
    model_name == 'EnsembleMean4T' ~ 'Ens4T',
    model_name == 'EnsembleMean5T' ~ 'Ens5T',
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

compute_relative_metrics <- function(data, type) {

  reference_data <- data |> 
    dplyr::group_by(method) |> 
    dplyr::slice_min(retrain_window) |> 
    dplyr::ungroup() |> 
    dplyr::select(-dplyr::any_of(c('type', 'retrain_window'))) |> 
    dplyr::rename_with(~ stringr::str_c(.x, "_ref"))

  if (type == 'evaluation') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |>
      dplyr::mutate(
        bias = abs(bias) / abs(bias_ref),
        mae = mae / mae_ref,
        mase = mase / mase_ref,
        mse = mse / mse_ref,
        msse = msse / msse_ref,
        rmse = rmse / rmse_ref,
        rmsse = rmsse / rmsse_ref,
        mqloss = mqloss / mqloss_ref,
        scaled_mqloss = scaled_mqloss / scaled_mqloss_ref,
        scaled_crps = scaled_crps / scaled_crps_ref
        # coverage_level50 = coverage_level50 / coverage_level50_ref,
        # coverage_level60 = coverage_level60 / coverage_level60_ref,
        # coverage_level70 = coverage_level70 / coverage_level70_ref,
        # coverage_level80 = coverage_level80 / coverage_level80_ref,
        # coverage_level90 = coverage_level90 / coverage_level90_ref,
        # coverage_level95 = coverage_level95 / coverage_level95_ref,
        # coverage_level99 = coverage_level99 / coverage_level99_ref,
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'time') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
      dplyr::mutate(
        total_fit_time = total_fit_time / total_fit_time_ref,
        total_predict_time = total_predict_time / total_predict_time_ref,
        total_sample_time = total_sample_time / total_sample_time_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'stability') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |>
      dplyr::mutate(
        stability_bias = abs(stability_bias) / abs(stability_bias_ref),
        mac = mac / mac_ref,
        masc = masc_ref,
        rmsc = rmsc / rmsc_ref,
        rmssc = rmssc_ref,
        smapc = smapc / smapc_ref,
        mqc = mqc / mqc_ref,
        smqc = smqc / smqc_ref
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else if (type == 'cost') {

    relative_data <- data |> 
      dplyr::left_join(reference_data, by = c("method" = "method_ref")) |> 
      dplyr::mutate(
  			cost_perc = cost / cost_ref * 100,
        savings = cost_ref - cost,
  			savings_perc = (cost_ref - cost) / cost_ref * 100
      ) |> 
      dplyr::select(-dplyr::ends_with("_ref"))

  } else {
    stop(paste0('Unknown type ', type))
  }

  return(relative_data)
        
}

table_retrain_results <- function(data, metric, title = "", digits = 2, format = 'numeric') {

  cat("Creating table...\n")  
  tab <- data |> 
    dplyr::select(dplyr::all_of(c('method', 'retrain_window', metric))) |> 
    tidyr::pivot_wider(names_from = 'retrain_window', values_from = metric) |> 
    dplyr::rename_with(stringr::str_to_title) |> 
    dt_table(title = title, caption = '', digits = digits, format = format)
  return(tab)

}

plot_retrain_results <- function(
		data, 
		metric, 
		scaling_fun = function(x) { scales::number(x, accuracy = 0.001) },
		metric_label = "", 
		title = "", 
		smooth = FALSE, 
		add_average = FALSE
) {

  cat("Creating plot...\n")
	
	method_lvls <- c(
		'LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 'MLP',	'LSTM', 'TCN', 'NBEATSx', 'NHITS',
		'Ens2A', 'Ens3A', 'Ens4A', 'Ens5A',	'Ens2T', 'Ens3T',	'Ens4T', 'Ens5T'
	)
	# colors_lbls <- c(
	# 	"LR" = "#003366", "RF" = "#17BECF", "XGBoost" = "#B3E5FC", "LGBM" = "#2CA02C", "CatBoost" = "#B2DF8A",   
	# 	"MLP" = "#FEE08B", "LSTM" = "#FFD700", "TCN" = "#FFA500", "NBEATSx" = "#FF6961", "NHITS" = "#E31A1C",  
	# 	"Ens2A" = "#003366", "Ens3A" = "#17BECF", "Ens4A" = "#2CA02C", "Ens5A" = "#B2DF8A",   
	# 	"Ens2T" = "#FFD700", "Ens3T" = "#FFA500",	"Ens4T" = "#FF6961", "Ens5T" = "#E31A1C"
	# )
	colors_lbls <- c(
		"LR" = "#003366", "RF" = "#17BECF", "XGBoost" = "#B3E5FC", "LGBM" = "#2CA02C", "CatBoost" = "#B2DF8A",   
		"MLP" = "#FEE08B", "LSTM" = "#FFD700", "TCN" = "#FFA500", "NBEATSx" = "#FF6961", "NHITS" = "#E31A1C",  
		"Ens2A" = "#FFB6C1", "Ens3A" = "#E754B1", "Ens4A" = "#9467BD", "Ens5A" = "#6A0DAD",
		"Ens2T" = "#F3D2B3", "Ens3T" = "#E6AB8D",	"Ens4T" = "#C97B63", "Ens5T" = "#8C564B"
	)

  data_plot <- data |> 
    dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE))
  
  g <- data_plot |> 
  	ggplot2::ggplot(
  		ggplot2::aes(
  			x = .data[['retrain_window']], 
  			y = .data[[metric]], 
  			color = .data[['method']],
  			linetype = .data[['type']],
  			group = .data[['method']]
  		)
  	)
  
  if (smooth) {
  	g <- g + 
  		ggplot2::geom_smooth(
  			method = 'lm', formula = 'y ~ log(x)', linewidth = 1, se = FALSE
  		)
  } else {
  	g <- g +  
  		ggplot2::geom_point(size = 2) +
  		ggplot2::geom_line(linewidth = 1)
  }
  
  if (add_average) {
  	
  	data_ave <- data_plot |> 
  		dplyr::group_by(retrain_window) |> 
  		dplyr::summarise('average' = mean(.data[[metric]]), .groups = 'drop') |>
  		dplyr::mutate(method = 'Average', .before = 1) |> 
  		purrr::set_names(c('method', 'retrain_window', metric))
  	
  	if (smooth)  {
  		g <- g +
  			ggplot2::geom_smooth(
  				data = data_ave, 
  				mapping = ggplot2::aes(linetype = NULL),
  				method = 'lm', formula = 'y ~ log(x)', se = FALSE,
  				col = 'black', linewidth = 0.5, linetype = 1,
  			)
  	} else {
  		g <- g +
  			ggplot2::geom_point(
  				data = data_ave, 
  				mapping = ggplot2::aes(linetype = NULL),
  				col = 'black', size = 1
  			) +
  			ggplot2::geom_line(
  				data = data_ave, 
  				mapping = ggplot2::aes(linetype = NULL),
  				col = 'black', linewidth = 0.5, linetype = 1,
  			)
  	}
  	
  }
  
  g <- g +
  	ggplot2::scale_y_continuous(labels = scaling_fun) +
  	ggplot2::scale_color_manual(values = colors_lbls) +
  	ggplot2::labs(
  		title = title, 
  		x = 'Retrain Scenario (r)', y = metric_label,
  		color = 'Method', linetype = 'Method Type', group = 'Method'
  	) + 
  	ggplot2::theme_minimal() +
  	ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
  
  return(g)

}

test_differences <- function(data, .metric, by = 'retrain_window', .method = NULL, .retrain_window = NULL) {

  if (by == 'retrain_window') {

    cat(paste0("Testing differences in ", .metric, " for method ", .method, "...\n"))
    if (is.null(.method)) {.method <- unique(data$method)[1]}
    data_test <- data |> dplyr::filter(method == .method)

    if (nrow(data_test) == 0) {
      return(NULL)
    } else {
  
      min_n_series <- min(table(data_test$retrain_window))
      n_scn <- length(unique(data_test$retrain_window))
      data_test <- data_test |> 
        dplyr::group_by(retrain_window) |> 
        dplyr::slice_sample(n = min_n_series) |> # obtain homogenous samples for each retrain window
        dplyr::ungroup() |> 
        dplyr::select(dplyr::all_of(c('retrain_window', .metric))) |> 
        dplyr::mutate(id = rep(1:min_n_series, n_scn), .before = 1) |> 
        tidyr::pivot_wider(names_from = 'retrain_window', values_from = .metric) |> 
        dplyr::select(-id)
      
      test_res <- greybox::rmcb(data = data_test, level = 0.95, outplot = "none")
      data_test <- tibble::tibble(
        'method' = as.character(.method),
        'retrain_window' = as.integer(names(test_res$mean)),
        'metric' = .metric, 
        'mean' = test_res$mean,
        'lower' = test_res$interval[, 1],
        'upper' = test_res$interval[, 2],
        'pvalue' = test_res$p.value
      )
  
      return(data_test)
  
    }

  } else if (by == 'method') {

    cat(paste0("Testing differences in ", .metric, " for retrain window ", .retrain_window, "...\n"))

    if (is.null(.retrain_window)) {.retrain_window <- min(data$retrain_window)}
    data_test <- data |> dplyr::filter(retrain_window == .retrain_window)

    if (nrow(data_test) == 0) {
      return(NULL)
    } else {
  
      min_n_series <- min(table(as.character(data_test$method)))
      n_met <- length(unique(data_test$method))
      data_test <- data_test |> 
        dplyr::group_by(method) |> 
        dplyr::slice_sample(n = min_n_series) |> # obtain homogenous samples for each retrain window
        dplyr::ungroup() |> 
        dplyr::select(dplyr::all_of(c('method', .metric))) |> 
        dplyr::mutate(id = rep(1:min_n_series, n_met), .before = 1) |> 
        tidyr::pivot_wider(names_from = 'method', values_from = .metric) |> 
        dplyr::select(-id)
      
      test_res <- greybox::rmcb(data = data_test, level = 0.95, outplot = "none")
      data_test <- tibble::tibble(
        'method' = as.character(names(test_res$mean)),
        'retrain_window' = as.integer(.retrain_window),
        'metric' = .metric, 
        'mean' = test_res$mean,
        'lower' = test_res$interval[, 1],
        'upper' = test_res$interval[, 2],
        'pvalue' = test_res$p.value
      )
  
      return(data_test)
  
    }    

  } else {
    stop(paste0('Unknown by ', by))
  }

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

plot_test_results <- function(
  data, 
  .metric, 
  by = "retrain_window", 
  .method = NULL, 
  .retrain_window = NULL, 
  metric_label = "", 
  title = ""
) {

  cat("Creating plot...\n")

	method_lvls <- c(
		'LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 'MLP',	'LSTM', 'TCN', 'NBEATSx', 'NHITS',
		'Ens2A', 'Ens3A', 'Ens4A', 'Ens5A',	'Ens2T', 'Ens3T',	'Ens4T', 'Ens5T'
	)
  
  if (by == "retrain_window") {

    if (is.null(.method)) {.method <- unique(data$method)[1]}

    data_plot <- data |> 
      dplyr::filter(testing == by) |> 
      dplyr::filter(method == .method, metric == .metric) |> 
      dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE))
    data_min <- data_plot |> dplyr::slice_min(mean)
  
    g <- data_plot |> 
      ggplot2::ggplot(ggplot2::aes(x = retrain_window, y = mean)) +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = lower, ymax = upper), 
        col = 'lightblue', width = 0.1, linewidth = 1
      ) +
      ggplot2::geom_point(size = 2, col = 'lightblue') +
      ggplot2::geom_point(data = data_min, size = 2, col = 'red') +
      ggplot2::geom_hline(yintercept = data_min$lower, col = 'gray', linetype = 2) +
      ggplot2::geom_hline(yintercept = data_min$upper, col = 'gray', linetype = 2) +
      ggplot2::labs(title = title, x = 'Retrain Scenario (r)', y = 'Rank') + 
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  } else if (by == "method") {

    if (is.null(.retrain_window)) {.retrain_window <- min(data$retrain_window)}

    data_plot <- data |> 
      dplyr::filter(testing == by) |> 
      dplyr::filter(retrain_window == .retrain_window, metric == .metric) |> 
      dplyr::mutate(method = factor(method, levels = method_lvls, ordered = TRUE))
    data_min <- data_plot |> dplyr::slice_min(mean)
  
    g <- data_plot |> 
      ggplot2::ggplot(ggplot2::aes(x = method, y = mean)) +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = lower, ymax = upper), 
        col = 'lightblue', width = 0.1, linewidth = 1
      ) +
      ggplot2::geom_point(size = 2, col = 'lightblue') +
      ggplot2::geom_point(data = data_min, size = 2, col = 'red') +
      ggplot2::geom_hline(yintercept = data_min$lower, col = 'gray', linetype = 2) +
      ggplot2::geom_hline(yintercept = data_min$upper, col = 'gray', linetype = 2) +
      ggplot2::labs(title = title, x = '', y = 'Rank') + 
      ggplot2::theme_minimal() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  } else {
    stop(paste0('Unknown by ', by))
  }
  
  return(g)

}

plot_test_results_facet <- function(
  data, 
  .metric, 
  by = "retrain_window",  
  metric_label = "", 
  title = ""
) {
	
	cat("Creating plot...\n")

	method_lvls <- c(
		'LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 'MLP',	'LSTM', 'TCN', 'NBEATSx', 'NHITS',
		'Ens2A', 'Ens3A', 'Ens4A', 'Ens5A',	'Ens2T', 'Ens3T',	'Ens4T', 'Ens5T'
	)

  if (by == "retrain_window") {

    data_plot <- data |> 
      dplyr::filter(testing == by) |> 
      dplyr::filter(metric == .metric) |> 
      dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE)) |> 
      dplyr::mutate(method = factor(method, levels = method_lvls, ordered = TRUE))
    data_min <- data_plot |> 
      dplyr::group_by(method) |> 
      dplyr::slice_min(mean) |> 
      dplyr::ungroup()
    
    g <- data_plot |> 
      ggplot2::ggplot(ggplot2::aes(x = retrain_window, y = mean)) +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = lower, ymax = upper), 
        col = 'lightblue', width = 0.1, linewidth = 1
      ) +
      ggplot2::geom_point(size = 1, col = 'lightblue') +
      ggplot2::geom_point(data = data_min, size = 1, col = 'red') +
      ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = lower), col = 'gray', linetype = 2) +
      ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = upper), col = 'gray', linetype = 2) +
      ggplot2::labs(title = title, x = 'Retrain Scenario (r)', y = 'Rank') + 
      ggplot2::facet_wrap(~ method, ncol = 2, scales = 'free_y') +
      ggplot2::theme_bw() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  } else if (by == "method") {

    data_plot <- data |> 
      dplyr::filter(testing == by) |> 
      dplyr::filter(metric == .metric) |> 
      dplyr::mutate(retrain_window = factor(retrain_window, ordered = TRUE)) |> 
      dplyr::mutate(method = factor(method, levels = method_lvls, ordered = TRUE))
    data_min <- data_plot |> 
      dplyr::group_by(retrain_window) |> 
      dplyr::slice_min(mean) |> 
      dplyr::ungroup()
    
    g <- data_plot |> 
      ggplot2::ggplot(ggplot2::aes(x = method, y = mean)) +
      ggplot2::geom_errorbar(
        ggplot2::aes(ymin = lower, ymax = upper), 
        col = 'lightblue', width = 0.1, linewidth = 1
      ) +
      ggplot2::geom_point(size = 1, col = 'lightblue') +
      ggplot2::geom_point(data = data_min, size = 1, col = 'red') +
      ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = lower), col = 'gray', linetype = 2) +
      ggplot2::geom_hline(data = data_min, mapping = ggplot2::aes(yintercept = upper), col = 'gray', linetype = 2) +
      ggplot2::labs(title = title, x = 'Method', y = 'Rank') + 
      ggplot2::facet_wrap(~ retrain_window, ncol = 2, scales = 'free_y') +
      ggplot2::theme_bw() +
      ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))

  } else {
    stop(paste0('Unknown by ', by))
  }
	
	return(g)
	
}

plot_distribution_results <- function(data, metric, metric_label = "", title = "") {
	
	cat("Creating plot...\n")
	retrain_scenario <- sort(unique(data[["retrain_window"]]))
	
	g <- data |> 
		ggplot2::ggplot(
			ggplot2::aes(
				# x = .data[['retrain_window']], 
				x = .data[[metric]], 
				color = .data[['retrain_window']] |> factor(levels = retrain_scenario, ordered = TRUE)
			)
		) +
		ggplot2::geom_density() +
		ggplot2::facet_wrap(~ .data[['method']], nrow = 4, ncol = 2) + 
		# ggplot2::scale_x_continuous(breaks = retrain_scenario) +
		ggplot2::labs(
			title = title, 
			x = 'Retrain Scenario (r)', y = metric_label,
			color = 'Retrain Scenarios'
		) + 
		# ggplot2::theme_minimal() +
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

compute_costs <- function(
  data, 
  time_var, 
  n_skus, 
  dataset_n_skus, 
  cost_per_hour = 3.5, 
  add_average = FALSE,
  compute_relative_costs = TRUE
) {
	
  if (add_average) {
    data_mean <- data |> 
      aggregate_data(
        group_columns = c('retrain_window'),
        drop_columns = c('type', 'method', 'sample', 'test_window', 'horizon'),
        function_name = 'mean',
        adjust_metrics = FALSE
      ) |> 
      dplyr::mutate(method = 'Average', .before = 1) |> 
      dplyr::mutate(type = NA_character_, .before = 1)
    data_cost <- data |> dplyr::bind_rows(data_mean)
  } else {
    data_cost <- data
  }

  data_cost <- data_cost |>  
		dplyr::mutate(
			ct_per_sku = .data[[time_var]] / dataset_n_skus,
			ct_hour = .data[[time_var]] / 60 / 60,
			ct_hour_per_sku = ct_hour / dataset_n_skus,
			ct_hour_tot = ct_hour_per_sku * n_skus,
			cost = ct_hour_tot * cost_per_hour
		) |> 
		dplyr::select(dplyr::all_of(c('type', 'method', 'retrain_window', 'cost'))) 
  
  if (compute_relative_costs) {
    data_cost <- data_cost |> 
      compute_relative_metrics(type = 'cost')
  }
	
	return (data_cost)
	
}

flatten_list <- function(list) {
  return(purrr::map_chr(list, ~ paste0(.x, collapse = "_")))
}

create_results_list <- function(dataset_names, analysis_types, data_types, model_types, final_analyses) {

  dataset_names <- flatten_list(dataset_names)
  analysis_types <- flatten_list(analysis_types)
  data_types <- flatten_list(data_types)
  model_types <- flatten_list(model_types)
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
      res_list[[i]][[j]][['results']] <- vector("list", length(model_types)) |> 
        purrr::set_names(model_types)
    }
  }
  for (i in seq_along(res_list)) {
    for (j in seq_along(res_list[[i]])) {
      for (k in seq_along(res_list[[i]][[j]][['results']])) {
        res_list[[i]][[j]][['results']][[k]] <- vector("list", length(final_analyses)) |> 
          purrr::set_names(final_analyses)
      }
    }
  }

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

analyze_results <- function(config) {

  # analysis config
  analysis_name <- config$analysis$name
  analysis_types <- config$analysis$types
  analysis_method <- config$analysis$method
  analysis_sample_type <- config$analysis$sample_type
  # dataset config
  dataset_names <- config$dataset$dataset_names
  frequencies <- config$dataset$frequencies
  dataset_names_full <- paste(dataset_names, frequencies, sep = "_")
  retrain_scenarios <- purrr::map(frequencies, get_retrain_scenarios) |> 
    purrr::set_names(dataset_names_full)
  ext <- config$dataset$ext
  # model config
  model_types <- config$models$types
  model_names <- config$models$model_names
  model_names_abbr <- config$models$model_names_abbr
  model_type_levels <- unlist(model_types) # c('SF', 'ML', 'DL', 'ENSACC', 'ENSTIME')
  # evaluation, time, stability and cost config
  eval_metrics <- config$evaluation_params$metrics
  eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
  eval_outlier_cleaning_quantiles <- config$evaluation_params$outlier_cleaning_quantiles
  time_metrics <- config$time_params$metrics
  stab_metrics <- config$stability_params$metrics
  stab_outlier_cleaning_metrics <- config$stability_params$outlier_cleaning_metrics
  stab_outlier_cleaning_quantiles <- config$stability_params$outlier_cleaning_quantiles
  cost_metrics <- config$cost_params$metrics
  cost_time_var <- config$cost_params$time_var
  cost_n_skus <- config$cost_params$n_skus
  cost_per_hour <- config$cost_params$cost_per_hour
  cost_datasets_n_skus <- as.list(unlist(config$cost_params$cost_datasets_n_skus))
  
  # final analysis names
  data_types <- c('data', 'results')
  final_analyses <- c('tables', 'plots', 'tests')  
  analysis_results <- create_results_list(
    dataset_names_full, analysis_types, data_types, model_types, final_analyses
  )
  
  for (dn in dataset_names_full) {

    cat(paste0("*************** Analysing ", dn, " ***************\n"))
    dataset_name_tmp <- unlist(strsplit(dn, "_"))[1]
    freq_tmp <- unlist(strsplit(dn, "_"))[2]
    retrain_scn_tmp <- retrain_scenarios[[dn]]

    for (at in analysis_types) {

      cat(paste0("--- [ Analysis: ", at, " ] ---\n"))

      if (at == "evaluation") {

        cat("Loading and preparing the evaluation data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'eval', analysis_sample_type),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr) |> 
          clean_outliers(.metric = eval_outlier_cleaning_metrics, q = eval_outlier_cleaning_quantiles)
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('unique_id', 'test_window', 'horizon'),
            function_name = 'mean',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- eval_metrics

      } else if (at == "time") {

        cat("Loading and preparing the time data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'time'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr)
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('sample', 'test_window', 'horizon'),
            function_name = 'sum',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- time_metrics

      } else if (at == "stability") {

        cat("Loading and preparing the stability data...\n")
        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'stability'),
          name_list = c(dataset_name_tmp, freq_tmp, 'stab'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr) |> 
          clean_outliers(.metric = stab_outlier_cleaning_metrics, q = stab_outlier_cleaning_quantiles) |> 
        	dplyr::filter(type != 'ENSTIME') # remove ensemble time from stability analysis
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('unique_id', 'test_window', 'horizon'),
            function_name = 'mean',
            adjust_metrics = TRUE
          )
        # set the metrics to be used
        anal_metrics <- stab_metrics

      } else if (at == "cost") {

        cat("Loading and preparing the time data for cost analysis...\n")
        cost_dataset_n_skus_tmp <- cost_datasets_n_skus[[dn]]

        anal_df_tmp = load_data(
          path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
          name_list = c(dataset_name_tmp, freq_tmp, 'time'),
          ext = ext
        ) |> 
          tibble::as_tibble() |> 
          dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
          dplyr::filter(method %in% model_names) |> 
          recode_data(model_type_levels, model_names_abbr)
        anal_df_agg_tmp <- anal_df_tmp |> 
          aggregate_data(
            group_columns = c('type', 'method', 'retrain_window'),
            drop_columns = c('sample', 'test_window', 'horizon'),
            function_name = 'sum',
            adjust_metrics = TRUE
          ) |> 
          compute_costs(
            time_var = cost_time_var, 
            n_skus = cost_n_skus,
            dataset_n_skus = cost_dataset_n_skus_tmp,
            cost_per_hour = cost_per_hour, 
            add_average = FALSE
          )
        anal_df_tmp <- anal_df_tmp |>
          compute_costs(
            time_var = cost_time_var, 
            n_skus = cost_n_skus,
            dataset_n_skus = cost_dataset_n_skus_tmp,
            cost_per_hour = cost_per_hour, 
            add_average = FALSE,
            compute_relative_costs = FALSE
          )
        # set the metrics to be used
        anal_metrics <- cost_metrics

      } else {
        stop(paste0("Unknown analysis type: ", at))
      }

      # absolute or relative analysis
      if (analysis_method == 'relative' & at != 'cost') {
        cat("Compute relative metrics...\n")
        anal_df_agg_tmp <- compute_relative_metrics(anal_df_agg_tmp, type = at)
      }

      # store data of the analysis
      analysis_results[[dn]][[at]][['data']] <- anal_df_agg_tmp

      for (mt in model_types) {

        cat(paste0("--- [ Model Types: ", paste0(mt, collapse = ", "), " ] ---\n"))
        # filter datasets
        anal_df_mt_tmp <- anal_df_tmp |> dplyr::filter(type %in% mt)
        anal_df_agg_mt_tmp <- anal_df_agg_tmp |> dplyr::filter(type %in% mt)
        model_names_abbr_mt_tmp <- unique(as.character(anal_df_mt_tmp$method))

        # analysis tables, plots and tests
        anal_tab <- anal_plot <- anal_test <- vector("list", length(anal_metrics)) |> 
          purrr::set_names(anal_metrics)

        for (am in anal_metrics) {

          cat(paste0("Creating evaluation table, plot and tests for ", toupper(am), "...\n"))
          tp_par <- get_table_plot_params(am, analysis_method)

          anal_tab[[am]] <- table_retrain_results(
            anal_df_agg_mt_tmp, 
            metric = am,
            title = toupper(paste(dataset_name_tmp, '-', tp_par$label)),
            digits = tp_par$digits,
            format = tp_par$format
          )

          anal_plot[[am]] <- plot_retrain_results(
            anal_df_agg_mt_tmp, 
            metric = am, 
            scaling_fun = tp_par$scaling_fun,
            metric_label = tp_par$label,
            title = toupper(dataset_name_tmp),
            add_average = tp_par$add_average
          )

          if (!am %in% c('savings', 'savings_perc')) {
            anal_test[[am]] <- dplyr::bind_rows(
              model_names_abbr_mt_tmp |> 
                purrr::map(
                  ~ test_differences(
                    anal_df_mt_tmp, 
                    .metric = am, 
                    by = "retrain_window",
                    .method = .x, 
                  )
                ) |> 
                dplyr::bind_rows() |> 
                dplyr::mutate("testing" = "retrain_window", .before = 1),
              retrain_scn_tmp |> 
                purrr::map(
                  ~ test_differences(
                    anal_df_mt_tmp, 
                    .metric = am, 
                    by = "method",
                    .retrain_window = .x, 
                  )
                ) |> 
                dplyr::bind_rows() |> 
                  dplyr::mutate("testing" = "method", .before = 1)
            )
          } else {
            anal_test[[am]] <- NULL
          }

        }

        # store results
        mt_name <- paste0(mt, collapse = "_")
        analysis_results[[dn]][[at]][['results']][[mt_name]][['tables']] <- anal_tab 
        analysis_results[[dn]][[at]][['results']][[mt_name]][['plots']] <- anal_plot 
        analysis_results[[dn]][[at]][['results']][[mt_name]][['tests']] <- anal_test 

      }
      
    }

  }

  cat("Saving results...\n")
  file_name <- paste0(
    analysis_method,
    "_",
    stringr::str_sub_all(analysis_types, start = 1, end = 4) |> 
      unlist() |> 
      paste0(collapse = ""),  
    "_",
    analysis_sample_type, 
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

plot_compared_results <- function(
		data, 
		metric, 
		.retrain_window,
		analysis_method = 'absolute',
		# format = "numeric",
		# metric_label = "", 
		title = ""
) {
	
	cat("Creating plot...\n")
	
	colors_lbls_2 <- c("ML" = "#17BECF", "DL" = "#FFA500", "ENSACC" = "#E754B1", "ENSTIME" = "#C97B63")
	
	params <- get_table_plot_params(metric, analysis_method)
	
	data_plot <- data |> 
		dplyr::filter(retrain_window == .retrain_window) |> 
		dplyr::select(dplyr::all_of(c('type', 'method', metric)))
	
	g <- data_plot |> 
		ggplot2::ggplot(
			ggplot2::aes(
				x = .data[['method']], 
				y = .data[[metric]], 
				fill = .data[['type']]
			)
		)
	
	g <- g + ggplot2::geom_bar(stat = "identity", position = "dodge")
	
	g <- g +
		ggplot2::geom_text(
			ggplot2::aes(label = params$scaling_fun(.data[[metric]])), 
			position = ggplot2::position_dodge(width = 0.9), 
			vjust = -0.25
		) + 
		ggplot2::scale_y_continuous(labels = params$scaling_fun) +
		ggplot2::scale_fill_manual(values = colors_lbls_2) +
		ggplot2::labs(
			title = title, 
			x = '', y = params$label, fill = 'Method Type'
		) + 
		ggplot2::theme_minimal() +
		ggplot2::theme(
			plot.title = ggplot2::element_text(hjust = 0.5),
			legend.position = "bottom",
			axis.text.x = ggplot2::element_text(angle = 45, hjust = 1)
		)
	
	return(g)
	
}

plot_scatter_results <- function(
		data, 
		metrics, 
		.retrain_window = NULL,
		analysis_method = 'absolute',
		title = ""
) {
	
	cat("Creating plot...\n")
	
	colors_lbls_2 <- c("ML" = "#17BECF", "DL" = "#FFA500", "ENSACC" = "#E754B1", "ENSTIME" = "#C97B63")
	
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

analyze_optimal_frequency <- function(config, adjust = 1) {
	
	# analysis config
	analysis_name <- config$analysis$name
	analysis_types <- config$analysis$types 
	analysis_method <- config$analysis$method
	analysis_sample_type <- config$analysis$sample_type
	# dataset config
	dataset_names <- config$dataset$dataset_names
	frequencies <- config$dataset$frequencies
	dataset_names_full <- paste(dataset_names, frequencies, sep = "_")
	retrain_scenarios <- purrr::map(frequencies, get_retrain_scenarios) |> 
		purrr::set_names(dataset_names_full)
	ext <- config$dataset$ext
	# model config
	model_types <- config$models$types
	model_names <- config$models$model_names
	model_names_abbr <- config$models$model_names_abbr
	model_type_levels <- unlist(model_types) # c('SF', 'ML', 'DL', 'ENSACC', 'ENSTIME')
	# evaluation, and stability config
	eval_metrics <- config$evaluation_params$metrics
	eval_outlier_cleaning_metrics <- config$evaluation_params$outlier_cleaning_metrics
	eval_outlier_cleaning_quantiles <- config$evaluation_params$outlier_cleaning_quantiles
	stab_metrics <- config$stability_params$metrics
	stab_outlier_cleaning_metrics <- config$stability_params$outlier_cleaning_metrics
	stab_outlier_cleaning_quantiles <- config$stability_params$outlier_cleaning_quantiles
	
	# final analysis names
	data_types <- c('results')
	final_analyses <- c('plots')  
	analysis_results <- create_results_list(
		dataset_names_full, analysis_types, data_types, model_types, final_analyses
	)
	
	for (dn in dataset_names_full) {
		
		cat(paste0("*************** Analysing ", dn, " ***************\n"))
		dataset_name_tmp <- unlist(strsplit(dn, "_"))[1]
		freq_tmp <- unlist(strsplit(dn, "_"))[2]
		retrain_scn_tmp <- retrain_scenarios[[dn]]
		
		for (at in analysis_types) {
			
			cat(paste0("--- [ Analysis: ", at, " ] ---\n"))
			
			if (at == "evaluation") {
				
				cat("Loading and preparing the evaluation data...\n")
				anal_df_tmp = load_data(
					path_list = c('results', dataset_name_tmp, freq_tmp, 'evaluation'),
					name_list = c(dataset_name_tmp, freq_tmp, 'eval', analysis_sample_type),
					ext = ext
				) |> 
					tibble::as_tibble() |> 
					dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
					dplyr::filter(method %in% model_names) |> 
					recode_data(model_type_levels, model_names_abbr) |> 
					clean_outliers(.metric = eval_outlier_cleaning_metrics, q = eval_outlier_cleaning_quantiles)
				# set the metrics to be used
				anal_metrics <- eval_metrics
				
			} else if (at == "stability") {
				
				cat("Loading and preparing the stability data...\n")
				anal_df_tmp = load_data(
					path_list = c('results', dataset_name_tmp, freq_tmp, 'stability'),
					name_list = c(dataset_name_tmp, freq_tmp, 'stab'),
					ext = ext
				) |> 
					tibble::as_tibble() |> 
					dplyr::filter(retrain_window %in% retrain_scn_tmp) |> 
					dplyr::filter(method %in% model_names) |> 
					recode_data(model_type_levels, model_names_abbr) |> 
					clean_outliers(.metric = stab_outlier_cleaning_metrics, q = stab_outlier_cleaning_quantiles) |> 
					dplyr::filter(type != 'ENSTIME') # remove ensemble time from stability analysis
				# set the metrics to be used
				anal_metrics <- stab_metrics
				
			} else if (at == "time" | at == "cost") {
				next
			}	else {
				stop(paste0("Unknown analysis type: ", at))
			}
			
			for (mt in model_types) {
				
				cat(paste0("--- [ Model Types: ", paste0(mt, collapse = ", "), " ] ---\n"))
				# filter datasets
				anal_df_mt_tmp <- anal_df_tmp |> dplyr::filter(type %in% mt)
				model_names_abbr_mt_tmp <- unique(as.character(anal_df_mt_tmp$method))
				
				# analysis tables, plots and tests
				anal_plot <- vector("list", length(anal_metrics)) |> 
					purrr::set_names(anal_metrics)
				
				for (am in anal_metrics) {
					
					cat(paste0("Creating evaluation table, plot and tests for ", toupper(am), "...\n"))
					tp_par <- get_table_plot_params(am, analysis_method)
					
					anal_df_mt_optimal_tmp <- anal_df_mt_tmp |> 
						dplyr::select(dplyr::all_of(c('type', 'method', 'retrain_window', 'unique_id', am))) |> 
						dplyr::group_by(type, method, unique_id) |> 
						dplyr::arrange(type, method, unique_id, .data[[am]]) |> 
						dplyr::slice_head(n = 1) |> 
						dplyr::ungroup()
					
					anal_plot[[am]] <- list(
						"overall" = plot_optimal_retrain_results(
							anal_df_mt_optimal_tmp, 
							metric = am, 
							title = toupper(dataset_name_tmp),
							overall_only = TRUE,
							adjust = adjust
						),
						"bymethod" = plot_optimal_retrain_results(
							anal_df_mt_optimal_tmp, 
							metric = am, 
							title = toupper(dataset_name_tmp),
							overall_only = FALSE,
							adjust = adjust
						)
					)
					
				}
				
				# store results
				mt_name <- paste0(mt, collapse = "_")
				analysis_results[[dn]][[at]][['results']][[mt_name]][['plots']] <- anal_plot 
				
			}
			
		}
		
	}
	
	cat("Done!\n")
	return(invisible(analysis_results))
	
}

plot_optimal_retrain_results <- function(
		data, 
		metric, 
		title = "", 
		overall_only = TRUE,
		adjust = 1
) {
	
	cat("Creating plot...\n")
	
	method_lvls <- c(
		'LR', 'RF', 'XGBoost', 'LGBM', 'CatBoost', 'MLP',	'LSTM', 'TCN', 'NBEATSx', 'NHITS',
		'Ens2A', 'Ens3A', 'Ens4A', 'Ens5A',	'Ens2T', 'Ens3T',	'Ens4T', 'Ens5T'
	)
	colors_lbls <- c(
		"LR" = "#003366", "RF" = "#17BECF", "XGBoost" = "#B3E5FC", "LGBM" = "#2CA02C", "CatBoost" = "#B2DF8A",   
		"MLP" = "#FEE08B", "LSTM" = "#FFD700", "TCN" = "#FFA500", "NBEATSx" = "#FF6961", "NHITS" = "#E31A1C",  
		"Ens2A" = "#FFB6C1", "Ens3A" = "#E754B1", "Ens4A" = "#9467BD", "Ens5A" = "#6A0DAD",
		"Ens2T" = "#F3D2B3", "Ens3T" = "#E6AB8D",	"Ens4T" = "#C97B63", "Ens5T" = "#8C564B"
	)
	
	data_plot <- data |> 
		dplyr::mutate(method = factor(method, levels = method_lvls, ordered = FALSE))
	
	if (min(data$retrain_window) == 7) {
		brks <- c(7, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
		lim <- c(1, 365)
	} else {
		brks <- c(1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52)
		lim <- c(1, 53)
	}
	
	if (overall_only) {
		data_plot <- data_plot |> 
			dplyr::mutate(type = 'Overall', method = 'Overall')
		g <- data_plot |>
			ggplot2::ggplot(
				ggplot2::aes(
					x = .data[['retrain_window']],
					y = ggplot2::after_stat(scaled)
				)
			)
		g <- g + ggplot2::geom_density(adjust = adjust)
	} else {
		g <- data_plot |> 
			ggplot2::ggplot(
				ggplot2::aes(
					x = .data[['retrain_window']],
					y = ggplot2::after_stat(scaled), 
					color = .data[['method']],
					linetype = .data[['type']],
					group = .data[['method']],
					key_glyph = "line"
				)
			)
		g <- g + 
			ggplot2::geom_density(adjust = adjust, show.legend = FALSE)	+
			ggplot2::stat_density(geom = "line", position = "identity", adjust = adjust)
	}
	
	g <- g +
		ggplot2::scale_x_continuous(breaks = brks, limits = lim) +
		ggplot2::scale_color_manual(values = colors_lbls) +
		ggplot2::labs(
			title = title, 
			x = 'Retrain Scenario (r)', y = 'Density',
			color = 'Method', linetype = 'Method Type', group = 'Method'
		) + 
		ggplot2::theme_minimal() +
		ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5))
	
	if (!overall_only) {
		g <- g + 
			ggplot2::facet_wrap(. ~ method, scales = 'free_x', ncol = 2) +
			ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45))
	}
	
	return(g)
	
}
