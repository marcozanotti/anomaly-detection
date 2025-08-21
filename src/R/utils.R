
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
    function_name = 'median'
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
