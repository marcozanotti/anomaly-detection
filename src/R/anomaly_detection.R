#' @name anomaly_detection
#'
#' @title Anomaly Detection
#'
#' @description Anomaly detection
#'
#' @param data Data
#' @param date Date
#' @param dates Dates
#' @param methods Methods
#' @param score Whether to compute the score
#' @param methods_ranking Ranking of methods for weighted score
#' @param anomaly_data Anomaly data
#' @param weights Weights for the weighted score
#' @param anomaly_score Anomaly detection score
#' @param threshold Threshold to identify anomaly on scores
#' @param indexes Whether to return indexes of anomalies
#' @param sources Sources
#' @param last_only Last only
#' @param frequency Frequency
#' @param period Period
#' @param score_type Score type
#' @param check_nrows Whether to check data on consistent number of rows
#'
#' @return NULL
NULL


#' @describeIn anomaly_detection Detect anomalies into time series data
#' @export
anomaly_detection <- function(
	data, 
	dates, 
	methods = c("forecast", "anomalize", "stray", "otsad", "anomaly"),
	score = TRUE,
	methods_ranking = NULL
) {
	
	# Initial set up of methods
	# to add a method:
	# 1) increase the num_tot_methods variable
	# 2) add the method name into available_methods variable
	# 3) add the method sub name to full_methods_names
	# 3) apply the method based on if-else conditionals
	data_tbl <- dplyr::tibble("datetime" = dates, "value" = data)
	n_tot_methods <- 10
	available_methods <- c(
		"forecast", "anomalize", "tsoutliers", "otsad", 
		"otsad_knn", "anomaly", "stray"
	)
	full_methods_names <- c(
		"forecast", "anomalize", "tsoutliers", 
		"otsad_cpp", "otsad_cpsd", "otsad_cpts",
		"otsad_knn", 
		"anomaly_capa", "anomaly_scapa", 
		"stray"
	)
	
	zero_vector <- vector("numeric", length(data))
	res_list <- purrr::map(seq_len(n_tot_methods), ~ rep(zero_vector, 1)) %>% 
		purrr::set_names(full_methods_names)
	
	apply_methods <- intersect(methods, available_methods)
	
	# forecast
	if ("forecast" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method forecast...")
		out <- forecast::tsoutliers(x = data)$index
		res_list[["forecast"]][out] <- 1
	} else {
		res_list[["forecast"]] <- NULL
	}
	
	# anomalize
	if ("anomalize" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method anomalize...")
		out <- data_tbl %>%  
			anomalize::time_decompose(target = "value", method = "twitter", message = FALSE) %>% 
			anomalize::anomalize(target = "remainder", method = "gesd") %>% 
			dplyr::pull("anomaly")
		res_list[["anomalize"]][which(out == "Yes")] <- 1
	} else {
		res_list[["anomalize"]] <- NULL
	}
	
	# tsoutliers
	if ("tsoutliers" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method tsoutliers...")
		data_ts <- stats::ts(data) 
		out <- tsoutliers::tso(y = data_ts)$outliers$ind
		res_list[["tsoutliers"]][out] <- 1
	} else {
		res_list[["tsoutliers"]] <- NULL
	}
	
	# otsad
	if ("otsad" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method otsad...")
		n_train <- otsad::GetNumTrainingValues(length(data))
		# CpP
		out <- otsad::CpPewma(data, n_train)
		res_list[["otsad_cpp"]][which(out$is.anomaly == 1)] <- 1
		# CpSd
		out <- otsad::CpSdEwma(data, n_train)
		res_list[["otsad_cpsd"]][which(out$is.anomaly == 1)] <- 1
		# CpTs
		out <- otsad::CpTsSdEwma(data, n_train)
		res_list[["otsad_cpts"]][which(out$is.anomaly == 1)] <- 1
	} else {
		res_list[["otsad_cpp"]] <- NULL # CpP
		res_list[["otsad_cpsd"]] <- NULL # CpSd
		res_list[["otsad_cpts"]] <- NULL # CpTs
	}
	
	# otsad_knn
	if ("otsad_knn" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method otsad knn...")
		n_train <- otsad::GetNumTrainingValues(length(data))
		k_groups <- length(data) * 0.1 # 10% of data points taken into account
		out <- otsad::CpKnnCad(data, n_train, threshold = 0.95, k = k_groups)
		res_list[["otsad_knn"]][which(out$is.anomaly == 1)] <- 1
	} else {
		res_list[["otsad_knn"]] <- NULL
	}
	
	# anomaly
	if ("anomaly" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method anomaly...")
		check <- check_constant_variable(data)
		if (check == 1) {
			logging::loginfo("Skip calculations through method anomaly because of constant variable...")
			res_list[["anomaly_capa"]] <- NULL # CAPA
			res_list[["anomaly_scapa"]] <- NULL # SCAPA
		} else {
			# CAPA
			out <- anomaly::point_anomalies(anomaly::capa.uv(data, transform = scale)) # base scale function as transform to avoid NaN problems
			res_list[["anomaly_capa"]][out[out$strength > 0, "location"]] <- 1
			# SCAPA
			out <- anomaly::point_anomalies(anomaly::scapa.uv(data, transform = scale)) # base scale function as transform to avoid NaN problems
			res_list[["anomaly_capa"]][out[out$strength > 0, "location"]] <- 1
		}
	} else {
		res_list[["anomaly_capa"]] <- NULL # CAPA
		res_list[["anomaly_scapa"]] <- NULL # SCAPA
	}
	
	# stray
	if ("stray" %in% apply_methods) {
		logging::loginfo("Detecting anomalies through method stray...")
		n_train <- otsad::GetNumTrainingValues(length(data))
		k_groups <- length(data) * 0.1 # 10% of data points taken into account
		out <- stray::find_HDoutliers(
			data, 
			k = k_groups,	knnsearchtype = "kd_tree",
			alpha = .05, p = .05, tn = n_train
		)$outliers
		res_list[["stray"]][out] <- 1
	} else {
		res_list[["stray"]] <- NULL
	}
	
	
	res_df <- dplyr::bind_cols(res_list)
	
	
	if (score) {
		
		score <- anomaly_score(res_df)
		if (is.null(methods_ranking)) {methods_ranking <- methods}
		ws <- rev(seq_along(res_df)) / sum(seq_along(res_df)) # compute weights as ranking
		res_df <- dplyr::select(res_df, dplyr::contains(methods_ranking)) # arrange columns
		score_w <- anomaly_score(res_df, ws)
		
		res_df$score <- score
		res_df$score_w <- score_w
		res_df <- dplyr::select(
			res_df, 
			"score", dplyr::contains("score_w"), dplyr::everything()
		)
		
	}
	
	res_df <- dplyr::bind_cols(data_tbl, res_df)
	
	
	return(res_df)
	
	
}


#' @describeIn anomaly_detection Compute anomaly score
#' @export
anomaly_score <- function(anomaly_data, weights = NULL) {
	
	if (is.null(weights)) {
		logging::loginfo("Computing anomaly score...")
		score <- rowSums(anomaly_data) / ncol(anomaly_data)
	} else {
		logging::loginfo("Computing anomaly weighted score...")
		score <- rowSums(purrr::map2_df(anomaly_data, weights, ~ .x * .y))
	}
	return(score)
	
}


#' @describeIn anomaly_detection Select anomaly based on score and threshold
#' @export
anomaly_select <- function(anomaly_score, threshold = 0.5, indexes = FALSE) {
	
	logging::loginfo("Selecting anomalies based on anomaly score...")
	idx <- (anomaly_score > threshold)
	if (indexes) {idx <- which(idx)}
	idx <- as.numeric(idx)
	return(idx)
	
}


#' @describeIn anomaly_detection Wrapper to compute anomalies on sl project data
#' @export
get_anomaly <- function(
	data,
	frequency = getOption("anomaly.detection.frequency"),
	period = getOption("anomaly.detection.period"),
	sources = getOption("anomaly.detection.source"), # for future developments on single source
	methods = getOption("anomaly.detection.anomaly_methods"),
	threshold = getOption("anomaly.detection.anomaly_threshold"), 
	score_type = getOption("anomaly.detection.anomaly_score_type"),
	last_only = FALSE,
	check_nrows = TRUE
) {
	
	if (check_nrows) {
		n_obs <- generate_nobs(frequency, period)
		if (nrow(data) < n_obs) {
			stop(paste0("Data has ", nrow(data), " observations, but ", n_obs, " are required."))
		}
	} else {
		n_obs <- nrow(data)
	}
	
	data <- dplyr::slice_tail(data, n = n_obs) 
	datetimes <- dplyr::pull(data, "datetime")
	if (is.null(sources)) {
		values <- dplyr::select(data, -"datetime")
	} else {
		values <- dplyr::select(data, dplyr::all_of(sources))
	}
	
	# Compute anomaly detection scores
	scores <- purrr::map(
		values, 
		anomaly_detection, 
		dates = datetimes, 
		methods = methods
	)
	scores <- dplyr::bind_cols(
		list("datetime" = data$datetime), 
		purrr::map(scores, score_type)
	)
	
	# Select outliers based on weighted score
	outliers <- purrr::map(
		dplyr::select(scores, -"datetime"), 
		anomaly_select, 
		threshold = threshold
	)
	outliers <- dplyr::bind_cols(
		list("datetime" = data$datetime), 
		outliers
	)
	
	if (last_only) {
		scores <- scores[nrow(scores), ]
		outliers <- outliers[nrow(outliers), ]
	}
	
	res <- list("scores" = scores, "outliers" = outliers)
	
	return(res)
	
}


#' @describeIn anomaly_detection Check anomalies
#' @export
check_anomaly <- function(data, date) {
	
	anomaly_data <- dplyr::filter(data, datetime == date)
	if (nrow(anomaly_data) == 0) {
		logging::loginfo("No data for selected period.")
		return(invisible(NULL))
	} 
	
	anomaly <- anomaly_data %>% 
		dplyr::select(-"datetime") %>% 
		rowSums()
	if (anomaly == 0) {
		res <- 0
	} else {
		res <- 1
	}
	
	return(res)
	
}


#' @describeIn anomaly_detection Verify the anomaly type
#' @export
anomaly_type <- function(data, date) {
	
	sources <- data %>% 
		dplyr::filter(datetime == date) %>% 
		dplyr::select(-dplyr::contains("datetime")) %>% 
		tidyr::pivot_longer(cols = dplyr::everything()) %>% 
		dplyr::filter(value > 0) %>% 
		dplyr::pull("name")
	
	text <- paste(sources, collapse = ", ")

	return(invisible(text))	
	
}


#' @describeIn anomaly_detection Collect the anomalies
#' @export
collect_anomaly <- function(data,	date) {
	
	check_res <- check_anomaly(data, date)
	
	if (is.null(check_res)) {
		return(invisible(NULL))
	}
	
	if (check_res == 0) {
		logging::loginfo(paste0("No anomalies on ", date))
		anomaly_data <- NULL
	} else {
		logging::loginfo(paste0("Collecting anomalies on ", date))
		anomaly_data <- data %>% 
			dplyr::select("datetime") %>% 
			dplyr::filter(datetime == date) %>% 
			dplyr::mutate(type = anomaly_type(data, date)) 
	}
	
	return(invisible(anomaly_data))
	
}
