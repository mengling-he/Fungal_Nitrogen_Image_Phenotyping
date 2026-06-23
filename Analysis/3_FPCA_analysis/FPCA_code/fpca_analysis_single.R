# FPCA single-trait analysis (refactored)
#
# This file defines a reusable function `run_fpca_trait()`.
# It depends on: tuning_nointer.R, pca_fun.R, pca_score.R (same folder).

suppressPackageStartupMessages({
  library(fda)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(lmerTest)
  library(emmeans)
  library(patchwork)
})
get_script_dir <- function() {
  # 1) When sourced: this works
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  fileArg <- "--file="
  path <- sub(fileArg, "", cmdArgs[grep(fileArg, cmdArgs)])
  if (length(path) > 0 && file.exists(path)) return(normalizePath(dirname(path)))
  
  # 2) When sourced in interactive session: use sys.frames
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(dirname(sys.frames()[[1]]$ofile)))
  }
  
  # 3) RStudio fallback
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    ctx <- rstudioapi::getActiveDocumentContext()
    if (nzchar(ctx$path)) return(normalizePath(dirname(ctx$path)))
  }
  
  # 4) last resort
  return(getwd())
}

CODE_DIR <- get_script_dir()

source(file.path(CODE_DIR, "tuning_nointer.R"))
source(file.path(CODE_DIR, "pca_fun.R"))
source(file.path(CODE_DIR, "pca_score.R"))

# ---- helper: robust numeric day_id from date ----
.ensure_day_id <- function(df, date_col = 'date', time_col = 'day_id') {
  if (!time_col %in% names(df)) {
    if (!date_col %in% names(df)) stop('Neither time_col nor date_col exist in df.')
    df[[date_col]] <- as.Date(df[[date_col]])
    df[[time_col]] <- as.numeric(difftime(df[[date_col]], min(df[[date_col]], na.rm = TRUE), units = 'days'))
  }
  df
}

# ---- helper: sign convention for eigenfunctions ----
.enforce_phi_sign <- function(phi_grid, scores = NULL) {
  # Make phi(midpoint) positive for each component; flip scores accordingly.
  mid <- which.min(abs(seq_len(nrow(phi_grid)) - (nrow(phi_grid) + 1) / 2))
  flips <- rep(1, ncol(phi_grid))
  for (k in seq_len(ncol(phi_grid))) {
    if (is.finite(phi_grid[mid, k]) && phi_grid[mid, k] < 0) flips[k] <- -1
  }
  phi_grid2 <- sweep(phi_grid, 2, flips, `*`)
  if (!is.null(scores)) {
    scores2 <- scores
    for (k in seq_len(min(ncol(scores2), length(flips)))) scores2[, k] <- scores2[, k] * flips[k]
    return(list(phi = phi_grid2, scores = scores2, flips = flips))
  }
  list(phi = phi_grid2, flips = flips)
}

# ---- helper: parse plate ID into metadata columns (optional) ----
.parse_id_underscore <- function(df_scores, id_col = 'plate', into = c('Fungal_Strain','Nitrogen_Level','Replication')) {
  if (!id_col %in% names(df_scores)) return(df_scores)
  # Keep only the last 3 underscore-separated parts by default, matching your original pattern.
  parts <- max(length(into) + 2, 5)
  tmp <- df_scores %>%
    tidyr::separate(
      .data[[id_col]],
      into = paste0('..p', seq_len(parts)),
      sep = '_',
      remove = FALSE,
      fill = 'right',
      extra = 'merge'
    )
  # Map the 2..(k+1)th positions similar to your prior: col1, Fungal, Nitrogen, Replication, col5
  # If structure differs, user can disable parsing.
  if (parts >= 5 && length(into) == 3) {
    tmp[[into[1]]] <- tmp[['..p2']]
    tmp[[into[2]]] <- tmp[['..p3']]
    tmp[[into[3]]] <- tmp[['..p4']]
    tmp <- tmp %>% dplyr::select(-starts_with('..p'))
  } else {
    tmp <- tmp %>% dplyr::select(-starts_with('..p'))
  }
  tmp
}

# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
run_fpca_trait <- function(
  df,
  trait,
  id_col = 'plate',
  time_col = 'day_id',
  date_col = 'date',
  order = 4,
  K_int = 12,
  J = 500,
  pve_threshold = 0.90,
  max_k = 10,
  K_est = NULL,
  parse_id = TRUE,
  score_group_vars = c('Nitrogen_Level', 'Fungal_Strain'),
  score_plot_style = c('box', 'jitter_mean_ci')
  #lm_formula = NULL,
  #emmeans_specs = NULL
) {
  score_plot_style <- match.arg(score_plot_style)

  stopifnot(trait %in% names(df))
  stopifnot(id_col %in% names(df))

  df <- .ensure_day_id(df, date_col = date_col, time_col = time_col)
  df <- df %>% arrange(.data[[id_col]], .data[[time_col]])
  na_rows <- is.na(df[[trait]])
  if (any(na_rows)) {
    warning(sprintf(
      "Removing %d rows with NA in trait '%s' (IDs: %s)",
      sum(na_rows),
      trait,
      paste(unique(df[[id_col]][na_rows]), collapse = ", ")
    ))
    df <- df[!na_rows, ]
  }

  sample_ID <- factor(df[[id_col]], levels = unique(df[[id_col]]))
  Tlist <- split(df[[time_col]], sample_ID)
  Ylist <- split(df[[trait]], sample_ID)

  T.vec <- Reduce(c, Tlist)
  tlimit <- range(T.vec)
  tmin <- tlimit[1]; tmax <- tlimit[2]

  # ---- basis and penalty ----
  knots <- tmin + (tmax - tmin) * (1:K_int) / (1 + K_int)
  K <- K_int + order
  basis <- create.bspline.basis(tlimit, K, norder = order)
  Omega <- inprod(basis, basis, 2, 2)
  inte1 <- kronecker(inprod(basis, basis, 2, 2), inprod(basis, basis))
  inte2 <- kronecker(inprod(basis, basis, 1, 1), inprod(basis, basis, 1, 1))
  inte3 <- kronecker(inprod(basis, basis), inprod(basis, basis, 2, 2))
  Omega2 <- inte1 + 2 * inte2 + inte3

  # ---- design matrix ----
  N <- length(T.vec)
  Xmat <- matrix(0, N, K)
  start.temp <- 1
  for (i in seq_along(Tlist)) {
    n.i <- length(Tlist[[i]])
    Xmat[start.temp:(start.temp + n.i - 1), ] <- eval.basis(Tlist[[i]], basis, 0)
    start.temp <- start.temp + n.i
  }

  # ---- grid for plotting ----
  tt <- seq(tmin, tmax, length.out = J)
  BS <- eval.basis(tt, basis, 0)

  # ---- empirical mean ----
  Y.vec <- Reduce(c, Ylist)
  ylimit <- range(Y.vec, na.rm = TRUE)
  t.all <- unique(sort(T.vec))
  mu.emp <- sapply(t.all, function(ti) mean(Y.vec[T.vec == ti], na.rm = TRUE))
  df_emp <- data.frame(t = t.all, mu_emp = mu.emp)

  # ---- mean function (penalized LS) ----
  lam_mu <- tuning_nointer(-10, 15, Omega, Xmat, Y.vec)
  bhat <- solve(t(Xmat) %*% Xmat + lam_mu * Omega) %*% (t(Xmat) %*% Y.vec)
  mu_hat <- as.numeric(BS %*% bhat)
  df_mu <- data.frame(t = tt, mu_hat = mu_hat)
  
  # ---- FPCA (covariance + eigens) ----
  fpca <- pca_fun(
    Y_list = Ylist, T_list = Tlist, Xmat = Xmat, bhat = bhat,
    K = K, J = J, BS = BS, basis = basis, Omega = Omega, Omega2 = Omega2
  )
  v1 <- fpca$v1
  V1 <- fpca$V1
  sigma_e2 <- fpca$sigma_e2

  vpos <- v1[v1 > 0]
  if (length(vpos) == 0) stop('No positive eigenvalues found. Check inputs / smoothing.')
  pve <- vpos / sum(vpos)
  cum_pve <- cumsum(pve)
  
  # if (is.null(K_est)) {
  #   K_est <- which(cum_pve >= pve_threshold)[1]
  #   if (is.na(K_est)) K_est <- min(length(vpos), max_k)
  # }
  # K_est <- min(K_est, length(vpos), max_k)
  
  K_est = K_est
  
  df_scree <- data.frame(
    PC = seq_along(vpos),
    eigenvalue = vpos,
    PVE = pve,
    cumPVE = cum_pve
  )

  # p_scree <- ggplot(df_scree, aes(x = PC, y = eigenvalue)) +
  #   geom_point() + geom_line() +
  #   labs(title = paste0('Scree plot: ', trait), x = 'PC', y = 'Eigenvalue') +
  #   theme_bw()
  # 
  # p_pve <- ggplot(df_scree, aes(x = PC, y = cumPVE)) +
  #   geom_point() + geom_line() +
  #   geom_hline(yintercept = pve_threshold, linetype = 2) +
  #   scale_y_continuous(limits = c(0, 1)) +
  #   labs(title = paste0('Cumulative PVE: ', trait), x = 'PC', y = 'Cumulative PVE',
  #        subtitle = paste0('Selected K=', K_est, ' at threshold=', pve_threshold)) +
  #   theme_bw()

  # eigenfunctions on grid
  phi_grid <- BS %*% V1
  phi_grid <- phi_grid[, seq_len(min(ncol(phi_grid), max_k)), drop = FALSE]
  # enforce sign convention on grid (will flip scores later)
  sign_res <- .enforce_phi_sign(phi_grid)
  phi_grid <- sign_res$phi
  flips <- sign_res$flips

  # plot first 2 eigenfunctions
  df_phi <- do.call(rbind, lapply(seq_len(K_est), function(k) {
    data.frame(t = tt, PC = k, phi = phi_grid[, k])
  }))
  label_map <- setNames(
    sprintf('PC%d (%.1f%%)', df_scree$PC[seq_len(K_est)], 100 * df_scree$PVE[seq_len(K_est)]),
    seq_len(K_est)
  )
  df_phi$label <- label_map[as.character(df_phi$PC)]
  
  phi_plots <- lapply(seq_len(K_est), function(k) {
    ggplot(data.frame(t = tt, phi = phi_grid[, k]), aes(x = t, y = phi)) +
      geom_line(linewidth = 1.0) +
      labs(title = sprintf('PC%d (%.1f%%)', k, 100 * pve[k]),
           x = 'Day', y = expression(phi(t))) +
      theme_bw()
  })
  
  # ---- adaptive alpha: scaled to mean curve range ----
  mean_range <- diff(range(mu_hat))
  phi_ranges  <- apply(phi_grid[, seq_len(K_est), drop = FALSE], 2, function(col) diff(range(col)))
  # alpha per PC: make the perturbation ~30% of the mean range
  alphas <- ifelse(phi_ranges > 0, 0.3 * mean_range / phi_ranges, 1)
  mean_pm_plots <- lapply(seq_len(K_est), function(k) {
    a <- alphas[k]
    
    df_mean_pm <- data.frame(
      t     = rep(tt, 3),
      value = c(mu_hat,
                mu_hat + a * phi_grid[, k],
                mu_hat - a * phi_grid[, k]),
      curve = rep(c('mean',
                    paste0('mean + ', round(a, 1), '*phi'),
                    paste0('mean - ', round(a, 1), '*phi')), each = length(tt))
    )
    
    ggplot(df_mean_pm, aes(x = t, y = value, color = curve, linewidth = curve)) +
      geom_line() +
      scale_color_manual(values = setNames(
        c('black', 'red', 'blue'),
        c('mean',
          paste0('mean + ', round(a, 1), '*phi'),
          paste0('mean - ', round(a, 1), '*phi'))
      )) +
      scale_linewidth_manual(values = setNames(
        c(1.2, 0.8, 0.8),
        c('mean',
          paste0('mean + ', round(a, 1), '*phi'),
          paste0('mean - ', round(a, 1), '*phi'))
      )) +
      labs(title = sprintf('Mean \u00b1 %.1f\u00d7\u03c6%d(t)', a, k),
           x = 'Day', y = trait, color = NULL, linewidth = NULL) +
      theme_bw() +
      theme(legend.position = 'bottom',
            legend.text = element_text(size = 7))
  })
  
  
  # combine into 2 x K_est grid
  top_row    <- Reduce(`|`, phi_plots)
  bottom_row <- Reduce(`|`, mean_pm_plots)
  p_phi      <- (top_row / bottom_row) +
    plot_annotation(title = paste0('Eigenfunctions: ', trait))
  
  
  
  # ---- scores (BLUP) ----
  Phi_by_id_orig <- vector('list', length(Tlist))
  for (i in seq_along(Tlist)) {
    Phi_by_id_orig[[i]] <- eval.basis(Tlist[[i]], basis, 0) %*% V1[, seq_len(K_est), drop = FALSE]
  }

  PC <- pca_score(
    Xmat = Xmat, bhat = bhat, Y_vec = Y.vec,
    T_list = Tlist, K.est = K_est,
    v1.est = vpos[seq_len(K_est)],
    Phi1.est = Phi_by_id_orig,
    sigma.e2.hat = sigma_e2
  )
  # Then flip scores post-hoc:
  PC <- sweep(PC, 2, flips[seq_len(K_est)], `*`)

  PC_df <- as.data.frame(PC)
  PC_df[[id_col]] <- rownames(PC_df)
  if (parse_id) PC_df <- .parse_id_underscore(PC_df, id_col = id_col)

  # ---- score distribution by group ----
  score_plots <- list()
  for (k in seq_len(min(2, K_est))) {
    sc <- paste0('fpca', k)
    if (!all(score_group_vars %in% names(PC_df))) next

    if (length(score_group_vars) == 1) {
      g1 <- score_group_vars[1]
      p <- ggplot(PC_df, aes(x = .data[[g1]], y = .data[[sc]]))
      if (score_plot_style == 'box') {
        p <- p + geom_boxplot(outlier.alpha = 0.2) + geom_jitter(width = 0, alpha = 0.7)
      } else {
        p <- p + geom_jitter(width = 0, alpha = 0.7) +
          stat_summary(fun = mean, geom = 'point', size = 3) +
          stat_summary(fun.data = mean_cl_normal, geom = 'errorbar', width = 0.2)
      }
      p <- p + theme_bw() + labs(title = paste0(sc, ' distribution by ', g1), x = g1, y = sc)
      score_plots[[paste0(sc, '_by_', g1)]] <- p
    } else {
      g1 <- score_group_vars[1]
      g2 <- score_group_vars[2]
      p <- ggplot(PC_df, aes(x = .data[[g1]], y = .data[[sc]], color = .data[[g2]]))
      if (score_plot_style == 'box') {
        p <- p + geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.75)) +
          geom_jitter(position = position_jitterdodge(jitter.width = 0, dodge.width = 0.75), alpha = 0.7)
      } else {
        p <- p + geom_jitter(position = position_jitterdodge(jitter.width = 0, dodge.width = 0.75), alpha = 0.7) +
          stat_summary(fun = mean, geom = 'point', position = position_dodge(width = 0.75), size = 3) +
          stat_summary(fun.data = mean_cl_normal, geom = 'errorbar', position = position_dodge(width = 0.75), width = 0.2)
      }
      p <- p + theme_bw() + labs(title = paste0(sc,' of ',trait
                                                #,' distribution by ', g1, ' (colored by ', g2, ')'
                                                ), x = g1, y = sc)
      score_plots[[paste0(sc, '_by_', g1, '_color_', g2)]] <- p
    }
  }

  # # ---- optional linear model + emmeans ----
  # lm_fits <- list()
  # emmeans_out <- list()
  # if (!is.null(lm_formula)) {
  #   for (k in seq_len(K_est)) {
  #     sc <- paste0('fpca', k)
  #     fml <- as.formula(gsub('SCORE', sc, lm_formula, fixed = TRUE))
  #     fit <- lm(fml, data = PC_df)
  #     lm_fits[[sc]] <- fit
  #     if (!is.null(emmeans_specs)) {
  #       # emmeans_specs example: "~ Fungal_Strain | Nitrogen_Level"
  #       emm <- emmeans(fit, as.formula(emmeans_specs))
  #       emmeans_out[[sc]] <- list(emm = emm, pairs = pairs(emm))
  #     }
  #   }
  # }
  
  list(
    inputs = list(trait = trait, id_col = id_col, time_col = time_col, order = order, K_int = K_int, J = J,
                  pve_threshold = pve_threshold, K_est = K_est),
    mean   = list(lambda = lam_mu, bhat = bhat, tt = tt, mu_hat = mu_hat, empirical = df_emp),
    fpca   = list(v1 = v1, vpos = vpos, pve = pve, cum_pve = cum_pve, V1 = V1, phi_grid = phi_grid,
                  sigma_e2 = sigma_e2),
    scores = list(PC = PC, PC_df = PC_df),
    plots  = c(list(phi = p_phi), score_plots)
  )
  
  

  # list(
  #   inputs = list(trait = trait, id_col = id_col, time_col = time_col, order = order, K_int = K_int, J = J,
  #                 pve_threshold = pve_threshold, K_est = K_est),
  #   mean = list(lambda = lam_mu, bhat = bhat, tt = tt, mu_hat = mu_hat, empirical = df_emp),
  #   fpca = list(v1 = v1, vpos = vpos, pve = pve, cum_pve = cum_pve, V1 = V1, phi_grid = phi_grid,
  #               sigma_e2 = sigma_e2),
  #   scores = list(PC = PC, PC_df = PC_df),
  #   plots = c(list(
  #     #raw_mean = p_raw_mean, mean_compare = p_mean_compare, scree = p_scree, pve = p_pve, 
  #                  phi = p_phi), score_plots),
  #   #models = list(lm = lm_fits, emmeans = emmeans_out)
  # )
}

# ------------------------------------------------------------------
# Example usage (keep, but comment out if you only want the function)
# ------------------------------------------------------------------

# df_wide <- read.csv('../data/metrics_wide.csv', stringsAsFactors = FALSE)
# res <- run_fpca_trait(
#   df = df_wide,
#   trait = 'area_pixels_all_raw',
#   lm_formula = 'SCORE ~ Fungal_Strain * Nitrogen_Level',
#   emmeans_specs = '~ Fungal_Strain | Nitrogen_Level'
# )
# res$plots$scree
# res$plots$pve
# res$plots$phi
# res$plots$fpca1_by_Nitrogen_Level_color_Fungal_Strain
