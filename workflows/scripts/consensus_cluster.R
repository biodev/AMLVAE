#!/usr/bin/env Rscript
# consensus_cluster.R
# ------------------------------------------------------------------
# Consensus clustering across cross-validation folds.
# For each fold, reads the latent-space embedding (z matrix), computes
# PAM and hierarchical clustering with euclidean and correlation
# distances for k = k_min..k_max. Then builds a co-occurrence matrix
# across folds and derives final consensus clusters + silhouette scores.
# Outputs heatmap PDFs and a CSV of final cluster assignments.
# ------------------------------------------------------------------

library(argparse)
library(tidyverse)
library(cluster)
library(ComplexHeatmap)


# ---- helper: unique off-diagonal consensus values ---------------------------
# A consensus matrix is symmetric N x N. The CDF is built over the unique
# pairwise values, i.e. the lower triangle excluding the diagonal.
.consensus_values <- function(m) {
  m <- as.matrix(m)
  n <- nrow(m)
  if (n != ncol(m)) stop("Each consensus matrix must be square.")
  # If only one triangle is populated (asymmetric input), fold it down so we
  # still recover the pairwise values. For a proper symmetric matrix this is
  # a no-op (the two triangles are equal).
  if (!isSymmetric(unname(m))) {
    lt <- m[lower.tri(m)]
    ut <- t(m)[lower.tri(m)]
    m[lower.tri(m)] <- pmax(lt, ut, na.rm = TRUE)
  }
  m[lower.tri(m)]
}


# ---- helper: resolve a k value for every matrix -----------------------------
.resolve_kvec <- function(ml, kvec) {
  nK <- length(ml)
  if (is.null(kvec)) {
    nm <- names(ml)
    if (!is.null(nm) && all(grepl("^[0-9]+$", nm))) {
      kvec <- as.integer(nm)
    } else {
      kvec <- seq.int(2L, length.out = nK)  # ml[[1]] == k = 2
    }
  }
  if (length(kvec) != nK)
    stop("`kvec` must have one value per matrix in `ml`.")
  kvec
}


# ---- helper: normalize counts -> [0,1] consensus index ----------------------
.normalize_matrix <- function(m, normalize) {
  m <- as.matrix(m)
  if (is.numeric(normalize)) return(m / normalize)
  if (identical(normalize, "auto") && max(m, na.rm = TRUE) > 1)
    return(m / max(m, na.rm = TRUE))
  m
}


# ---- core: compute CDF curves, AUC and delta-area ---------------------------
# ml        : list of consensus matrices. By convention element i is for the
#             i-th k you supply (see `kvec`).
# kvec      : integer vector of the cluster counts each matrix corresponds to.
#             If NULL, taken from numeric names(ml) when present, otherwise
#             assumed to be 2, 3, 4, ... (i.e. ml[[1]] is k = 2).
# breaks    : number of histogram bins over [0,1] (ConsensusClusterPlus uses 100).
# normalize : "auto"  -> if a matrix has values > 1 it is divided by its max
#                        (turns co-clustering COUNTS into a [0,1] index);
#             "none"  -> use values as supplied;
#              numeric -> divide every matrix by this value (e.g. the number of
#                        resampling iterations).
#
# Returns (invisibly) a list with:
#   $summary : data.frame(k, area, delta)
#   $curves  : data.frame(k, consensus, CDF)   -- long format, handy for ggplot
consensus_cdf_compute <- function(ml, kvec = NULL, breaks = 100,
                                  normalize = "auto") {
  if (!is.list(ml) || length(ml) < 1)
    stop("`ml` must be a non-empty list of consensus matrices.")
  nK   <- length(ml)
  kvec <- .resolve_kvec(ml, kvec)
  ml   <- lapply(ml, .normalize_matrix, normalize = normalize)
  
  # --- empirical CDF + area under curve for each k ---
  brk    <- seq(0, 1, by = 1 / breaks)
  curves <- vector("list", nK)
  areaK  <- numeric(nK)
  
  for (i in seq_len(nK)) {
    v <- .consensus_values(ml[[i]])
    v <- v[is.finite(v)]
    h   <- hist(v, breaks = brk, plot = FALSE)
    cdf <- cumsum(h$counts) / sum(h$counts)
    curves[[i]] <- data.frame(k = kvec[i], consensus = h$mids, CDF = cdf)
    # area under the CDF, rectangle (histogram) integration
    areaK[i] <- sum(cdf * diff(h$breaks))
  }
  
  # --- delta area: relative change in AUC vs. previous k ---
  # (matches ConsensusClusterPlus: first entry is the absolute AUC, then the
  #  proportional increase relative to the preceding k.)
  deltaK <- numeric(nK)
  deltaK[1] <- areaK[1]
  if (nK >= 2)
    for (i in 2:nK) deltaK[i] <- (areaK[i] - areaK[i - 1]) / areaK[i - 1]
  
  invisible(list(
    summary = data.frame(k = kvec, area = areaK, delta = deltaK),
    curves  = do.call(rbind, curves)
  ))
}


# ---- base-R plotting (faithful to ConsensusClusterPlus look) ----------------
# Draws the CDF plot and the delta-area plot. Returns the summary invisibly.
# Tip: call par(mfrow = c(1, 2)) first to see them side by side.
consensus_cdf_plot <- function(ml, kvec = NULL, breaks = 100,
                               normalize = "auto", colors = NULL) {
  res    <- consensus_cdf_compute(ml, kvec, breaks, normalize)
  kvec   <- res$summary$k
  nK     <- length(kvec)
  if (is.null(colors)) colors <- grDevices::rainbow(nK)
  
  # CDF
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), las = 1,
       xlab = "consensus index", ylab = "CDF", main = "Consensus CDF")
  for (i in seq_len(nK)) {
    ci <- res$curves[res$curves$k == kvec[i], ]
    lines(ci$consensus, ci$CDF, col = colors[i], lwd = 2)
  }
  legend("bottomright", legend = paste0("k = ", kvec),
         col = colors, lwd = 2, bty = "n")
  
  # Delta area
  plot(kvec, res$summary$delta, type = "b", pch = 19, las = 1,
       xlab = "k", ylab = "relative change in area under CDF curve",
       main = "Delta area")
  
  invisible(res$summary)
}


# ---- optional ggplot2 versions ----------------------------------------------
# Returns a named list of two ggplot objects: $cdf and $delta.
consensus_cdf_gg <- function(ml, kvec = NULL, breaks = 100, normalize = "auto") {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Package 'ggplot2' is required for consensus_cdf_gg().")
  res <- consensus_cdf_compute(ml, kvec, breaks, normalize)
  res$curves$k  <- factor(res$curves$k)
  
  p_cdf <- ggplot2::ggplot(
    res$curves,
    ggplot2::aes(x = consensus, y = CDF, colour = k)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::labs(x = "consensus index", y = "CDF",
                  title = "Consensus CDF", colour = "k") +
    ggplot2::theme_bw()
  
  p_delta <- ggplot2::ggplot(
    res$summary,
    ggplot2::aes(x = k, y = delta)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(size = 2) +
    ggplot2::scale_x_continuous(breaks = res$summary$k) +
    ggplot2::labs(x = "k", y = "relative change in area under CDF curve",
                  title = "Delta area") +
    ggplot2::theme_bw()
  
  list(cdf = p_cdf, delta = p_delta)
}


# ---- PAC: Proportion of Ambiguous Clustering --------------------------------
# Senbabaoglu et al. (2014). A quantitative, less subjective alternative to the
# delta-area elbow. For each k:
#       PAC_k = CDF_k(u2) - CDF_k(u1)
# i.e. the fraction of pairwise consensus values landing in the ambiguous band
# (u1, u2]. LOWER is better (fewer "on-the-fence" pairs); the recommended k is
# the one that MINIMIZES PAC. Computed directly from the raw consensus values
# via the empirical CDF, so it is independent of histogram binning.
#
# bounds : numeric length-2, the ambiguous band (default c(0.1, 0.9)).
# Returns a data.frame(k, PAC) ordered by k, with attribute "best_k".
consensus_pac <- function(ml, kvec = NULL, bounds = c(0.1, 0.9),
                          normalize = "auto") {
  if (!is.list(ml) || length(ml) < 1)
    stop("`ml` must be a non-empty list of consensus matrices.")
  if (length(bounds) != 2 || bounds[1] >= bounds[2])
    stop("`bounds` must be c(low, high) with low < high.")
  kvec <- .resolve_kvec(ml, kvec)
  u1 <- bounds[1]; u2 <- bounds[2]
  
  pac <- vapply(seq_along(ml), function(i) {
    v <- .consensus_values(.normalize_matrix(ml[[i]], normalize))
    v <- v[is.finite(v)]
    Fn <- stats::ecdf(v)
    as.numeric(Fn(u2) - Fn(u1))      # fraction of values in (u1, u2]
  }, numeric(1))
  
  out <- data.frame(k = kvec, PAC = pac)
  out <- out[order(out$k), ]
  attr(out, "best_k") <- out$k[which.min(out$PAC)]
  rownames(out) <- NULL
  out
}


# ---- convenience: summarise both criteria and suggest k ---------------------
# Combines the delta-area table with PAC and reports the k each criterion
# favours. The two need not agree -- when they do you can be confident; when
# they don't, inspect the heatmaps and use domain knowledge.
#   - PAC  : recommends argmin(PAC)
#   - delta: recommends the largest k whose delta is still "appreciable"
#            (>= delta_tol of the maximum delta); a simple elbow proxy.
choose_k <- function(ml, kvec = NULL, bounds = c(0.1, 0.9),
                     normalize = "auto", delta_tol = 0.1) {
  comp <- consensus_cdf_compute(ml, kvec, normalize = normalize)
  pac  <- consensus_pac(ml, kvec, bounds = bounds, normalize = normalize)
  
  tab <- merge(comp$summary, pac, by = "k")
  tab <- tab[order(tab$k), ]
  rownames(tab) <- NULL
  
  # delta elbow: ignore the first row (absolute area, not a change), then take
  # the largest k whose relative gain clears delta_tol * max(relative gains).
  if (nrow(tab) >= 2) {
    rel <- tab$delta[-1]
    thr <- delta_tol * max(rel, na.rm = TRUE)
    keep <- which(rel >= thr)
    best_delta <- if (length(keep)) tab$k[max(keep) + 1L] else tab$k[2]
  } else {
    best_delta <- tab$k[1]
  }
  
  list(
    table       = tab,
    best_k_pac  = attr(pac, "best_k"),
    best_k_delta = best_delta
  )
}

# --- CLI ---------------------------------------------------------
parser <- ArgumentParser(description = "Consensus clustering over CV folds")
parser$add_argument("--run_dir", required = TRUE,
                    help = "Path to the run directory (contains fold_0/, fold_1/, …)")
parser$add_argument("--out", required = TRUE,
                    help = "Output directory for figures and results")
parser$add_argument("--dataset", required = TRUE,
                    help = "Dataset name prefix used in file names (e.g. 'mds')")
parser$add_argument("--num_folds", type = "integer", required = TRUE,
                    help = "Number of folds")
parser$add_argument("--k_min", type = "integer", default = 2,
                    help = "Minimum number of clusters (default 2)")
parser$add_argument("--k_max", type = "integer", default = 10,
                    help = "Maximum number of clusters (default 10)")
parser$add_argument("--seed", type = "integer", default = 976,
                    help = "Base random seed")

args <- parser$parse_args()

run_dir   <- args$run_dir
out_dir   <- args$out
dataset   <- args$dataset
num_folds <- args$num_folds
k_min     <- args$k_min
k_max     <- args$k_max
base_seed <- args$seed

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- Step 1: Cluster each fold x k ------------------------------
k_range <- seq(k_min, k_max)

k_res <- lapply(setNames(as.character(k_range), as.character(k_range)), function(k_char) {
  k <- as.integer(k_char)

  lapply(0:(num_folds - 1), function(fold) {

    z_path <- file.path(run_dir, paste0("fold_", fold), "distance",
                        paste0(dataset, "_z.csv"))
    z_tbl <- readr::read_csv(z_path, show_col_types = FALSE)
    id_col <- names(z_tbl)[1]
    z_mat <- as.matrix(select(z_tbl, -1))
    rownames(z_mat) <- z_tbl[[1]]

    set.seed(base_seed + fold + k)

    dist.e <- dist(z_mat)
    dist.p <- as.dist(1 - cor(t(z_mat)))

    pam.e <- pam(dist.e, k = k, diss = TRUE)
    pam.p <- pam(dist.p, k = k, diss = TRUE)

    hc.e <- cutree(hclust(dist.e, method = "average"), k = k)
    hc.p <- cutree(hclust(dist.p, method = "average"), k = k)

    stopifnot(all(names(pam.e$clustering) == names(pam.p$clustering)))
    stopifnot(all(names(pam.e$clustering) == names(hc.e)))
    stopifnot(all(names(hc.p) == names(hc.e)))

    clust_res <- cbind(pam_e = pam.e$clustering,
                       pam_p = pam.p$clustering,
                       hc_e  = hc.e,
                       hc_p  = hc.p) %>%
      as_tibble(rownames = id_col)

    list(clust_res = clust_res, k = k, id_col = id_col)
  })
})

# Detect the ID column name from the first result
id_col_name <- k_res[[1]][[1]]$id_col

# --- Step 2: Co-occurrence & consensus --------------------------
method_names <- c("pam_e", "pam_p", "hc_e", "hc_p")

cluster_cooc_list <- lapply(setNames(method_names, method_names), function(samp_name) {

  lapply(k_res, function(k_list) {

    fold_clust_mats <- lapply(k_list, function(fold_list) {

      tmp_res <- select(fold_list$clust_res,
                        all_of(c(id_col_name, "clusters" = samp_name))) %>%
        mutate(value = 1)

      tmp_tmat <- pivot_wider(tmp_res, id_cols = all_of(id_col_name),
                              names_from = clusters, values_from = value,
                              values_fill = 0)
      tmp_mat <- as.matrix(select(tmp_tmat, -1))
      rownames(tmp_mat) <- tmp_tmat[[id_col_name]]

      tmp_mat %*% t(tmp_mat)
    })

    cluster_cooc <- Reduce(function(x, y) {
      x + y[rownames(x), colnames(x)]
    }, fold_clust_mats)

    use_k <- k_list[[1]]$k

    final_clusts <- cutree(hclust(as.dist(1 - (cluster_cooc / num_folds)),
                                  method = "average"), k = use_k)

    final_clust_tbl <- final_clusts %>%
      as_tibble(rownames = id_col_name) %>%
      rename(clusters = value)

    sils <- silhouette(final_clusts, as.dist(1 - (cluster_cooc / num_folds)))

    list(cooc = cluster_cooc, final_clusts = final_clust_tbl, sil = sils)
  })
  
})

# --- Step 3: Save cluster assignments ---------------------------
all_results <- list()
for (method in names(cluster_cooc_list)) {
  for (k_char in names(cluster_cooc_list[[method]])) {
    tbl <- cluster_cooc_list[[method]][[k_char]]$final_clusts %>%
      mutate(method = method, k = as.integer(k_char))
    all_results[[length(all_results) + 1]] <- tbl
  }
}
results_tbl <- bind_rows(all_results)
write_csv(results_tbl, file.path(out_dir, "consensus_clusters.csv"))

# Save silhouette summaries
stat_summaries <- lapply(names(cluster_cooc_list), function(method){
  
  sil_summary <- lapply(names(cluster_cooc_list[[method]]), function(k_char) {
    
    #Summary silhouette scores
    sil_obj <- cluster_cooc_list[[method]][[k_char]]$sil
    
    tibble(
        method = method,
        k = as.integer(k_char),
        avg_silhouette = mean(sil_obj[, "sil_width"])
        )
    
  }) %>% bind_rows()
  
  chosen_k <- choose_k(lapply(cluster_cooc_list[[method]], function(y) y$cooc))
  
  sil_summary <- 
    inner_join(
      sil_summary,
      chosen_k$table,
      by="k"
    ) %>%
    arrange(k)
  
  sil_summary
})

stat_tbl <- bind_rows(stat_summaries)
write_csv(stat_tbl, file.path(out_dir, "statistic_summary.csv"))

# --- Step 4: Heatmap PDFs ---------------------------------------
fig_dir <- file.path(out_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

for (samp_name in names(cluster_cooc_list)) {

  pdf(file = file.path(fig_dir, paste0(samp_name, "_cluster_heatmaps.pdf")),
      width = 7, height = 7)

  for (k_char in names(cluster_cooc_list[[samp_name]])) {

    tmp_cooc   <- cluster_cooc_list[[samp_name]][[k_char]]$cooc
    tmp_clusts <- cluster_cooc_list[[samp_name]][[k_char]]$final_clusts

    tmp_cooc <- tmp_cooc[tmp_clusts[[id_col_name]], tmp_clusts[[id_col_name]]]

    k_int <- as.integer(k_char)
    col_map <- setNames(
      RColorBrewer::brewer.pal(n = max(k_int, 3), "Set3")[1:k_int],
      as.character(1:k_int)
    )

    ha <- HeatmapAnnotation(top_clusts = anno_block(gp = gpar(fill = col_map)))
    hr <- rowAnnotation(right_clusts = anno_block(gp = gpar(fill = col_map)))

    ht1 <- Heatmap(tmp_cooc,
                   col = circlize::colorRamp2(breaks = c(0, num_folds),
                                              colors = c("white", "darkgreen")),
                   column_split = tmp_clusts$clusters,
                   row_split = tmp_clusts$clusters,
                   show_column_names = FALSE, show_row_names = FALSE,
                   border = TRUE, show_heatmap_legend = FALSE,
                   top_annotation = ha, left_annotation = hr,
                   width = unit(2, "null"), height = unit(3, "null"),
                   row_names_side = "left",
                   column_title = paste("k =", k_char))

    draw(ht1)
  }

  dev.off()
}

cat("Consensus clustering complete.\n")
