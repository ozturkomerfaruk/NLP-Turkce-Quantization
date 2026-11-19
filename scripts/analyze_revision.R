# Apply the frozen revision protocol using CR2, Satterthwaite and HTZ.
suppressPackageStartupMessages(library(clubSandwich))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript scripts/analyze_revision.R pair-input.csv output-directory")
data <- read.csv(args[1], stringsAsFactors = FALSE)
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
stopifnot(nrow(data) == 3662L, !anyDuplicated(data$pair_id), sum(data$in_baseline) == 1000L)

models <- c("eurollm", "qwen")
precisions <- c("fp16", "q8", "q4")
methods <- c("mean", "total")
categories <- unique(data$category)
margin <- 0.01
family_sizes <- c(RQ1 = 8L, RQ2 = 4L, RQ3 = 4L, RQ4 = 10L, RQ5 = 4L)

valid <- function(se, df) is.finite(se) & se > 0 & is.finite(df) & df > 0
limits <- function(estimate, se, df, level = 0.95) {
  radius <- rep(NA_real_, length(se))
  ok <- valid(se, df)
  radius[ok] <- qt((1 + level) / 2, df[ok]) * se[ok]
  data.frame(lo_pp = 100 * (estimate - radius), hi_pp = 100 * (estimate + radius))
}
scalar <- function(y, cluster) {
  estimate <- mean(y)
  if (all(y == y[1])) {
    return(data.frame(estimate_pp = 100 * estimate, se_pp = NA_real_, df = NA_real_,
                      lo_pp = NA_real_, hi_pp = NA_real_, p_raw = NA_real_))
  }
  fit <- lm(y ~ 1)
  covariance <- vcovCR(fit, cluster = cluster, type = "CR2")
  test <- coef_test(fit, vcov = covariance, test = "Satterthwaite")
  se <- test$SE[1]
  df <- test$df_Satt[1]
  ok <- valid(se, df)
  data.frame(
    estimate_pp = 100 * estimate, se_pp = if (ok) 100 * se else NA_real_,
    df = if (ok) df else NA_real_, limits(estimate, se, df),
    p_raw = if (ok) test$p_Satt[1] else NA_real_
  )
}
tost <- function(result) {
  estimate <- result$estimate_pp / 100
  se <- result$se_pp / 100
  df <- result$df
  ok <- valid(se, df)
  lower <- if (ok) pt((estimate + margin) / se, df, lower.tail = FALSE) else NA_real_
  upper <- if (ok) pt((estimate - margin) / se, df) else NA_real_
  data.frame(
    result[c("estimate_pp", "se_pp", "df")], limits(estimate, se, df, 0.90),
    p_lower = lower, p_upper = upper, p_raw = max(lower, upper)
  )
}
decisions <- function(d, model, precision, method) d[[paste(model, precision, method, sep = "_")]]
correct <- function(d, model, precision, method) as.integer(decisions(d, model, precision, method) == 1)
effect <- function(d, model, precision, method) {
  correct(d, model, precision, method) - correct(d, model, "fp16", method)
}
macro <- function(y, category) mean(tapply(y, category, mean))

cluster_stats <- function(cluster) {
  sizes <- as.integer(table(cluster))
  data.frame(pairs = sum(sizes), clusters = length(sizes),
             singleton_clusters = sum(sizes == 1), maximum_size = max(sizes))
}

full_rank_covariance <- function(covariance) {
  if (any(!is.finite(covariance))) return(FALSE)
  values <- eigen(covariance, symmetric = TRUE, only.values = TRUE)$values
  min(values) > max(values) * .Machine$double.eps * nrow(covariance)
}

buffers <- new.env(parent = emptyenv())
append_row <- function(table, row) {
  buffers[[table]] <- c(buffers[[table]], list(row))
}
append_test <- function(rq, sample, dependence, model, precision, method, comparison, result) {
  fields <- c("estimate_pp", "se_pp", "df", "lo_pp", "hi_pp", "p_raw",
              "p_lower", "p_upper", "statistic", "df_num")
  values <- lapply(fields, function(field) if (is.null(result[[field]])) NA_real_ else result[[field]])
  names(values) <- fields
  append_row("tests", data.frame(rq, sample, dependence, model, precision, method, comparison, as.data.frame(values)))
}

category_inference <- function(y, category, cluster, sample, model, method, dependence) {
  category <- factor(category, levels = categories)
  fit <- lm(y ~ 0 + category)
  covariance <- vcovCR(fit, cluster = cluster, type = "CR2")
  coeffs <- coef_test(fit, vcov = covariance, test = "Satterthwaite")
  estimates <- as.numeric(coef(fit))
  constant <- vapply(split(y, category), function(values) all(values == values[1]), logical(1))
  coeffs$SE[constant] <- NA_real_
  if (dependence == "template") {
    append_row("q4_category_effects", data.frame(
      sample, model, method, category = categories, n = as.integer(table(category)),
      delta_pp = 100 * estimates, limits(estimates, coeffs$SE, coeffs$df_Satt)
    ))
  }

  # Subject Agreement minus the other categories' pair-weighted average.
  if (sample == "new_2662" && dependence == "template") {
    weights <- as.numeric(table(category))
    subject <- match("subject_agreement", categories)
    weights[subject] <- 0
    contrast <- -weights / sum(weights)
    contrast[subject] <- 1
    if (is.finite(drop(contrast %*% covariance %*% contrast)) &&
        drop(contrast %*% covariance %*% contrast) > 0) {
      subject_test <- linear_contrast(fit, vcov = covariance, contrasts = matrix(contrast, nrow = 1))
      interval <- c(subject_test$CI_L, subject_test$CI_U)
    } else {
      interval <- c(NA_real_, NA_real_)
    }
    append_row("subject_agreement", data.frame(
      sample, model, method, delta_vs_other_pp = 100 * sum(contrast * estimates),
      lo_pp = 100 * interval[1], hi_pp = 100 * interval[2]
    ))
  }

  if (sample != "old_1000") {
    constraints <- cbind(-1, diag(length(categories) - 1L))
    contrast_covariance <- constraints %*% covariance %*% t(constraints)
    result <- list()
    if (!any(constant) && full_rank_covariance(covariance) && full_rank_covariance(contrast_covariance)) {
      ht <- Wald_test(fit, constraints, vcov = covariance, test = "HTZ")
      if (is.finite(ht$df_denom) && ht$df_denom > 0 && is.finite(ht$p_val)) {
        result <- list(p_raw = ht$p_val, statistic = ht$Fstat, df = ht$df_denom, df_num = ht$df_num)
      }
    }
    append_test("RQ3", sample, dependence, model, "q4", method, "category_omnibus", result)
  }
}

partitions <- list(full_3662 = data, old_1000 = data[data$in_baseline == 1L, ], new_2662 = data[data$in_baseline == 0L, ])
for (sample in names(partitions)) {
  d <- partitions[[sample]]
  for (category in c("all", categories)) {
    grouping <- if (category == "all") d$cluster else d$cluster[d$category == category]
    append_row("clusters", data.frame(sample, category, cluster_stats(grouping)))
    sizes <- table(as.integer(table(grouping)))
    append_row("cluster_sizes", data.frame(sample, category, size = as.integer(names(sizes)), clusters = as.integer(sizes)))
  }
  for (model in models) for (precision in precisions) {
    for (method in methods) {
      y <- correct(d, model, precision, method)
      estimate <- scalar(y, d$cluster)
      append_row("accuracy", data.frame(
        sample, model, precision, method, n = nrow(d), correct = sum(y),
        ties = sum(decisions(d, model, precision, method) == 0L),
        micro_pct = estimate$estimate_pp, macro_pct = 100 * macro(y, d$category),
        lo_pct = estimate$lo_pp, hi_pct = estimate$hi_pp
      ))
      for (category in categories) {
        within <- d$category == category
        append_row("category_accuracy", data.frame(
          sample, model, precision, method, category, n = sum(within),
          correct = sum(y[within]), accuracy_pct = 100 * mean(y[within])
        ))
      }
      if (precision != "fp16") {
        delta <- effect(d, model, precision, method)
        loss <- sum(delta == -1)
        gain <- sum(delta == 1)
        mcnemar <- if (sample == "old_1000") NA_real_ else if (loss + gain == 0) 1 else binom.test(gain, loss + gain)$p.value
        estimate <- scalar(delta, d$cluster)
        append_row("paired", data.frame(
          sample, model, precision, method, n = nrow(d),
          micro_delta_pp = estimate$estimate_pp, macro_delta_pp = 100 * macro(delta, d$category),
          lo_pp = estimate$lo_pp, hi_pp = estimate$hi_pp, correct_to_incorrect = loss,
          incorrect_to_correct = gain,
          preference_changes = sum(decisions(d, model, precision, method) != decisions(d, model, "fp16", method)),
          mcnemar_exact_p = mcnemar
        ))
      }
    }

    lengths <- sign(d[[paste0(model, "_length_difference")]])
    a <- decisions(d, model, precision, "mean")
    b <- decisions(d, model, precision, "total")
    for (direction in c(-1L, 0L, 1L)) {
      selected <- lengths == direction
      if (!any(selected)) next
      append_row("length", data.frame(
        sample, model, precision, preferred_length = c("shorter", "equal", "longer")[direction + 2L],
        n = sum(selected), mean_pct = 100 * mean(a[selected] == 1L),
        total_pct = 100 * mean(b[selected] == 1L), preference_changes = sum(a[selected] != b[selected])
      ))
    }
  }

  for (dependence in c("template", "independent_pairs")) {
    if (sample == "old_1000" && dependence == "independent_pairs") next
    cluster <- if (dependence == "template") d$cluster else seq_len(nrow(d))
    for (model in models) for (method in methods) {
      category_inference(effect(d, model, "q4", method), d$category, cluster, sample, model, method, dependence)
    }
    if (sample == "old_1000") next
    for (model in models) for (precision in precisions) {
      contrast <- correct(d, model, precision, "total") - correct(d, model, precision, "mean")
      append_test("RQ4", sample, dependence, model, precision, "total_minus_mean", "accuracy",
                  scalar(contrast, cluster))
      if (precision == "fp16") next
      for (method in methods) {
        fitted <- scalar(effect(d, model, precision, method), cluster)
        append_test("RQ1", sample, dependence, model, precision, method, "vs_fp16", fitted)
        if (precision == "q8") {
          append_test("RQ5", sample, dependence, model, precision, method, "equivalence_pm_1pp", tost(fitted))
        }
      }
      contrast <- effect(d, model, precision, "total") - effect(d, model, precision, "mean")
      append_test("RQ4", sample, dependence, model, precision, "total_minus_mean", "effect_vs_fp16",
                  scalar(contrast, cluster))
    }
    for (precision in precisions[-1]) for (method in methods) {
      contrast <- effect(d, "qwen", precision, method) - effect(d, "eurollm", precision, method)
      append_test("RQ2", sample, dependence, "qwen_minus_eurollm", precision, method, "effect_vs_fp16",
                  scalar(contrast, cluster))
    }
  }
  message(sample, ": ", nrow(d), " paired records analyzed")
}

tests <- do.call(rbind, buffers[["tests"]])
tests$p_holm <- NA_real_
for (family in split(seq_len(nrow(tests)), list(tests$sample, tests$dependence, tests$rq), drop = TRUE)) {
  expected <- family_sizes[[tests$rq[family[1]]]]
  stopifnot(length(family) == expected)
  tests$p_holm[family] <- p.adjust(tests$p_raw[family], method = "holm", n = expected)
}
tests$decision <- ifelse(
  is.na(tests$p_holm), "not_estimable",
  ifelse(tests$rq == "RQ5",
         ifelse(tests$p_holm < 0.05, "equivalent", "equivalence_not_demonstrated"),
         ifelse(tests$p_holm < 0.05, "difference_detected", "difference_not_detected"))
)
buffers[["tests"]] <- list(tests)
for (name in ls(buffers)) {
  write.csv(do.call(rbind, buffers[[name]]), file.path(output, paste0(name, ".csv")), row.names = FALSE, na = "")
}
writeLines(capture.output(sessionInfo()), file.path(output, "session-info.txt"))
