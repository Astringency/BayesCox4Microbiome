#!/usr/bin/env Rscript
# Clinical-only PH test and exploratory score-calibration PH diagnostics.
suppressPackageStartupMessages(library(survival))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2L) stop("usage: check_t1d_ph.R T1D.csv output_dir [joint_ph_input.csv]")
dat <- read.csv(args[1], check.names=FALSE)
dir.create(args[2], recursive=TRUE, showWarnings=FALSE)
dat$Sex <- factor(dat$Sex, levels=c("F", "M"))
dat$Antibiotics <- factor(dat$Antibiotics, levels=c("Control", "Tylosin"))
save_test <- function(fit, label, note) {
  z <- cox.zph(fit, transform="km", terms=TRUE, global=TRUE)
  tab <- data.frame(term=rownames(z$table), z$table, row.names=NULL, check.names=FALSE)
  tab$analysis <- label
  tab$interpretation <- note
  write.csv(tab, file.path(args[2], paste0(label, "_ph.csv")), row.names=FALSE)
  pdf(file.path(args[2], paste0(label, "_schoenfeld.pdf")), width=8, height=6)
  plot(z)
  dev.off()
  saveRDS(list(fit=fit, zph=z), file.path(args[2], paste0(label, ".rds")))
}
fit <- coxph(Surv(time, status) ~ Sex + Antibiotics, data=dat, x=TRUE, y=TRUE, ties="efron")
save_test(fit, "clinical_only", "PH score test for a frequentist Cox fit with Sex and Antibiotics only")
if (length(args) >= 3L) {
  joint <- read.csv(args[3])
  stopifnot(nrow(joint)==nrow(dat), all(joint$time==dat$time), all(joint$status==dat$status))
  fit <- coxph(Surv(time, status) ~ microbiome_score + Sex_M + Antibiotics_Tylosin,
               data=joint, x=TRUE, y=TRUE, ties="efron")
  save_test(fit, "joint_score_calibration",
    "Exploratory PH diagnostic after refitting the posterior microbiome score with clinical variables; score estimated on these subjects; not a taxon-wise Bayesian PH test")
}
writeLines(c("Time transform: Kaplan-Meier (km); Efron ties.",
  "Term-wise and global Schoenfeld residual score tests from survival::cox.zph.",
  "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/cox.zph.html",
  capture.output(sessionInfo())), file.path(args[2], "method_and_session.txt"))
