#!/usr/bin/env Rscript
# Posterior-mean Schoenfeld diagnostics. No likelihood optimization is performed.
suppressPackageStartupMessages(library(survival))

posterior_schoenfeld <- function(time, status, X, beta) {
  stopifnot(nrow(X)==length(time), ncol(X)==length(beta), all(is.finite(X)),
            all(is.finite(beta)), all(status %in% 0:1))
  eta <- drop(X %*% beta)
  events <- which(status==1L)
  events <- events[order(time[events],events)]
  residuals <- matrix(NA_real_,length(events),ncol(X))
  for (tt in unique(time[events])) {
    deaths <- which(time==tt & status==1L)
    risk <- which(time>=tt)
    shift <- max(eta[risk])
    w <- exp(eta[risk]-shift); wd <- exp(eta[deaths]-shift)
    s0 <- sum(w); e0 <- sum(wd)
    s1 <- colSums(X[risk,,drop=FALSE]*w)
    e1 <- colSums(X[deaths,,drop=FALSE]*wd)
    expected <- numeric(ncol(X))
    d <- length(deaths)
    for (l in 0:(d-1L)) expected <- expected+(s1-l/d*e1)/(s0-l/d*e0)/d
    idx <- which(time[events]==tt)
    residuals[idx,] <- sweep(X[events[idx],,drop=FALSE],2,expected,'-')
  }
  list(residuals=residuals,subject_index=events,time=time[events])
}

run_posterior_ph <- function(design_file, coefficient_file, output_dir) {
  dat <- read.csv(design_file,check.names=FALSE)
  coef <- read.csv(coefficient_file,check.names=FALSE)
  labels <- as.character(coef$feature_id)
  stopifnot(identical(names(dat)[-(1:3)],labels),identical(dat$subject_index,seq_len(nrow(dat))))
  X <- as.matrix(dat[,labels,drop=FALSE]); beta <- coef$posterior_mean
  r <- posterior_schoenfeld(dat$time,dat$status,X,beta)
  dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
  # Construct a survival object ONLY at beta=posterior mean. iter.max=0 forbids refitting.
  # Separate syntactic variable names preserve a test row for every feature.
  d <- data.frame(time=dat$time,status=dat$status,X,check.names=TRUE)
  names(d)[-(1:2)] <- paste0('v',seq_along(labels))
  formula <- reformulate(names(d)[-(1:2)],response='Surv(time,status)')
  fixed <- coxph(formula,data=d,init=beta,x=TRUE,y=TRUE,
                 ties='efron',control=coxph.control(iter.max=0,timefix=FALSE))
  stopifnot(isTRUE(all.equal(unname(coef(fixed)),beta,tolerance=1e-12)),fixed$iter==0)
  ref <- as.matrix(residuals(fixed,type='schoenfeld'))
  # In ties, compare residual sums at each event time, independent of library tie order.
  ours <- rowsum(r$residuals,r$time,reorder=TRUE)
  theirs <- rowsum(ref,as.numeric(rownames(ref)),reorder=TRUE)
  error <- max(abs(ours-theirs))
  stopifnot(is.finite(error),error<1e-8)
  raw <- data.frame(subject_index=dat$subject_index[r$subject_index],time=r$time,r$residuals,check.names=FALSE)
  names(raw)[-(1:2)] <- labels
  write.csv(raw,file.path(output_dir,'schoenfeld_residuals.csv'),row.names=FALSE)
  # The plots are the primary Watson et al. diagnostic. Off-MLE chi-square tests
  # below are exploratory plug-in diagnostics, not calibrated Bayesian p-values.
  z <- cox.zph(fixed,transform='km',terms=FALSE,global=TRUE)
  tab <- data.frame(feature_id=c(labels,'GLOBAL'),z$table,row.names=NULL,check.names=FALSE)
  names(tab)[names(tab)=='p'] <- 'p_plugin_exploratory'
  tab$estimator <- 'joint Bayesian posterior mean; zero optimization iterations'
  tab$interpretation <- 'Exploratory plug-in score diagnostic: shrinkage estimate is not MLE; chi-square calibration is not guaranteed. Primary assessment uses residual plots.'
  write.csv(tab,file.path(output_dir,'posterior_mean_ph.csv'),row.names=FALSE)
  feature_table <- file.path(dirname(coefficient_file),'posterior_features.csv')
  titles <- labels
  if (file.exists(feature_table)) {
    ft <- read.csv(feature_table,check.names=FALSE)
    genus <- ft$genus[match(labels,ft$feature_id)]
    titles <- ifelse(is.na(genus)|genus=='',labels,paste0(labels,' / ',genus))
  }
  pdf(file.path(output_dir,'posterior_mean_schoenfeld.pdf'),width=12,height=9,onefile=TRUE)
  par(mfrow=c(3,3),mar=c(4,4,3,1),oma=c(1,1,2,1))
  trend <- list()
  for (j in seq_along(labels)) {
    plot(r$time,r$residuals[,j],pch=16,cex=.45,col='#66666688',
         xlab='Observed event time',ylab='Schoenfeld residual',main=titles[j],cex.main=.8)
    abline(h=0,lty=2,col='gray50')
    smooth <- smooth.spline(r$time,r$residuals[,j],df=4)
    lines(smooth,col='#C04B24',lwd=2)
    trend[[j]] <- data.frame(feature_id=labels[j],event_time=smooth$x,smooth_residual=smooth$y)
  }
  dev.off()
  write.csv(do.call(rbind,trend),file.path(output_dir,'schoenfeld_smooth.csv'),row.names=FALSE)
  writeLines(c('Joint model parameters: Bayesian posterior means, all included taxa and clinical covariates.',
    'Raw Schoenfeld residuals use full observed risk sets and Efron tied-event means.',
    'Primary diagnostic: residual-versus-time plots with a four-degree-of-freedom smoothing spline.',
    'No auxiliary model or coefficient re-estimation; coxph iter.max=0 is a numerical residual cross-check.',
    'cox.zph p-values are exploratory plug-in values, not posterior probabilities or calibrated Bayesian tests.',
    paste('Independent residual check maximum absolute difference:',format(error,digits=16)),
    'https://doi.org/10.1111/rssc.12501',capture.output(sessionInfo())),file.path(output_dir,'method_and_session.txt'))
  invisible(list(max_residual_error=error,coefficients=coef(fixed),iterations=fixed$iter,ph=tab))
}

if (sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if (length(args)!=3L) stop('usage: check_t1d_ph.R ph_design.csv ph_coefficients.csv OUTPUT_DIR')
  run_posterior_ph(args[1],args[2],args[3])
}
