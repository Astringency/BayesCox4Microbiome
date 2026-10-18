#!/usr/bin/env Rscript
# Common T1D splits, predictor transform, strict concordance, and honest offsets.
# Native historical outputs are retained separately. No outer score selects a model.
suppressPackageStartupMessages({library(FLORAL); library(glmnet); library(survival)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(3L,4L))
repo <- args[1]; baseline <- args[2]; out <- args[3]
zero_tol <- if(length(args)==4L) as.numeric(args[4]) else 1e-8
stopifnot(is.finite(zero_tol),zero_tol>0)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
source(file.path(baseline,'realdata_survival_cv_common.R'))
p <- jsonlite::fromJSON(file.path(repo,'results/t1d_nested_relaxed_20261016/protocol.json'),simplifyVector=FALSE)
d <- read.csv(file.path(repo,'data/T1D.csv'),check.names=FALSE)
stopifnot(digest::digest(file=file.path(repo,'data/T1D.csv'),algo='sha256')==p$data_sha256)
features <- unlist(p$component_cols)
x <- as.matrix(d[,features]); z <- log((x+1e-6)/rowSums(x+1e-6))
clinical <- cbind(Sex_M=as.numeric(d$Sex=='M'),Antibiotics_Tylosin=as.numeric(d$Antibiotics=='Tylosin'))
lr_pairs <- combn(seq_along(features),2)
lr <- z[,lr_pairs[1,]]-z[,lr_pairs[2,]]
colnames(lr) <- paste0('ratio_',seq_len(ncol(lr)))
rows <- list(); predictions <- list()

# Numerical repair is independent of outcomes and applies to *all* path fits,
# including each inner training fit, before validation deviance is calculated.
# Euclidean projection within the active taxon support keeps the lasso zeros.
# A one-taxon support must be the null model under a zero-sum constraint.
project_taxa <- function(b,ncov) {
  taxa <- seq.int(ncov+1L,length(b))
  b[taxa[abs(b[taxa])<zero_tol]] <- 0
  active <- taxa[b[taxa]!=0]
  if(length(active)==1L) b[active] <- 0
  if(length(active)>1L) {
    b[active] <- b[active]-mean(b[active])
    b[tail(active,1)] <- -sum(b[head(active,-1)])
  }
  stopifnot(abs(sum(b[taxa]))<1e-10*max(1,sum(abs(b[taxa]))))
  b
}
original <- getFromNamespace('LogRatioCoxLasso','FLORAL')
changes <- 0L
time_repairs <- 0L
replace_1se <- function(expr) {
  if(missing(expr)) return(quote(expr=))
  if(!is.call(expr)) return(expr)
  if(identical(expr[[1]],as.name('concordance_index'))) {
    # FLORAL jitters training event times, but validation C-index must use the
    # original times so that tied ages are excluded just as in the Bayes metric.
    expr[['surv.time']] <- quote(y[labels == cv, 1])
    time_repairs <<- time_repairs+1L
  }
  if(identical(expr[[1]],as.name('<-')) && identical(expr[[2]],as.name('idx.1se'))) {
    changes <<- changes+1L
    return(quote(idx.1se <- {
      eligible <- which(mean.cvdev<=mean.cvdev[idx.min]+se.min)
      eligible[which.max(lambda[lidx][eligible])]
    }))
  }
  as.call(lapply(as.list(expr),replace_1se))
}
audited <- original; body(audited) <- replace_1se(body(original)); stopifnot(changes==1L,time_repairs==1L)
audit_env <- new.env(parent=asNamespace('FLORAL'))
native_solver <- getFromNamespace('cox_enet_al','FLORAL')
audit_env$cox_enet_al <- function(...) {
  a <- list(...); ans <- do.call(native_solver,a)
  ncov <- if(!is.null(a$ncov)) a$ncov else a[[12]]
  ans$beta <- apply(ans$beta,2,project_taxa,ncov=ncov)
  ans
}
audit_env$concordance_index <- function(x,surv.time,surv.event) {
  list(c.index=bayes_cindex(surv.time,surv.event,x)$cindex)
}
environment(audited) <- audit_env
writeLines(deparse(audited),file.path(out,'audited_LogRatioCoxLasso.R'))
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))

save_model <- function(method,adjustment,k,beta,details,fit=NULL) {
  s <- p$splits[[k]]; tr <- as.integer(unlist(s$outer_train)); te <- as.integer(unlist(s$outer_test))
  design <- if(adjustment=='adjusted') cbind(z,clinical) else z
  risk <- drop(design[te,,drop=FALSE]%*%beta)
  score <- bayes_cindex(d$time[te],d$status[te],risk)
  target <- file.path(out,adjustment,method,sprintf('outer_%02d',k))
  dir.create(target,recursive=TRUE,showWarnings=FALSE)
  pred <- data.frame(method,adjustment,outer_fold=k,row_index=te,time=d$time[te],event=d$status[te],risk=risk)
  baseline_write_csv(pred,file.path(target,'predictions.csv'))
  baseline_write_csv(data.frame(feature=c(features,if(adjustment=='adjusted')colnames(clinical)),coefficient=beta),file.path(target,'coefficients.csv'))
  jsonlite::write_json(details,file.path(target,'selection.json'),auto_unbox=TRUE,pretty=TRUE,digits=NA)
  if(!is.null(fit)) saveRDS(fit,file.path(target,'fit.rds'))
  rows[[length(rows)+1L]] <<- data.frame(family=if(grepl('coda',method))'coda4microbiome' else 'FLORAL',
    method,adjustment,outer_fold=k,n_train=length(tr),n_test=length(te),test_Cindex=score$cindex,
    comparable_pairs=score$n_comparable_pairs,taxa_selected=sum(beta[seq_along(features)]!=0),taxa_sum=sum(beta[seq_along(features)]))
  predictions[[length(predictions)+1L]] <<- pred
  baseline_write_csv(do.call(rbind,rows),file.path(out,'folds.csv'))
  cat(method,adjustment,k,score$cindex,'\n'); flush.console()
}

clinical_fit <- function(ids) {
  dd <- data.frame(time=d$time[ids],status=d$status[ids],clinical[ids,,drop=FALSE])
  coef(coxph(Surv(time,status)~Sex_M+Antibiotics_Tylosin,data=dd,ties='efron'))
}

for(adjustment in c('unadjusted','adjusted')) for(k in seq_len(5L)) {
  s <- p$splits[[k]]; tr <- as.integer(unlist(s$outer_train))
  foldid <- integer(length(tr))
  for(f in 1:5) foldid[match(as.integer(unlist(s$inner_validation[[f]])),tr)] <- f
  stopifnot(all(foldid %in% 1:5),length(unique(unlist(s$inner_validation)))==length(tr))
  nc <- if(adjustment=='adjusted')2L else 0L
  bclin <- if(nc)clinical_fit(tr) else numeric()
  off <- if(nc)drop(clinical[tr,,drop=FALSE]%*%bclin) else rep(0,length(tr))
  full <- glmnet(lr[tr,,drop=FALSE],Surv(d$time[tr],d$status[tr]),family='cox',alpha=.9,
                 offset=off,standardize=TRUE)
  lambda <- full$lambda
  scores <- matrix(NA_real_,length(lambda),5)
  inner_offsets <- list()
  for(f in 1:5) {
    it <- tr[foldid!=f]; iv <- tr[foldid==f]
    bc <- if(nc)clinical_fit(it) else numeric()
    ot <- if(nc)drop(clinical[it,,drop=FALSE]%*%bc) else rep(0,length(it))
    ov <- if(nc)drop(clinical[iv,,drop=FALSE]%*%bc) else rep(0,length(iv))
    inner <- glmnet(lr[it,,drop=FALSE],Surv(d$time[it],d$status[it]),family='cox',alpha=.9,
                    offset=ot,lambda=lambda,standardize=TRUE)
    risks <- predict(inner,newx=lr[iv,,drop=FALSE],s=lambda,newoffset=ov,type='link')
    scores[,f] <- apply(risks,2,function(r)bayes_cindex(d$time[iv],d$status[iv],r)$cindex)
    inner_offsets[[f]] <- list(training_rows=it,validation_rows=iv,clinical_beta=bc)
  }
  means <- rowMeans(scores); se <- apply(scores,1,sd)/sqrt(5)
  best <- which.max(means); eligible <- which(means>=means[best]-se[best])
  selected <- c(min=best,`1se`=eligible[which.max(lambda[eligible])])
  for(rule in names(selected)) {
    j <- selected[[rule]]; b <- as.numeric(coef(full,s=lambda[j])); bt <- numeric(length(features))
    for(v in seq_along(b)) {bt[lr_pairs[1,v]]<-bt[lr_pairs[1,v]]+b[v];bt[lr_pairs[2,v]]<-bt[lr_pairs[2,v]]-b[v]}
    save_model(paste0('coda_matched_',rule),adjustment,k,c(bt,bclin),list(rule=rule,lambda=lambda[j],
      inner_scores=scores,inner_mean=means,inner_se=se,foldid=foldid,inner_offsets=inner_offsets,
      tuning='strict validation C-index; clinical offset refitted on each inner training fold',
      coefficient_scale='unnormalized Cox log-contrast, no unit-sum-absolute rescaling',
      transform='log closed(x+1e-6)',alpha=.9),full)
  }
  zz <- if(nc)cbind(clinical,z) else z
  # Preserve default deviance tuning; also report matched-C-index sensitivity.
  for(criterion in c('Deviance','Cindex')) {
    set.seed(20260602L+sum(utf8ToInt('T1D'))+100000L*k)
    fit <- audited(zz[tr,,drop=FALSE],Surv(d$time[tr],d$status[tr]),ncov=nc,
      wcov=0,a=1,mu=1,maxiter=1000,ncv=5,foldid=foldid,step2=FALSE,progress=FALSE,
      plot=FALSE,mcv=criterion,ncore=1)
    saveRDS(fit,file.path(out,paste0('FLORAL_path_',adjustment,'_',criterion,'_',k,'.rds')))
    for(rule in c('min','1se')) {
      j <- fit$best.idx[[paste0('idx.',rule)]]
      b <- fit$beta[,j]
      bt <- b[seq.int(nc+1L,length(b))]; bc <- if(nc)b[seq_len(nc)] else numeric()
      details <- list(rule=rule,lambda=fit$lambda[j],selected_index=j,inner_criterion=criterion,
        cv_mean=fit$cvdev.mean,cv_se=fit$cvdev.se,foldid=foldid,stage=1,
        repairs='conventional 1se; zero-sum projection within support in full and inner paths; numerical cleanup; original event times for C-index validation',
        numerical_zero_tolerance=zero_tol,
        transform='log closed(x+1e-6)',maxiter=1000)
      save_model(paste0('FLORAL_audited_',tolower(criterion),'_S1_',rule),adjustment,k,c(bt,bc),details)
    }
  }
}
baseline_write_csv(do.call(rbind,predictions),file.path(out,'predictions.csv'))
jsonlite::write_json(list(status='complete',folds=length(rows),data_sha256=p$data_sha256,numerical_zero_tolerance=zero_tol,
  native_FLORAL_source='https://vdblab.github.io/FLORAL/reference/FLORAL.html',
  stage='Stage 1 log-ratio lasso; original package and historical values are retained as sensitivity comparators',
  metric='strict earlier event time, exact risk ties half, same Bayes function'),
  file.path(out,'complete.json'),auto_unbox=TRUE,pretty=TRUE)
