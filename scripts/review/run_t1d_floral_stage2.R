#!/usr/bin/env Rscript
# Native two-stage FLORAL sensitivity; retain package selection and tie handling.
suppressPackageStartupMessages({library(FLORAL);library(survival)})
a <- commandArgs(trailingOnly=TRUE);stopifnot(length(a)==3L)
repo<-a[1];base<-a[2];out<-a[3];dir.create(out,recursive=TRUE,showWarnings=FALSE)
source(file.path(base,'realdata_survival_cv_common.R'))
p<-jsonlite::fromJSON(file.path(repo,'results/t1d_nested_relaxed_20261016/protocol.json'),simplifyVector=FALSE)
d<-read.csv(file.path(repo,'data/T1D.csv'),check.names=FALSE);taxa<-unlist(p$component_cols)
x<-as.matrix(d[,taxa]);z<-log((x+1e-6)/rowSums(x+1e-6))
c<-cbind(Sex_M=as.numeric(d$Sex=='M'),Antibiotics_Tylosin=as.numeric(d$Antibiotics=='Tylosin'))
rows<-list();failures<-list()
for(adj in c('unadjusted','adjusted')) for(k in 1:5) {
  s<-p$splits[[k]];tr<-as.integer(unlist(s$outer_train));te<-as.integer(unlist(s$outer_test))
  nc<-if(adj=='adjusted')2L else 0L;zz<-if(nc)cbind(c,z) else z
  folds<-integer(length(tr));for(f in 1:5)folds[match(unlist(s$inner_validation[[f]]),tr)]<-f
  set.seed(20260602L+sum(utf8ToInt('T1D'))+100000L*k)
  fit<-tryCatch(FLORAL(zz[tr,,drop=FALSE],Surv(d$time[tr],d$status[tr]),ncov=nc,
    family='cox',pseudo=NULL,ncv=5,ncore=1,foldid=folds,step2=TRUE,progress=FALSE,plot=FALSE),
    error=function(e)e)
  if(inherits(fit,'error')) {
    failures[[paste(adj,k)]]<-conditionMessage(fit);next
  }
  saveRDS(fit,file.path(out,paste0(adj,'_',k,'.rds')))
  for(rule in c('min','1se')) {
    model<-fit[[paste0('step2fit.',rule)]]
    beta<-setNames(numeric(ncol(zz)),colnames(zz))
    reconstruction_error<-0
    if(!is.null(model)) {
      b<-coef(model);stopifnot(all(is.finite(b)))
      clin<-intersect(names(b),colnames(c));beta[clin]<-b[clin]
      rb<-b[setdiff(names(b),clin)]
      idx<-fit$step2.ratios[[paste0(rule,'.idx')]]
      if(length(rb)) {
        idx<-matrix(idx,nrow=2);stopifnot(ncol(idx)==length(rb))
        for(j in seq_along(rb)) {
          beta[idx[1,j]]<-beta[idx[1,j]]+rb[j];beta[idx[2,j]]<-beta[idx[2,j]]-rb[j]
        }
      }
      # Verify ratio-to-taxon mapping against the native fitted risk on training
      # subjects. Cox centering contributes only one constant. Native objects
      # reuse a mutable call environment, so avoid re-evaluating that call.
      delta<-drop(zz[tr,,drop=FALSE]%*%beta)-model$linear.predictors
      reconstruction_error<-max(abs(delta-mean(delta)))
      stopifnot(is.finite(reconstruction_error),reconstruction_error<1e-10)
    }
    risk<-drop(zz[te,,drop=FALSE]%*%beta)
    score<-bayes_cindex(d$time[te],d$status[te],risk)
    method<-paste0('FLORAL_native_S2_',rule)
    dest<-file.path(out,adj,method,sprintf('outer_%02d',k));dir.create(dest,recursive=TRUE,showWarnings=FALSE)
    baseline_write_csv(data.frame(row_index=te,time=d$time[te],event=d$status[te],risk=risk),file.path(dest,'predictions.csv'))
    baseline_write_csv(data.frame(feature=names(beta),coefficient=beta),file.path(dest,'coefficients.csv'))
    rows[[length(rows)+1]]<-data.frame(family='FLORAL',method,adjustment=adj,outer_fold=k,
      n_train=length(tr),n_test=length(te),test_Cindex=score$cindex,comparable_pairs=score$n_comparable_pairs,
      taxa_selected=sum(beta[taxa]!=0),taxa_sum=sum(beta[taxa]),native_training_reconstruction_error=reconstruction_error)
    cat(adj,k,rule,score$cindex,'\n');flush.console()
  }
}
baseline_write_csv(do.call(rbind,rows),file.path(out,'folds.csv'))
jsonlite::write_json(list(status=if(length(failures))'incomplete' else 'complete',failures=failures,
  source='Unmodified FLORAL 0.7.0 two-stage pipeline with common predictor transform and outer/first-stage inner folds; native second-stage CV and stepwise refit',
  caveat='Stage-1 1se defect is retained as package-native sensitivity, not silently relabelled as repaired'),
  file.path(out,'complete.json'),auto_unbox=TRUE,pretty=TRUE)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
