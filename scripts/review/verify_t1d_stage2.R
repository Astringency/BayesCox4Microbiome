#!/usr/bin/env Rscript
# Recompute outer scores from saved native second-stage Cox fits. No refitting.
suppressPackageStartupMessages(library(survival))
a<-commandArgs(trailingOnly=TRUE);stopifnot(length(a)==1L)
repo<-a[1];root<-file.path(repo,'results/t1d_nested_relaxed_20261016')
p<-jsonlite::fromJSON(file.path(root,'protocol.json'),simplifyVector=FALSE)
d<-read.csv(file.path(repo,'data/T1D.csv'),check.names=FALSE)
taxa<-unlist(p$component_cols);x<-as.matrix(d[,taxa]);z<-log((x+1e-6)/rowSums(x+1e-6))
clinical<-cbind(Sex_M=as.numeric(d$Sex=='M'),Antibiotics_Tylosin=as.numeric(d$Antibiotics=='Tylosin'))
cindex<-function(t,e,r) {
  pairs<-outer(t,t,'<') & (e==1)
  sum((outer(r,r,'>')+.5*outer(r,r,'=='))[pairs])/sum(pairs)
}
rows<-list();equal<-list()
for(adj in c('unadjusted','adjusted')) for(k in 1:5) {
  fit<-readRDS(file.path(root,'floral_stage2',paste0(adj,'_',k,'.rds')))
  zz<-if(adj=='adjusted') cbind(clinical,z) else z
  tr<-unlist(p$splits[[k]]$outer_train);te<-unlist(p$splits[[k]]$outer_test)
  for(rule in c('min','1se')) {
    m<-fit[[paste0('step2fit.',rule)]]
    beta<-setNames(numeric(ncol(zz)),colnames(zz));training_error<-0
    if(!is.null(m)) {
      b<-coef(m);stopifnot(all(is.finite(b)))
      clin<-intersect(names(b),colnames(clinical));beta[clin]<-b[clin]
      ratios<-setdiff(names(b),clin);idx<-fit$step2.ratios[[paste0(rule,'.idx')]]
      if(length(ratios)) {
        idx<-matrix(idx,nrow=2);stopifnot(ncol(idx)==length(ratios))
        for(j in seq_along(ratios)) {
          beta[idx[1,j]]<-beta[idx[1,j]]+b[ratios[j]]
          beta[idx[2,j]]<-beta[idx[2,j]]-b[ratios[j]]
        }
      }
      delta<-drop(zz[tr,,drop=FALSE]%*%beta)-m$linear.predictors
      training_error<-max(abs(delta-mean(delta)));stopifnot(training_error<1e-10)
    } else stopifnot(adj=='unadjusted')
    risk<-drop(zz[te,,drop=FALSE]%*%beta)
    dest<-file.path(root,'floral_stage2',adj,paste0('FLORAL_native_S2_',rule),sprintf('outer_%02d',k))
    saved<-read.csv(file.path(dest,'predictions.csv'))
    stopifnot(identical(as.integer(saved$row_index),as.integer(te)),max(abs(risk-saved$risk))<1e-10)
    rows[[length(rows)+1L]]<-data.frame(adjustment=adj,outer_fold=k,rule,
      coefficient_source=paste0('step2fit.',rule),test_Cindex=cindex(d$time[te],d$status[te],risk),
      taxa_selected=sum(beta[taxa]!=0),training_reconstruction_error=training_error,
      prediction_reconstruction_error=max(abs(risk-saved$risk)))
    if(adj=='adjusted' && rule=='1se') {
      cp<-read.csv(file.path(root,'baselines_matched',adj,'coda_matched_1se',sprintf('outer_%02d',k),'predictions.csv'))
      stopifnot(identical(cp$row_index,saved$row_index))
      same<-identical(sign(outer(cp$risk,cp$risk,'-')),sign(outer(risk,risk,'-')))
      stopifnot(same)
      equal[[k]]<-data.frame(outer_fold=k,coda_Cindex=cindex(cp$time,cp$event,cp$risk),
        floral_stage2_Cindex=cindex(d$time[te],d$status[te],risk),
        risk_max_absolute_difference=max(abs(cp$risk-risk)),
        coda_risk_levels=length(unique(cp$risk)),floral_risk_levels=length(unique(risk)),
        all_pair_orderings_identical=same)
    }
  }
}
write.csv(do.call(rbind,rows),file.path(root,'stage2_direct_recalculation.csv'),row.names=FALSE)
write.csv(do.call(rbind,equal),file.path(root,'baseline_equal_cindex_explanation.csv'),row.names=FALSE)
cat('Verified 20 outer scores directly from native step2fit objects. All five adjusted 1se risk rankings agree.\n')
print(do.call(rbind,equal))
