#!/usr/bin/env Rscript
# Standalone manuscript figures from audited CSVs. No model fitting occurs here.
# Related sources: scripts/plot_realdata_method_comparison.R and the previous
# scripts/review/render_manuscript_figures.py. PDFs use vector text and marks.
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
a<-commandArgs(trailingOnly=TRUE);stopifnot(length(a)==2L)
root<-a[1];out<-a[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
fold<-read.csv(file.path(root,'comparison_folds.csv'),check.names=FALSE)
summary<-read.csv(file.path(root,'comparison_summary.csv'),check.names=FALSE)
primary_policy<-if(any(summary$policy=='full_grid'))'full_grid' else 'weak_grid'
primary_label<-if(primary_policy=='weak_grid')'Wide grid' else 'Full grid'
palette<-c('Cox-RHS'='#25658A','Prox-SSLASSO'='#B85C38','Baseline'='#555D66')
domain<-range(c(.5,fold$test_Cindex,summary$Cindex_mean-summary$Cindex_sd,
                summary$Cindex_mean+summary$Cindex_sd),na.rm=TRUE)
domain<-c(floor(max(0,domain[1])/.05)*.05,ceiling(min(1,domain[2])/.05)*.05)
theme_set(theme_classic(base_size=13,base_family='Times New Roman')+
  theme(text=element_text(colour='#000000'),
    axis.line=element_line(colour='#666666',linewidth=.3),
    axis.ticks=element_line(colour='#777777',linewidth=.3),
    panel.grid.major.x=element_line(colour='#E6E6E6',linewidth=.25),
    plot.title=element_text(size=13,face='bold',colour='#000000',margin=margin(b=8)),
    axis.text=element_text(colour='#000000'),
    axis.title=element_text(size=12,colour='#000000'),
    strip.text=element_text(colour='#000000'),
    legend.text=element_text(colour='#000000'),
    legend.position='bottom',legend.title=element_blank(),plot.margin=margin(8,10,8,8)))
save_plot<-function(p,name,w,h) {
  ggsave(file.path(out,paste0(name,'.pdf')),p,width=w,height=h,device=cairo_pdf)
  ggsave(file.path(out,paste0(name,'.png')),p,width=w,height=h,dpi=240,bg='white')
}
labels<-c('cox-rhs__prox'='Cox-RHS (Prox)','cox-rhs__soft_center'='Cox-RHS (Soft-center)',
  'coda_matched_min'='coda4microbiome (min)','coda_matched_1se'='coda4microbiome (1se)',
  'FLORAL_native_S2_min'='FLORAL (min)','FLORAL_native_S2_1se'='FLORAL (1se)')
panel<-function(adj,clinical) {
  ids<-c(paste0(clinical,c('_prox','_soft_center'),paste0('__',primary_policy)),names(labels))
  lab<-c('Prox-SSLASSO (Prox)','Prox-SSLASSO (Soft-center)',unname(labels))
  ss<-summary[summary$adjustment==adj & summary$model_id %in% ids,]
  ff<-fold[fold$adjustment==adj & fold$model_id %in% ids,]
  stopifnot(nrow(ss)==length(ids),nrow(ff)==5*length(ids))
  ss$label<-factor(lab[match(ss$model_id,ids)],levels=rev(lab))
  ff$label<-factor(lab[match(ff$model_id,ids)],levels=rev(lab))
  ss$group<-ifelse(ss$method=='prox-sslasso','Prox-SSLASSO',ifelse(ss$method=='cox-rhs','Cox-RHS','Baseline'))
  ff$y<-as.numeric(ff$label)+(ff$outer_fold-3)*.055
  ff$failed<-!is.na(ff$strict_pass)&tolower(as.character(ff$strict_pass))=='false'
  # A separate, aligned value column shows means without covering fold points.
  label_x<-domain[2]+.025
  ss$mean_label<-sprintf('%.4f',ss$Cindex_mean)
  write.csv(ss[,c('model_id','Cindex_mean','Cindex_sd','mean_label')],
    file.path(out,paste0('T1D_nested_cv_',adj,'_labels.csv')),row.names=FALSE)
  ggplot(ss,aes(x=Cindex_mean,y=label))+
    geom_vline(xintercept=.5,linetype='dashed',colour='#9B9B9B',linewidth=.35)+
    geom_point(data=ff,aes(x=test_Cindex,y=y),inherit.aes=FALSE,shape=1,size=1.35,stroke=.35,colour='#969BA1')+
    geom_segment(aes(x=Cindex_mean-Cindex_sd,xend=Cindex_mean+Cindex_sd,yend=label,colour=group),linewidth=.8)+
    geom_point(aes(colour=group),size=2.1)+
    geom_point(data=ff[ff$failed,],aes(x=test_Cindex,y=y),inherit.aes=FALSE,
      shape=4,size=1.9,stroke=.6,colour='#A12C35')+
    geom_text(aes(x=label_x,label=mean_label),hjust=0,family='Times New Roman',
      size=4.1,colour='#000000')+
    scale_colour_manual(values=palette)+scale_x_continuous(breaks=seq(domain[1],domain[2],.05))+
    scale_y_discrete(limits=rev(lab))+
    coord_cartesian(xlim=domain,clip='off')+labs(x='Outer-fold C-index',y=NULL)+
    theme(plot.margin=margin(8,62,8,8))
}
save_plot(panel('unadjusted','none'),'T1D_nested_cv_unadjusted',7.6,4.3)
save_plot(panel('adjusted','student_t'),'T1D_nested_cv_adjusted',7.6,4.3)

ss<-summary[grepl('^scale_',summary$policy),];ff<-fold[grepl('^scale_',fold$policy),]
pairs<-c('scale_2_10','scale_2_20','scale_5_20','scale_5_50','scale_10_50')
for(n in c('ss','ff')) {
  q<-get(n);q$scale<-factor(q$policy,levels=pairs,labels=c('2/10','2/20','5/20','5/50','10/50'))
  q$clinical<-factor(q$clinical_prior,levels=c('none','student_t'),
    labels=c('Unadjusted','Adjusted: Student-t(3,0,2.5)'))
  q$constraint<-factor(q$constraint,levels=c('prox','soft_center'),labels=c('Prox','Soft-center'))
  assign(n,q)
}
ff$x<-as.numeric(ff$scale)+(ff$outer_fold-3)*.055
ff$failed<-!is.na(ff$strict_pass)&tolower(as.character(ff$strict_pass))=='false'
ref<-data.frame(clinical=factor(c('Unadjusted','Adjusted: Student-t(3,0,2.5)'),levels=levels(ss$clinical)),
  reference=c(summary$Cindex_mean[summary$model_id=='coda_matched_1se'&summary$adjustment=='unadjusted'],
              rep(summary$Cindex_mean[summary$model_id=='coda_matched_1se'&summary$adjustment=='adjusted'],1)))
weak<-ggplot(ss,aes(x=scale,y=Cindex_mean))+
  geom_hline(yintercept=.5,colour='#B0B0B0',linetype='dotted',linewidth=.3)+
  geom_hline(data=ref,aes(yintercept=reference),colour='#5B6770',linetype='dashed',linewidth=.4)+
  geom_point(data=ff,aes(x=x,y=test_Cindex),inherit.aes=FALSE,shape=1,size=1.3,colour='#A2A6AA')+
  geom_linerange(aes(ymin=Cindex_mean-Cindex_sd,ymax=Cindex_mean+Cindex_sd),colour=palette['Prox-SSLASSO'],linewidth=.7)+
  geom_point(colour=palette['Prox-SSLASSO'],size=1.8)+
  geom_point(data=ff[ff$failed,],aes(x=x,y=test_Cindex),inherit.aes=FALSE,
    shape=4,size=1.9,stroke=.6,colour='#A12C35')+
  facet_grid(constraint~clinical,labeller=labeller(clinical=as_labeller(
    c('Unadjusted'="'Unadjusted'",
      'Adjusted: Student-t(3,0,2.5)'="'Adjusted: Student-'*italic(t)[3]*'(0,2.5)'"),
    default=label_parsed)))+scale_x_discrete(limits=c('2/10','2/20','5/20','5/50','10/50'))+
  coord_cartesian(ylim=domain)+
  labs(x='Spike/slab scales',y='Outer-fold C-index')+
  theme(strip.background=element_blank(),strip.text=element_text(size=12,colour='#000000'),panel.spacing=unit(1,'lines'))
save_plot(weak,'T1D_relaxed_scales',8.2,5.7)

selection<-read.csv(file.path(root,'full_data_selection_counts.csv'))
selection<-selection[selection$policy %in% c(primary_policy,pairs),]
selection$scale<-factor(selection$policy,levels=c(primary_policy,pairs),labels=c(primary_label,'2/10','2/20','5/20','5/50','10/50'))
selection$arm_label<-paste(ifelse(selection$clinical_prior=='none','Unadjusted',ifelse(selection$clinical_prior=='student_t','Adjusted: Student-t','Adjusted: Cauchy')),
                           ifelse(selection$constraint=='prox','prox','soft-center'),sep=' / ')
arm_order<-unlist(lapply(c('Unadjusted','Adjusted: Student-t'),
  function(x)paste(x,c('prox','soft-center'),sep=' / ')))
selection$arm_label<-factor(selection$arm_label,levels=rev(arm_order))
heat<-ggplot(selection,aes(x=scale,y=arm_label,fill=taxa_selected))+
  geom_tile(colour='white',linewidth=1)+geom_text(aes(label=taxa_selected),size=3.3,colour='#000000')+
  scale_fill_gradient(low='#F1F2F2',high='#D68D68',limits=c(0,max(5,selection$taxa_selected)),breaks=0:max(5,selection$taxa_selected))+
  labs(x='Full-data tuning policy',y=NULL,fill='Selected features')+
  theme(axis.line=element_blank(),axis.ticks=element_blank(),panel.grid=element_blank(),legend.position='none')
save_plot(heat,'T1D_selection_sensitivity',7.5,3)
writeLines(capture.output(sessionInfo()),file.path(out,'T1D_plot_sessionInfo.txt'))
