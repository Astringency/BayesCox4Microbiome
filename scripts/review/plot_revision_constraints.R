#!/usr/bin/env Rscript
# Usage: Rscript plot_revision_constraints.R INPUT_DATA_DIR OUTPUT_DIR
# Each box represents posterior draws in repetition 1, not repeated-fit means.
# Write two separate plots and a paired, ten-box comparison; all have no title.
.libPaths(c('~/R/library',.libPaths()))
suppressPackageStartupMessages({library(ggplot2);library(patchwork)})
a<-commandArgs(trailingOnly=TRUE);stopifnot(length(a)==2L)
root<-a[1];out<-a[2];dir.create(out,recursive=TRUE,showWarnings=FALSE)
z<-read.csv(gzfile(file.path(root,'zero_sum_draws.csv.gz')))
stopifnot(nrow(z)==200000,all(is.finite(z$sum_beta)),all(table(z$cell)==20000))
order<-c('none','exact','soft_center','prox1e-3','prox1e-6')
methods<-c('cox-rhs','prox-sslasso')
stopifnot(setequal(unique(z$constraint),order),setequal(unique(z$method),methods),
  all(table(z$constraint,z$method)==20000))
z$constraint<-factor(z$constraint,levels=rev(order))
z$method<-factor(z$method,levels=methods)
colours<-c('cox-rhs'='#2878A5','prox-sslasso'='#C18735')
method_labels<-c('cox-rhs'='Cox-RHS','prox-sslasso'='Prox-SSLASSO')
labels<-expression('None','Exact','Soft-center',
  'Prox ('*italic('\u03bb')[Sigma]*'=10'^{-3}*')','Prox ('*italic('\u03bb')[Sigma]*'=10'^{-6}*')')
theme_set(theme_bw(base_size=15,base_family='Times New Roman')+
  theme(text=element_text(colour='black'),axis.text=element_text(colour='black'),
    panel.grid.minor=element_blank(),panel.grid.major.y=element_blank(),
    plot.title=element_blank(),plot.margin=margin(8,12,8,8)))
save<-function(p,name,w,h){
  pdf_device<-function(...)grDevices::cairo_pdf(...,family='Times New Roman',
    symbolfamily=grDevices::cairoSymbolFont('Symbol'))
  png_device<-function(...)grDevices::png(...,type='cairo',units='in',res=300,family='Times New Roman',
    symbolfamily=grDevices::cairoSymbolFont('Symbol'))
  ggsave(file.path(out,paste0(name,'.pdf')),p,width=w,height=h,device=pdf_device)
  ggsave(file.path(out,paste0(name,'.png')),p,width=w,height=h,dpi=300,bg='white',device=png_device)
}
limits<-range(z$sum_beta);limits<-limits+c(-1,1)*diff(limits)*.04
sum_label<-expression('Posterior coefficient sum  '*sum(italic('\u03b2')[italic(j)],italic(j)==1,italic(p)))
for(method in methods) {
  colour<-colours[[method]]
  p<-ggplot(z[z$method==method,],aes(x=sum_beta,y=constraint))+
    geom_vline(xintercept=0,linetype='dashed',colour='grey45',linewidth=.55)+
    geom_boxplot(orientation='y',fill=colour,alpha=.58,width=.52,
      linewidth=.55,outlier.size=.65,outlier.alpha=.35)+
    scale_y_discrete(limits=rev(order),labels=rev(labels))+
    coord_cartesian(xlim=limits)+
    labs(x=sum_label,y=NULL,title=NULL)
  save(p,if(method=='cox-rhs') 'zero_sum_cox_rhs' else 'zero_sum_prox_sslasso',7.3,4.3)
}
combined<-ggplot(z,aes(x=sum_beta,y=constraint,fill=method,colour=method))+
  geom_vline(xintercept=0,linetype='dashed',colour='grey45',linewidth=.55)+
  geom_boxplot(orientation='y',position=position_dodge2(width=.8,preserve='single',reverse=TRUE),
    alpha=.58,width=.72,linewidth=.55,outlier.size=.65,outlier.alpha=.35)+
  scale_y_discrete(limits=rev(order),labels=rev(labels))+
  scale_fill_manual(values=colours,breaks=methods,labels=method_labels)+
  scale_colour_manual(values=colours,breaks=methods,labels=method_labels)+
  coord_cartesian(xlim=limits)+
  labs(x=sum_label,y=NULL,title=NULL,fill=NULL,colour=NULL)+
  guides(colour='none',fill=guide_legend(override.aes=list(colour='grey30',alpha=.58)))+
  theme(legend.position='top',legend.text=element_text(size=15),
    legend.key.width=grid::unit(1.1,'cm'))
stopifnot(nrow(ggplot_build(combined)$data[[2]])==10L)
save(combined,'zero_sum_combined',7.3,5.3)
writeLines(capture.output(sessionInfo()),file.path(out,'constraints_R_sessionInfo.txt'))
