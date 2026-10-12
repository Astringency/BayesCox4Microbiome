# Reproducible static research figures from the reviewed S1--S4 summaries.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L)
root <- args[1]
out <- file.path(root, 'figures')
dir.create(out, recursive=TRUE, showWarnings=FALSE)
all <- read.csv(file.path(root, 'performance_summary.csv'), check.names=FALSE)
x <- subset(all, estimate=='mean' & selection=='all_eligible')
ord <- c('cox-rhs','prox-sslasso','Independent','Correct CAR','Misspecified CAR')
x <- x[order(x$setting, match(x$comparison, ord)),]
stopifnot(nrow(x)==9L, all(x$n_replicates==50L))
labels <- paste(x$setting, c('RHS','SSL','RHS','SSL','RHS','SSL','Independent','Correct CAR','Wrong CAR'))
colors <- ifelse(x$comparison=='prox-sslasso','#C47724','#24638B')
shapes <- ifelse(x$comparison=='prox-sslasso',17,19)
render <- function(stem, width, height, draw) {
  pdf(file.path(out,paste0(stem,'.pdf')),width=width,height=height,family='Helvetica',useDingbats=FALSE)
  draw(); dev.off()
  png(file.path(out,paste0(stem,'.png')),width=width*160,height=height*160,res=160)
  draw(); dev.off()
}
render('scenarios_performance',11,8,function() {
  par(mfrow=c(2,2),mar=c(4,9,3,1),oma=c(2,0,2,0),las=1,col.axis='#343A40',col.lab='#343A40')
  for (metric in c('Cindex_test','MSE','coverage_95_nonzero','empirical_FDP')) {
    center <- x[[paste0(metric,'_mean')]]
    error <- 1.96*x[[paste0(metric,'_mcse')]]
    if(metric %in% c('coverage_95_nonzero','empirical_FDP')) { center <- 100*center; error <- 100*error }
    low <- center-error; high <- center+error
    title <- switch(metric,Cindex_test='Test C-index',MSE='Coefficient MSE',
                    coverage_95_nonzero='Signal coverage (%)',empirical_FDP='Empirical FDP (%)')
    # Monte Carlo intervals can cross 0 or 1; display their full range.
    limits <- range(c(low,high,if(metric=='coverage_95_nonzero') 95 else NULL))
    padding <- max(diff(limits)*.12,.002)
    yy <- 9:1
    plot(center,yy,xlim=limits+c(-padding,padding),ylim=c(.5,9.5),yaxt='n',
         ylab='',xlab=title,pch=NA,bty='l',main=title,cex.main=.95)
    abline(h=yy,col='#E8EAED',lwd=.6)
    abline(h=c(3.5,5.5,7.5),col='#BFC5CB',lty=3)
    if(metric=='coverage_95_nonzero') abline(v=95,col='#565B60',lty=2)
    segments(low,yy,high,yy,col=colors,lwd=1.5)
    points(center,yy,pch=shapes,col=colors,cex=1)
    axis(2,at=yy,labels=labels,tick=FALSE,cex.axis=.85)
  }
  mtext('S1-S4 simulation performance | posterior mean estimates | 50 repetitions per setting',outer=TRUE,side=3,line=.4,cex=1.05)
  mtext('Points: means across repetitions; intervals: mean +/- 1.96 Monte Carlo SE. All eligible fits included.',outer=TRUE,side=1,line=.5,cex=.85)
})
paired <- read.csv(file.path(root,'S4_paired_contrasts.csv'),check.names=FALSE)
y <- subset(paired,estimate=='mean' & selection=='all_eligible_pairs')
y <- y[match(c('correct','misspecified'),y$setting),]
stopifnot(nrow(y)==2L, all(y$n_pairs==50L))
render('S4_paired_effects',11,6,function() {
  par(mfrow=c(2,2),mar=c(4,7,3,1),oma=c(2,0,2,0),las=1)
  for(metric in c('Cindex_test','MSE','coverage_95_nonzero','empirical_FDP')) {
    center <- y[[paste0(metric,'_mean_difference')]]
    error <- 1.96*y[[paste0(metric,'_paired_mcse')]]
    if(metric %in% c('coverage_95_nonzero','empirical_FDP')) { center <- 100*center; error <- 100*error }
    limits <- range(c(0,center-error,center+error)); padding <- max(diff(limits)*.18,.0001)
    title <- switch(metric,Cindex_test='Test C-index difference',MSE='MSE difference',
                    coverage_95_nonzero='Signal coverage difference (percentage points)',empirical_FDP='FDP difference (percentage points)')
    plot(center,c(2,1),xlim=limits+c(-padding,padding),ylim=c(.5,2.5),yaxt='n',
         ylab='',xlab='CAR - Independent',pch=NA,bty='l',main=title,cex.main=.95)
    abline(h=c(1,2),col='#E8EAED',lwd=.6); abline(v=0,col='#343A40',lty=2)
    segments(center-error,c(2,1),center+error,c(2,1),col='#24638B',lwd=1.5)
    points(center,c(2,1),col='#24638B',pch=c(19,1),cex=1.1)
    axis(2,at=c(2,1),labels=c('Correct CAR','Wrong CAR'),tick=FALSE,cex.axis=.9)
  }
  mtext('S4: paired CAR effects on Brownian phylogenetic data',outer=TRUE,side=3,line=.4,cex=1.1)
  mtext('50 paired repetitions; intervals: mean difference +/- 1.96 paired Monte Carlo SE. Zero marks no difference.',outer=TRUE,side=1,line=.5,cex=.85)
})
cat('Saved four PDF/PNG figure files to',out,'\n')
