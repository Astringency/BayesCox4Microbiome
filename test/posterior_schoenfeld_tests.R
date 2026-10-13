suppressPackageStartupMessages(library(survival))
source('scripts/check_t1d_ph.R')
# Non-MLE coefficients, tied failures, a simultaneous censored subject, and a
# clinical indicator. Tests reject any accidental auxiliary refit or wrong risk set.
time <- c(1,1,1,2,3,3,4,5,6,7,8,9)
status <- c(1,1,0,1,1,0,1,1,0,1,1,0)
X <- cbind(log_taxon=c(.4,1.1,-.2,.7,1.2,-.6,.2,.9,-1,.3,-.4,.6),
           clinical=c(0,1,1,0,1,0,0,1,0,1,0,1))
beta <- c(.18,-.33)
out <- tempfile();dir.create(out)
design <- data.frame(subject_index=seq_along(time),time,status,X)
write.csv(design,file.path(out,'design.csv'),row.names=FALSE)
write.csv(data.frame(feature_id=colnames(X),posterior_mean=beta),file.path(out,'coefficients.csv'),row.names=FALSE)
result <- run_posterior_ph(file.path(out,'design.csv'),file.path(out,'coefficients.csv'),out)
stopifnot(result$iterations==0, max(abs(result$coefficients-beta))<1e-12,result$max_residual_error<1e-8)
# The summed Schoenfeld residual equals the finite-difference Efron log-likelihood score.
ll <- function(b) {
  eta <- drop(X%*%b);val <- 0
  for (tt in unique(time[status==1])) {
    D <- which(time==tt & status==1);R <- which(time>=tt);d <- length(D)
    val <- val+sum(eta[D])-sum(log(sum(exp(eta[R]))-(0:(d-1))/d*sum(exp(eta[D]))))
  }
  val
}
score <- sapply(seq_along(beta),function(j) {
  hi<-lo<-beta;hi[j]<-hi[j]+1e-5;lo[j]<-lo[j]-1e-5;(ll(hi)-ll(lo))/2e-5
})
stopifnot(max(abs(colSums(posterior_schoenfeld(time,status,X,beta)$residuals)-score))<1e-7)
cat('Posterior-mean Schoenfeld checks passed: fixed coefficients, Efron ties, full risk sets, finite-difference score.\n')
