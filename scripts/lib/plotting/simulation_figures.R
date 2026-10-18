# Shared publication panels; sourced by scripts/plot_simulation.R.
# Mathematical labels use R plotmath, rendered as typeset math in vector PDFs.
CHAIN_COLOURS <- c('#2878A5', '#D38B24', '#448C79', '#AA658C')
PLOT_BASE_SIZE <- 16
PLOT_SMALL_BASE_SIZE <- 16

plot_rhat_ess <- function(diag, p_dim) {
  d <- copy(diag)
  d[, index := parameter_index(parameters)]
  setorder(d,index)
  stopifnot(nrow(d)==p_dim, identical(d$index,seq_len(p_dim)))
  # Every coefficient is a point and an integer tick; labels are thinned only
  # for legibility. No coefficients or failed diagnostics are filtered out.
  label_at <- unique(c(1L,if(p_dim>=25L) seq(25L,p_dim,by=25L) else integer(),p_dim))
  if(p_dim < 25) label_at <- unique(c(1L,pretty(seq_len(p_dim)),p_dim))
  panels <- lapply(c('rhat','ess_bulk','ess_tail'),function(key) {
    vals <- d[[key]]
    if(key=='rhat') {
      threshold <- c(1.01,1.1); lim <- range(c(.99,1.11,vals[is.finite(vals)]))
      br <- sort(unique(c(.99,1,1.01,1.05,1.1,if(lim[2]>1.11) pretty(lim)[pretty(lim)>1.1] else numeric())))
      br <- br[br>=lim[1] & br<=lim[2]]
      title <- expression(hat(italic(R))); yl <- NULL
    } else {
      threshold <- 400; lim <- range(c(300,20000,vals[is.finite(vals)&vals>0]))
      title <- if(key=='ess_bulk') 'Bulk ESS' else 'Tail ESS'; yl <- NULL
    }
    d[, value := as.numeric(get(key))]
    missing <- sum(!is.finite(d$value) | (key!='rhat' & d$value<=0))
    g <- ggplot(d[is.finite(value) & (key=='rhat' | value>0)],aes(index,value))+
      geom_hline(yintercept=threshold,colour='#C33A3A',linetype='dashed',linewidth=.6)+
      geom_point(colour='#2878A5',size=1.65,alpha=.8)+
      scale_x_continuous(breaks=seq_len(p_dim),labels=function(x) ifelse(x %in% label_at,x,''),expand=expansion(mult=c(.02,.02)))+
      labs(x='Coefficient index',y=yl,title=title)+publication_theme(16)+
      theme(panel.grid.minor=element_blank(),panel.grid.major.x=element_blank(),
        axis.ticks.x=element_line(linewidth=.18),axis.ticks.length=grid::unit(1,'mm'),
        plot.title=element_text(hjust=.5,size=18),axis.text=element_text(size=13))
    if(key=='rhat') g<-g+scale_y_continuous(limits=lim,breaks=br)
    else g<-g+scale_y_log10(limits=lim,breaks=c(400,1000,5000,20000),labels=c('400','1,000','5,000','20,000'))
    if(missing>0) g<-g+labs(subtitle=sprintf('%d unavailable; see diagnostic CSV',missing))
    g
  })
  wrap_plots(panels,ncol=3)
}

plot_trace_density_acf <- function(samples,posterior,acf_data) {
  nz <- posterior_nonzero(posterior)
  if(!nrow(nz)) return(NULL)
  chains <- as.character(sort(unique(samples$chain)))
  colours <- setNames(rep(CHAIN_COLOURS,length.out=length(chains)),chains)
  panels <- list()
  for(i in seq_len(nrow(nz))) {
    par <- nz$variable[i]; truth <- as.numeric(nz$true_beta[i])
    z <- samples[parameter==par]; setorder(z,chain,iter)
    a <- acf_data[parameter==par]
    coef_label <- bquote(italic("\u03b2")[.(nz$beta_index[i])])
    base <- publication_theme(16)+theme(panel.grid.minor=element_blank(),
      legend.position='none',plot.margin=margin(4,5,4,5),
      axis.title=element_text(size=16),axis.text=element_text(size=13),
      plot.title=element_text(size=18,hjust=.5))
    colour_scale<-scale_colour_manual(values=colours,name='Chain')
    tr <- ggplot(z,aes(iter,value,colour=factor(chain)))+
      geom_line(linewidth=.18,alpha=.65,show.legend=FALSE)+
      geom_hline(yintercept=truth,colour='#C33A3A',linetype='dashed',linewidth=.5)+
      colour_scale+scale_x_continuous(breaks=unique(c(min(z$iter),pretty(z$iter,n=3)[pretty(z$iter,n=3)>min(z$iter)&pretty(z$iter,n=3)<max(z$iter)],max(z$iter))),labels=function(x)format(x,big.mark=',',trim=TRUE,scientific=FALSE))+
      labs(y=coef_label,x=if(i==nrow(nz)) 'Retained iteration' else NULL,
        title=if(i==1) 'Trace' else NULL)+base
    de <- ggplot(z,aes(value,colour=factor(chain)))+
      geom_density(linewidth=.65,show.legend=FALSE)+
      geom_vline(xintercept=truth,colour='#C33A3A',linetype='dashed',linewidth=.5)+
      colour_scale+labs(y=NULL,x=if(i==nrow(nz)) 'Coefficient value' else NULL,
        title=if(i==1) 'Density' else NULL)+base
    ac <- ggplot(a,aes(lag,acf,colour=factor(chain)))+
      geom_hline(yintercept=0,colour='grey60',linetype='dotted',linewidth=.35)+
      geom_line(linewidth=.55,show.legend=(i==nrow(nz)))+colour_scale+scale_y_continuous(limits=c(min(-.12,min(a$acf,na.rm=TRUE)-.03),1.02),n.breaks=4)+
      labs(y=NULL,x=if(i==nrow(nz)) 'Lag' else NULL,title=if(i==1) 'ACF' else NULL)+base
    if(i<nrow(nz)) {
      tr<-tr+theme(axis.text.x=element_blank());ac<-ac+theme(axis.text.x=element_blank())
    }
    panels<-c(panels,list(tr,de,ac))
  }
  # Draw a dedicated shared key: guide collection can lose line keys when
  # most small panels suppress their legends. The explicit key is invariant.
  panels<-lapply(panels,function(p)p+guides(colour='none'))
  key<-data.frame(chain=factor(chains,levels=chains),x=seq_along(chains))
  legend<-ggplot(key,aes(y=.5))+
    geom_segment(aes(x=x-.25,xend=x-.02,yend=.5,colour=chain),linewidth=.8,show.legend=FALSE)+
    geom_text(aes(x=x+.12,label=chain),family=PLOT_FONT_FAMILY,size=5.4)+
    annotate('text',x=.35,y=.5,label='Chain',hjust=1,family=PLOT_FONT_FAMILY,size=5.4)+
    scale_colour_manual(values=colours)+
    coord_cartesian(xlim=c(-2.8,length(chains)+3.2),ylim=c(0,1),clip='off')+
    theme_void()+theme(plot.margin=margin(0,0,0,0),legend.position='none')
  wrap_plots(wrap_plots(panels,ncol=3,widths=c(1.45,1,1)),legend,
    ncol=1,heights=c(40*nrow(nz),12))
}
