# Plot current, audited run identities instead of mixing old fits or batch folders.
process_manifest <- function(path, output_dir=NULL) {
  all <- fread(path)
  needed <- c('source','suite','cell','replicate','method','p','sum_beta_sum_posterior_means')
  stopifnot(all(needed %in% names(all)))
  suite_filter <- arg_value('suite',NULL); cells <- split_csv(arg_value('cells',''))
  if(!is.null(suite_filter)) all<-all[suite==suite_filter]
  if(length(cells)) all<-all[cell %in% cells]
  if(length(METHOD_FILTER)) all<-all[method %in% METHOD_FILTER]
  if(length(SCENARIO_FILTER)) all<-all[cell %in% SCENARIO_FILTER]
  stopifnot(!anyDuplicated(all[,.(suite,cell,replicate)]))
  if(is.null(output_dir)) output_dir<-file.path(dirname(path),'public_figs')
  selected<-if(ALL_REPS) copy(all) else all[replicate==FOCUS_REP]
  setorder(selected,suite,cell,replicate)
  shard_arg<-arg_value('shard',NULL); suffix<-''
  if(!is.null(shard_arg)) {
    shard<-as.integer(strsplit(shard_arg,'/',fixed=TRUE)[[1]])
    stopifnot(length(shard)==2L,all(is.finite(shard)),shard[1]>=1,shard[1]<=shard[2])
    groups<-unique(selected[,.(suite,cell)])
    groups<-groups[(seq_len(nrow(groups))-1L)%%shard[2]+1L==shard[1]]
    selected<-merge(selected,groups,by=c('suite','cell'),sort=FALSE)
    setorder(selected,suite,cell,replicate)
    suffix<-sprintf('_shard_%d_of_%d',shard[1],shard[2])
  }
  if(!nrow(selected)) stop('No matching runs in manifest')
  cat(sprintf('Manifest: %s\nRuns: %d\nOutput: %s\n',path,nrow(selected),output_dir))
  if(DRY_RUN) {print(selected[,.(suite,cell,replicate,source)]);return(invisible(NULL))}
  receipts<-list()
  for(i in seq_len(nrow(selected))) {
    row<-selected[i];run<-row$source
    out<-file.path(output_dir,row$suite,row$cell,sprintf('rep%04d',row$replicate))
    dir.create(out,recursive=TRUE,showWarnings=FALSE)
    cat(sprintf('[%d/%d] %s rep%d\n',i,nrow(selected),row$cell,row$replicate));flush.console()
    diag_path<-file.path(run,'rank_diagnostics.csv')
    if(!file.exists(diag_path)) diag_path<-file.path(run,'diagnose','diagnostics_summary.csv')
    if(!file.exists(diag_path)) stop('Missing diagnostics for ',run)
    diag<-fread(diag_path);diag<-diag[grepl('^beta\\[',parameters)]
    post<-normalize_posterior(fread(file.path(run,'posterior_summary.csv')),run);nz<-posterior_nonzero(post)
    if(is.finite(row$nonzero) && row$nonzero>0 && nrow(nz)!=row$nonzero) stop('Truth/support mismatch: ',run)
    samples_path<-file.path(run,'samples_wide.csv')
    stopifnot(file.exists(samples_path),nrow(diag)==row$p)
    save_pdf_png(plot_rhat_ess(diag,row$p),file.path(out,'fig01_rhat_ess'),280,105)
    if(nrow(nz)) {
      cols<-nz$variable
      wide<-fread(samples_path,select=c('chain','iter',cols))
      samples<-melt(wide,id.vars=c('chain','iter'),measure.vars=cols,variable.name='parameter',value.name='value')
      stopifnot(all(is.finite(samples$value)),!anyDuplicated(wide[,.(chain,iter)]))
      ac<-compute_acf(samples)
      save_pdf_png(plot_trace_density_acf(samples,post,ac),file.path(out,'fig01_trace_density_acf'),280,24+40*nrow(nz))
      fwrite(ac,file.path(out,'acf_values.csv'))
    }
    fwrite(diag,file.path(out,'coefficient_diagnostics.csv'))
    fwrite(nz,file.path(out,'nonzero_truth.csv'))
    if(file.exists(file.path(run,'coefficient_audit.csv'))) fwrite(data.table(source=file.path(run,'coefficient_audit.csv'),md5=unname(tools::md5sum(file.path(run,'coefficient_audit.csv')))),file.path(out,'truth_source.csv'))
    parent<-dirname(out)
    group<-all[suite==row$suite & cell==row$cell & replicate<=SUM_REPS]
    sums<-as.numeric(group$sum_beta_sum_posterior_means)
    stopifnot(all(is.finite(sums)))
    if(length(sums)>=2 && !any(vapply(receipts,function(x)identical(x$cell,row$cell)&&identical(x$suite,row$suite),logical(1)))) {
      save_pdf_png(plot_coef_sum(sums),file.path(parent,'fig03_coef_sum'),155,105)
      fwrite(group[,.(replicate,sum_beta_sum_posterior_means)],file.path(parent,'fig03_coef_sum_data.csv'))
    }
    receipts[[i]]<-data.table(suite=row$suite,cell=row$cell,replicate=row$replicate,
      source=run,diagnostics=diag_path,output=out,coefficients=nrow(diag),nonzero=nrow(nz),
      diagnostics_md5=unname(tools::md5sum(diag_path)),posterior_md5=unname(tools::md5sum(file.path(run,'posterior_summary.csv'))))
    # An explicit run receipt keeps per-fit figures distinct from across-run fig03.
    fwrite(receipts[[i]],file.path(out,'figure_source.csv'))
  }
  fwrite(rbindlist(receipts),file.path(output_dir,paste0('figure_manifest_rep_',REP_ARG,suffix,'.csv')))
  writeLines(capture.output(sessionInfo()),file.path(output_dir,paste0('R_sessionInfo',suffix,'.txt')))
  cat('Completed all ',length(receipts),' selected fits.\n',sep='')
}
