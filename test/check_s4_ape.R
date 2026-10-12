# Independent cross-check of the exported Newick tree and Julia BM covariance.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 1L, requireNamespace("ape", quietly=TRUE))
tree <- ape::read.tree(file.path(args[1], "tree.newick"))
expected <- ape::vcv.phylo(tree, model="Brownian", corr=TRUE)
actual <- as.matrix(read.csv(file.path(args[1], "latent_covariance.csv"), check.names=FALSE))
labels <- paste0("x", seq_len(ncol(actual)))
stopifnot(length(tree$tip.label) == length(labels), ape::is.ultrametric(tree),
          all(tree$edge.length > 0), setequal(tree$tip.label, labels))
error <- max(abs(actual - expected[labels, labels]))
stopifnot(error < 1e-12)
cat("PASS: S4 covariance matches ape::vcv.phylo; tips=", length(labels),
    "; maximum absolute difference=", error, "\n", sep="")
