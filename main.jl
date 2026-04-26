module BayesCox4Micro

using LinearAlgebra
using Random
using Statistics
using Distributions
using Distances
using Survival
using ProgressMeter
import Turing

export AbstractCoxPrior,
    HorseshoePrior,
    RegularizedHorseshoePrior,
    BayesianLassoPrior,
    SpikeSlabLassoPrior,
    CoxRiskSet,
    CoxPreparedData,
    MHSamples,
    MultiMHSamples,
    CoxSamplerResult,
    SamplingDiagnostics,
    prepare_cox_data,
    cox_tau0,
    linear_tau0,
    estimate_p0_univariate_cox,
    cox_paper_prior,
    car_distance_matrix,
    car_similarity_matrix,
    car_weight_matrix,
    cox_logpartiallikelihood,
    cox_score_gradient,
    cox_loglik_events,
    project_sum0,
    sum0_moreau,
    sum0_moreau_gradient,
    project_l1_epigraph,
    project_weighted_l1_epigraph,
    l1_moreau,
    weighted_l1_moreau,
    logposterior,
    initial_state,
    sample_bayes_cox,
    posterior_beta_draws,
    posterior_logposterior_draws,
    posterior_beta_map,
    posterior_beta_summary,
    sampler_diagnostics,
    sslasso_inclusion_probabilities,
    concordance_index,
    result_to_chains,
    as_mcmc_chains,
    chains_to_wide_dataframe,
    samples_wide_dataframe,
    samples_long_dataframe,
    dataframe_to_chains,
    save_run_outputs,
    diagnose_run

include("priors/abstract.jl")
include("priors/horseshoe.jl")
include("priors/regularized_horseshoe.jl")
include("priors/bayesian_lasso.jl")
include("priors/spike_slab_lasso.jl")

include("mainfuncs/types.jl")
include("mainfuncs/prior_common.jl")
include("mainfuncs/data.jl")
include("mainfuncs/cox_likelihood.jl")
include("mainfuncs/horseshoe_methods.jl")
include("mainfuncs/regularized_horseshoe_methods.jl")
include("mainfuncs/bayesian_lasso_methods.jl")
include("mainfuncs/spike_slab_lasso_methods.jl")
include("mainfuncs/posterior.jl")
include("mainfuncs/mh_sampler.jl")
include("mainfuncs/gibbs_sampler.jl")
include("mainfuncs/mala_sampler.jl")
include("mainfuncs/turing_sampler.jl")
include("mainfuncs/api.jl")
include("mainfuncs/diagnostics.jl")

end # module
