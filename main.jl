module BayesCox4Micro

using LinearAlgebra
using Random
using Statistics
using Distributions
using Distances
using Survival
using ProgressMeter
import AdvancedHMC
import LogDensityProblems
import Turing

export AbstractCoxPrior,
    AbstractCoxTarget,
    HorseshoePrior,
    RegularizedHorseshoePrior,
    BayesianLassoPrior,
    SpikeSlabLassoPrior,
    CoxProxLassoTarget,
    CoxProxSSLassoTarget,
    CoxRiskSet,
    CoxPreparedData,
    MHSamples,
    MultiMHSamples,
    MALAWorkspace,
    ProxGradientWorkspace,
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
    cox_score_gradient!,
    CoxGradientWorkspace,
    cox_loglik_events,
    project_sum0,
    sum0_moreau,
    sum0_moreau_gradient,
    project_l1_epigraph,
    project_l1_epigraph!,
    project_weighted_l1_epigraph,
    project_weighted_l1_epigraph!,
    l1_moreau,
    weighted_l1_moreau,
    logposterior,
    logdensity,
    gradient!,
    logdensity_and_gradient!,
    dimension,
    initial_state,
    state_names,
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
    dataframe_to_chains,
    save_run_outputs,
    save_diagnostic_outputs,
    diagnose_run

include("priors/abstract.jl")
include("priors/horseshoe.jl")
include("priors/regularized_horseshoe.jl")
include("priors/bayesian_lasso.jl")
include("priors/spike_slab_lasso.jl")

include("mainfuncs/types.jl")
include("mainfuncs/progress.jl")
include("mainfuncs/priors/prior_common.jl")
include("mainfuncs/data/data.jl")
include("mainfuncs/likelihood/cox_likelihood.jl")
include("mainfuncs/priors/horseshoe_methods.jl")
include("mainfuncs/priors/regularized_horseshoe_methods.jl")
include("mainfuncs/priors/bayesian_lasso_methods.jl")
include("mainfuncs/priors/spike_slab_lasso_methods.jl")
include("mainfuncs/priors/prox_lasso_methods.jl")
include("mainfuncs/priors/prox_spike_slab_lasso_methods.jl")
include("mainfuncs/posterior.jl")
include("mainfuncs/samplers/mh_sampler.jl")
include("mainfuncs/samplers/gibbs_sampler.jl")
include("mainfuncs/samplers/mala_sampler.jl")
include("mainfuncs/targets/cox_targets.jl")
include("mainfuncs/samplers/nuts_turing_sampler.jl")
include("mainfuncs/samplers/nuts_explicit_sampler.jl")
include("mainfuncs/api.jl")
include("mainfuncs/diagnostics/diagnostics.jl")

end # module
