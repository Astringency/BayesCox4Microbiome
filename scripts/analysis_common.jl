using BayesCox4Micro
using CSV
using DataFrames
using Distributions
using JSON
using LinearAlgebra
using Printf

if !haskey(ENV, "GKSwstype") && !haskey(ENV, "DISPLAY")
    ENV["GKSwstype"] = "100"
end

import Plots
using PosteriorStats
using Random
using Statistics

include(joinpath(@__DIR__, "validation_common.jl"))
include(joinpath(@__DIR__, "simulation_coverage.jl"))
include(joinpath(@__DIR__, "simulation_scenarios.jl"))

const DEFAULT_BETA0 = [1.0, -1.0]
const DEFAULT_BETA6 = [2.08, 1.31, -1.39, -1.86, 2.12, -2.26]
const DEFAULT_BETA12 = [2.08, -1.41, -1.39, -1.15, 2.12, 2.51, 1.31, -1.95, -1.86, 1.93, -1.34, -0.85]

function _unique_cli_keys(groups...)
    out = String[]
    seen = Set{String}()
    for group in groups
        for key in group
            value = String(key)
            value in seen && continue
            push!(seen, value)
            push!(out, value)
        end
    end
    return Tuple(out)
end

const SAMPLER_GENERAL_CLI_KEYS = (
    "sampler",
    "samples",
    "warmup",
    "thin",
    "chains",
    "multichains",
    "target-accept",
    "noncentered",
    "init",
    "seed",
    "progress",
    "approx",
    "approx-point",
)
const NUTS_SAMPLER_CLI_KEYS = (
    "nuts-backend",
    "max-depth",
)
const HMC_SAMPLER_CLI_KEYS = (
    "hmc-n-leapfrog",
    "hmc-step-size",
)
const MALA_SAMPLER_CLI_KEYS = (
    "mala-step-size",
    "mala-target-accept",
    "mala-adapt-every",
    "mala-min-step",
    "mala-max-step",
    "mala-precondition",
    "mala-precondition-mode",
    "mala-precondition-diag",
    "mala-metric-ridge",
    "mala-metric-max-cholesky-tries",
    "mala-stepsize-schedule",
    "mala-sgld-c0",
    "mala-sgld-target-accept",
    "mala-sgld-accept-low",
    "mala-sgld-accept-high",
    "mala-sgld-adapt-c0",
    "mala-sgld-adapt-every",
    "mala-sgld-c0-min",
    "mala-sgld-c0-max",
)
const PROX_BLOCK_SAMPLER_CLI_KEYS = (
    "prox-block-step-size",
    "prox-block-min-step-size",
    "prox-block-max-step-size",
    "prox-block-adapt-every",
    "prox-block-divergence-threshold",
    "prox-block-metric",
)
const SAMPLER_HYPERPARAMETER_CLI_KEYS = _unique_cli_keys(
    SAMPLER_GENERAL_CLI_KEYS,
    NUTS_SAMPLER_CLI_KEYS,
    HMC_SAMPLER_CLI_KEYS,
    MALA_SAMPLER_CLI_KEYS,
    PROX_BLOCK_SAMPLER_CLI_KEYS,
)

const HORSESHOE_PRIOR_CLI_KEYS = (
    "hs-tau-scale",
    "hs-local-scale",
)
const RHS_PRIOR_CLI_KEYS = (
    "rhs-slab-df",
    "rhs-slab-s2",
    "rhs-tau-scale",
    "rhs-local-scale",
    "rhs-pseudo-sigma-df",
    "rhs-pseudo-sigma-scale",
)
const LASSO_PRIOR_CLI_KEYS = (
    "lasso-lambda",
    "lasso-lambda2-shape",
    "lasso-lambda2-rate",
)
const SSLASSO_PRIOR_CLI_KEYS = (
    "sslasso-spike-scale",
    "sslasso-slab-scale",
    "sslasso-theta-a",
    "sslasso-theta-b",
)
const PROXIMAL_L1_PRIOR_CLI_KEYS = (
    "prox-l1-lambda",
    "alpha-prior-shape",
    "alpha-prior-scale",
    "alpha-prior-kind",
    "prox-root-tol",
    "prox-root-maxiter",
)
const CAR_PRIOR_CLI_KEYS = (
    "use-car",
    "car-distance",
    "car-weight-quantile",
    "car-precision",
    "sample-car-strength",
    "car-strength-prior-a",
    "car-strength-prior-b",
    "car-degree-scale",
    "car-kappa-init",
)
const PRIOR_HYPERPARAMETER_CLI_KEYS = _unique_cli_keys(
    HORSESHOE_PRIOR_CLI_KEYS,
    RHS_PRIOR_CLI_KEYS,
    LASSO_PRIOR_CLI_KEYS,
    SSLASSO_PRIOR_CLI_KEYS,
    PROXIMAL_L1_PRIOR_CLI_KEYS,
    CAR_PRIOR_CLI_KEYS,
)

const COX_MODEL_CLI_KEYS = (
    "p0",
    "p0-alpha",
    "constraint",
    "prox-sum0-lambda",
    "standardize",
    "tie-method",
)
const DIAGNOSTIC_CLI_KEYS = (
    "diagnose",
    "diagnose-format",
    "diagnose-top-k-beta",
    "diagnose-max-lag",
)
const TUNING_CLI_KEYS = (
    "tune_priorsparams",
    "tune_algparams",
    "cv-folds",
    "outer-cv-folds",
    "nested-cv",
    "tune-seed",
    "tune-run-best",
    "prior-configs",
)
const COMMON_ANALYSIS_CLI_KEYS = _unique_cli_keys(
    SAMPLER_HYPERPARAMETER_CLI_KEYS,
    PRIOR_HYPERPARAMETER_CLI_KEYS,
    COX_MODEL_CLI_KEYS,
    DIAGNOSTIC_CLI_KEYS,
    TUNING_CLI_KEYS,
)
const BASELINE_PARAMETER_CLI_KEYS = (
    "weibull-rate",
    "weibull-lambda",
    "weibull-scale",
    "weibull-shape",
    "weibull-nu",
    "gompertz-rate",
    "gompertz-lambda",
    "gompertz-scale",
    "gompertz-shape",
    "gompertz-alpha",
    "exponential-rate",
    "exponential-lambda",
    "exponential-scale",
    "exponential-shape",
    "baseline-rate",
    "baseline-lambda",
    "baseline-scale",
    "baseline-shape",
    "baseline-nu",
    "baseline-alpha",
)

function parse_cli_args(args = ARGS)
    out = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = arg[3:end]
            if occursin("=", key)
                k, v = split(key, "="; limit = 2)
                out[k] = v
            elseif i < length(args) && !startswith(args[i + 1], "--")
                out[key] = args[i + 1]
                i += 1
            else
                out[key] = "true"
            end
        end
        i += 1
    end
    return out
end

function _collect_cli_key!(allowed::Set{String}, key::AbstractString)
    push!(allowed, String(key))
    return allowed
end

function _collect_cli_key!(allowed::Set{String}, keys)
    for key in keys
        _collect_cli_key!(allowed, key)
    end
    return allowed
end

function require_known_args!(args, valid_keys)
    allowed = _collect_cli_key!(Set{String}(), valid_keys)
    unknown = sort([key for key in keys(args) if !(key in allowed)])
    isempty(unknown) && return nothing
    formatted = join("--" .* unknown, ", ")
    throw(ArgumentError("unknown command line argument(s): $formatted"))
end

arg(args, key, default::String) = get(args, key, default)
arg(args, key, default::Int) = parse(Int, get(args, key, string(default)))
arg(args, key, default::Float64) = parse(Float64, get(args, key, string(default)))
arg(args, key, default::Bool) = lowercase(get(args, key, string(default))) in ("true", "1", "yes", "y")

function optional_float_arg(args, keys...)
    for key in keys
        haskey(args, key) && return parse(Float64, args[key])
    end
    return nothing
end

function sampler_hyperparameter_kwargs(args)
    mala_adapt_every = arg(args, "mala-adapt-every", 50)
    return (
        sampler = Symbol(arg(args, "sampler", "nuts")),
        nuts_backend = Symbol(arg(args, "nuts-backend", "turing")),
        n_samples = arg(args, "samples", 500),
        warmup = arg(args, "warmup", 200),
        thin = arg(args, "thin", 1),
        chains = arg(args, "chains", 4),
        multichains = arg(args, "multichains", false),
        target_accept = arg(args, "target-accept", 0.8),
        max_depth = arg(args, "max-depth", 10),
        hmc_n_leapfrog = arg(args, "hmc-n-leapfrog", 10),
        hmc_step_size = optional_float_arg(args, "hmc-step-size"),
        noncentered = arg(args, "noncentered", false),
        seed = arg(args, "seed", 1),
        progress = arg(args, "progress", true),
        mala_step_size = arg(args, "mala-step-size", 1e-4),
        mala_target_accept = arg(args, "mala-target-accept", 0.574),
        mala_adapt_every = mala_adapt_every,
        mala_min_step = arg(args, "mala-min-step", 1e-8),
        mala_max_step = arg(args, "mala-max-step", 1.0),
        mala_precondition = arg(args, "mala-precondition", false),
        mala_precondition_mode = Symbol(arg(args, "mala-precondition-mode", "auto")),
        mala_precondition_diag = arg(args, "mala-precondition-diag", false),
        mala_metric_ridge = arg(args, "mala-metric-ridge", 1e-6),
        mala_metric_max_cholesky_tries = arg(args, "mala-metric-max-cholesky-tries", 8),
        mala_stepsize_schedule = Symbol(arg(args, "mala-stepsize-schedule", "adaptive_mcmc")),
        mala_sgld_c0 = arg(args, "mala-sgld-c0", 1.0),
        mala_sgld_target_accept = arg(args, "mala-sgld-target-accept", 0.5),
        mala_sgld_accept_low = arg(args, "mala-sgld-accept-low", 0.40),
        mala_sgld_accept_high = arg(args, "mala-sgld-accept-high", 0.60),
        mala_sgld_adapt_c0 = arg(args, "mala-sgld-adapt-c0", true),
        mala_sgld_adapt_every = arg(args, "mala-sgld-adapt-every", mala_adapt_every),
        mala_sgld_c0_min = arg(args, "mala-sgld-c0-min", 1e-12),
        mala_sgld_c0_max = arg(args, "mala-sgld-c0-max", 1e12),
        prox_block_step_size = optional_float_arg(args, "prox-block-step-size"),
        prox_block_min_step_size = arg(args, "prox-block-min-step-size", 1e-8),
        prox_block_max_step_size = arg(args, "prox-block-max-step-size", 1.0),
        prox_block_adapt_every = arg(args, "prox-block-adapt-every", 50),
        prox_block_divergence_threshold = arg(args, "prox-block-divergence-threshold", 1000.0),
        prox_block_metric = Symbol(arg(args, "prox-block-metric", "unit_diag")),
        approx = arg(args, "approx", false),
        approx_point = Symbol(arg(args, "approx-point", "zero")),
    )
end

function car_hyperparameter_kwargs(args)
    return (
        use_car = arg(args, "use-car", true),
        car_distance = Symbol(arg(args, "car-distance", "bray")),
        car_weight_quantile = arg(args, "car-weight-quantile", 0.75),
        car_precision = arg(args, "car-precision", 1.0),
        sample_car_strength = arg(args, "sample-car-strength", false),
        car_strength_prior_a = arg(args, "car-strength-prior-a", 1.0),
        car_strength_prior_b = arg(args, "car-strength-prior-b", 1.0),
        car_degree_scale = Symbol(arg(args, "car-degree-scale", "mean_active")),
        car_kappa_init = arg(args, "car-kappa-init", 0.5),
    )
end

function prior_hyperparameter_kwargs(args)
    return (
        rhs_slab_df = something(optional_float_arg(args, "rhs-slab-df"), 4.0),
        rhs_slab_s2 = something(optional_float_arg(args, "rhs-slab-s2"), 2.0),
        hs_tau_scale = something(optional_float_arg(args, "hs-tau-scale"), 1.0),
        hs_local_scale = something(optional_float_arg(args, "hs-local-scale"), 1.0),
        rhs_tau_scale = something(optional_float_arg(args, "rhs-tau-scale"), 1.0),
        rhs_local_scale = something(optional_float_arg(args, "rhs-local-scale"), 1.0),
        rhs_pseudo_sigma_df = something(optional_float_arg(args, "rhs-pseudo-sigma-df"), 3.0),
        rhs_pseudo_sigma_scale = something(optional_float_arg(args, "rhs-pseudo-sigma-scale"), 10.0),
        lasso_lambda = optional_float_arg(args, "lasso-lambda"),
        lasso_lambda2_shape = something(optional_float_arg(args, "lasso-lambda2-shape"), 1.0),
        lasso_lambda2_rate = something(optional_float_arg(args, "lasso-lambda2-rate"), 0.1),
        sslasso_spike_scale = optional_float_arg(args, "sslasso-spike-scale"),
        sslasso_slab_scale = optional_float_arg(args, "sslasso-slab-scale"),
        sslasso_theta_a = something(optional_float_arg(args, "sslasso-theta-a"), 1.0),
        sslasso_theta_b = something(optional_float_arg(args, "sslasso-theta-b"), 1.0),
        prox_l1_lambda = arg(args, "prox-l1-lambda", 1e-3),
        alpha_prior_shape = optional_float_arg(args, "alpha-prior-shape"),
        alpha_prior_scale = optional_float_arg(args, "alpha-prior-scale"),
        alpha_prior_kind = Symbol(arg(args, "alpha-prior-kind", "invgamma")),
        prox_root_tol = arg(args, "prox-root-tol", 1e-10),
        prox_root_maxiter = arg(args, "prox-root-maxiter", 100),
    )
end

function prior_distribution_hyperparameter_kwargs(kwargs)
    return (
        rhs_slab_df = kwargs.rhs_slab_df,
        rhs_slab_s2 = kwargs.rhs_slab_s2,
        hs_tau_scale = kwargs.hs_tau_scale,
        hs_local_scale = kwargs.hs_local_scale,
        rhs_tau_scale = kwargs.rhs_tau_scale,
        rhs_local_scale = kwargs.rhs_local_scale,
        rhs_pseudo_sigma_df = kwargs.rhs_pseudo_sigma_df,
        rhs_pseudo_sigma_scale = kwargs.rhs_pseudo_sigma_scale,
        lasso_lambda = kwargs.lasso_lambda,
        lasso_lambda2_shape = kwargs.lasso_lambda2_shape,
        lasso_lambda2_rate = kwargs.lasso_lambda2_rate,
        sslasso_spike_scale = kwargs.sslasso_spike_scale,
        sslasso_slab_scale = kwargs.sslasso_slab_scale,
        sslasso_theta_a = kwargs.sslasso_theta_a,
        sslasso_theta_b = kwargs.sslasso_theta_b,
    )
end

function prior_hyperparameter_metadata(kwargs)
    d = kwargs isa AbstractDict ? kwargs : Dict{Symbol,Any}(pairs(kwargs))
    return Dict(
        "rhs_slab_df" => d[:rhs_slab_df],
        "rhs_slab_s2" => d[:rhs_slab_s2],
        "hs_tau_scale" => d[:hs_tau_scale],
        "hs_local_scale" => d[:hs_local_scale],
        "rhs_tau_scale" => d[:rhs_tau_scale],
        "rhs_local_scale" => d[:rhs_local_scale],
        "rhs_pseudo_sigma_df" => d[:rhs_pseudo_sigma_df],
        "rhs_pseudo_sigma_scale" => d[:rhs_pseudo_sigma_scale],
        "lasso_lambda" => d[:lasso_lambda],
        "lasso_lambda2_shape" => d[:lasso_lambda2_shape],
        "lasso_lambda2_rate" => d[:lasso_lambda2_rate],
        "sslasso_spike_scale" => d[:sslasso_spike_scale],
        "sslasso_slab_scale" => d[:sslasso_slab_scale],
        "sslasso_theta_a" => d[:sslasso_theta_a],
        "sslasso_theta_b" => d[:sslasso_theta_b],
        "prox_l1_lambda" => d[:prox_l1_lambda],
        "alpha_prior_shape" => d[:alpha_prior_shape],
        "alpha_prior_scale" => d[:alpha_prior_scale],
        "alpha_prior_kind" => string(d[:alpha_prior_kind]),
        "prox_root_tol" => d[:prox_root_tol],
        "prox_root_maxiter" => d[:prox_root_maxiter],
    )
end

function cox_model_cli_options(args; default_standardize = false)
    return (
        p0_arg = parse_optional_p0(args; default = nothing),
        p0_alpha = arg(args, "p0-alpha", 0.05),
        constraint = Symbol(arg(args, "constraint", "soft_center")),
        prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3),
        standardize = arg(args, "standardize", Bool(default_standardize)),
        tie_method = parse_tie_method(args),
    )
end

function diagnostic_cli_options(args)
    return (
        diagnose = arg(args, "diagnose", true),
        diagnose_formats = get(args, "diagnose-format", "pdf"),
        diagnose_top_k_beta = arg(args, "diagnose-top-k-beta", 10),
        diagnose_max_lag = arg(args, "diagnose-max-lag", 50),
    )
end

function tuning_cli_options(args; realdata::Bool = false)
    return (
        tune_priorparams = arg(args, "tune_priorsparams", realdata),
        tune_algparams = arg(args, "tune_algparams", false),
        cv_folds = arg(args, "cv-folds", 5),
        outer_cv_folds = arg(args, "outer-cv-folds", 5),
        nested_cv = arg(args, "nested-cv", realdata),
        tune_seed = arg(args, "tune-seed", 1),
    )
end

function add_prior_hyperparameter_columns!(table::DataFrame, kwargs)
    meta = kwargs isa AbstractDict ? kwargs : prior_hyperparameter_metadata(kwargs)
    for (key, value) in pairs(meta)
        table[!, Symbol(key)] = fill(value === nothing ? missing : value, nrow(table))
    end
    return table
end

function split_symbols(s::AbstractString)
    return Symbol.(strip.(split(s, ",")))
end

function split_ints(s::AbstractString)
    return parse.(Int, strip.(split(s, ",")))
end

function split_strings(s::AbstractString)
    return strip.(split(s, ","))
end

function option_value(args, default::AbstractString, keys::AbstractString...)
    for key in keys
        haskey(args, key) && return args[key]
    end
    return default
end

function parse_tie_method(args; default::AbstractString = "efron")
    value = lowercase(strip(option_value(args, default, "tie-method")))
    value in ("efron", "breslow") ||
        throw(ArgumentError("--tie-method must be efron or breslow."))
    return Symbol(value)
end

function parse_optional_p0(args; default = nothing)
    haskey(args, "p0") || return default
    value = lowercase(strip(args["p0"]))
    value in ("auto", "none", "missing") && return nothing
    value == "truth" && return :truth
    return parse(Float64, value)
end

ensure_dir(path::AbstractString) = (mkpath(path); path)

const DEFAULT_BASELINE_SETTINGS = Dict(
    :exponential => (rate = 1 / 14, shape = nothing, censoring_upper = 64.0),
    :weibull => (rate = 1 / 144, shape = 2.0, censoring_upper = 35.0),
    :gompertz => (rate = 1 / 25, shape = 0.1, censoring_upper = 35.0),
)

function baseline_default_settings(baseline::Symbol)
    haskey(DEFAULT_BASELINE_SETTINGS, baseline) ||
        throw(ArgumentError("baseline must be :weibull, :exponential, or :gompertz."))
    return DEFAULT_BASELINE_SETTINGS[baseline]
end

function baseline_parameters(baseline::Symbol; rate = nothing, scale = nothing, shape = nothing)
    defaults = baseline_default_settings(baseline)
    if rate !== nothing && scale !== nothing
        throw(ArgumentError("pass either baseline rate or reciprocal-rate scale, not both."))
    end
    rate_value =
        rate !== nothing ? Float64(rate) :
        scale !== nothing ? inv(Float64(scale)) :
        defaults.rate
    rate_value > 0 || throw(ArgumentError("baseline rate must be positive."))

    if baseline == :exponential
        shape === nothing || throw(ArgumentError("exponential baseline does not use a shape parameter."))
        return (rate = rate_value, shape = nothing)
    end

    shape_value = shape === nothing ? defaults.shape : Float64(shape)
    shape_value > 0 || throw(ArgumentError("baseline shape must be positive."))
    return (rate = rate_value, shape = shape_value)
end

baseline_censoring_upper(baseline::Symbol) = baseline_default_settings(baseline).censoring_upper

function baseline_parameter_args(args, baseline::Symbol)
    prefix = string(baseline)
    rate = optional_float_arg(args, "$prefix-rate", "$prefix-lambda", "baseline-rate", "baseline-lambda")
    scale = optional_float_arg(args, "$prefix-scale", "baseline-scale")
    shape =
        baseline == :weibull ?
        optional_float_arg(args, "weibull-shape", "weibull-nu", "baseline-shape", "baseline-nu") :
        baseline == :gompertz ?
        optional_float_arg(args, "gompertz-shape", "gompertz-alpha", "baseline-shape", "baseline-alpha") :
        optional_float_arg(args, "exponential-shape", "baseline-shape")
    return (baseline_rate = rate, baseline_scale = scale, baseline_shape = shape)
end

function true_beta(p::Integer, nonzero::Integer)
    values =
        nonzero == 6 ? DEFAULT_BETA6 :
        nonzero == 12 ? DEFAULT_BETA12 :
        nonzero % 2 == 0 ? vcat(
            fill(DEFAULT_BETA0[1], nonzero ÷ 2),
            fill(DEFAULT_BETA0[2], nonzero ÷ 2),
        ) :
        throw(ArgumentError("nonzero must be 6/12 or a even number."))

    paper_idx = nonzero in (6, 12) ? collect(18:2:(16 + 2nonzero)) : Int[]
    idx =
        !isempty(paper_idx) && maximum(paper_idx) <= p ? paper_idx :
        nonzero <= p ? collect(1:nonzero) :
        throw(ArgumentError("p=$p is too small for nonzero=$nonzero."))
    beta = zeros(Float64, p)
    beta[idx] .= values
    return beta, idx
end

function paper_covariance(p::Integer, index::AbstractVector{<:Integer})
    p > 0 || throw(ArgumentError("p must be positive."))
    idx = collect(Int, index)
    all(i -> 1 <= i <= p, idx) ||
        throw(ArgumentError("covariance index entries must be between 1 and p."))
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("covariance index entries must be unique."))

    sigma = Matrix{Float64}(I, p, p)
    for a in 1:(length(idx) - 1)
        i = idx[a]
        for b in (a + 1):length(idx)
            j = idx[b]
            sigma[i, j] = 0.75 - 0.015 * abs(i - j)
            sigma[j, i] = sigma[i, j]
        end
    end
    return Symmetric(sigma)
end

function softmax_rows(U::AbstractMatrix)
    X = similar(U, Float64)
    for i in axes(U, 1)
        row = view(U, i, :)
        m = maximum(row)
        denom = sum(exp.(row .- m))
        @views X[i, :] .= exp.(row .- m) ./ denom
    end
    return X
end

function baseline_event_time(
    rng::AbstractRNG,
    eta::Real,
    baseline::Symbol;
    rate = nothing,
    scale = nothing,
    shape = nothing,
)
    params = baseline_parameters(baseline; rate = rate, scale = scale, shape = shape)
    target = -log(rand(rng)) / exp(eta)
    if baseline == :exponential
        return target / params.rate
    elseif baseline == :weibull
        return (target / params.rate)^(1 / params.shape)
    elseif baseline == :gompertz
        return log1p(params.shape * target / params.rate) / params.shape
    else
        throw(ArgumentError("baseline must be :weibull, :exponential, or :gompertz."))
    end
end

function simulation_event_process(
    rng::AbstractRNG,
    n::Integer,
    p::Integer,
    nonzero::Integer;
    baseline::Symbol = :weibull,
    baseline_rate = nothing,
    baseline_scale = nothing,
    baseline_shape = nothing,
    scenario = :baseline,
    weak_multiplier = 0.35,
    null_correlation = 0.6,
)
    scenario = resolve_simulation_scenario(scenario)
    beta, idx = true_beta(p, nonzero)
    if scenario == :weak
        0 < weak_multiplier <= 1 || throw(ArgumentError("weak multiplier must be in (0,1]"))
        beta .*= weak_multiplier
    end
    latent = scenario_latent_parameters(p, idx, scenario; null_correlation)
    U = rand(rng, MvNormal(latent.mu, latent.covariance), n)'
    X = softmax_rows(U)
    eta = log.(X) * beta
    params = baseline_parameters(baseline; rate = baseline_rate, scale = baseline_scale, shape = baseline_shape)
    event_time = [baseline_event_time(rng, eta[i], baseline; rate = params.rate, shape = params.shape) for i in 1:n]
    return (X = X, beta = beta, nonzero_index = idx, eta = eta, event_time = Float64.(event_time),
        scenario = scenario, paired_nulls = latent.paired_nulls, latent_covariance = latent.covariance)
end

function random_censoring_times(
    rng::AbstractRNG,
    n::Integer;
    upper::Real,
)
    upper_value = Float64(upper)
    upper_value > 0 || throw(ArgumentError("censoring upper bound must be positive."))
    return upper_value .* rand(rng, n)
end

function apply_censoring(event_time::AbstractVector{<:Real}, censor_time::AbstractVector{<:Real})
    length(event_time) == length(censor_time) ||
        throw(DimensionMismatch("event_time and censor_time must have the same length."))
    event_time_f = Float64.(event_time)
    censor_time_f = Float64.(censor_time)
    time = min.(event_time_f, censor_time_f)
    event = Int.(event_time_f .<= censor_time_f)
    if count(!=(0), event) == 0
        idx = argmin(event_time_f)
        event[idx] = 1
        time[idx] = event_time_f[idx]
        censor_time_f[idx] = event_time_f[idx]
    end
    return (time = Float64.(time), event = event, censor_time = censor_time_f)
end

function uniform_censoring_rate(event_time::AbstractVector{<:Real}, upper::Real)
    upper_value = Float64(upper)
    upper_value > 0 || throw(ArgumentError("uniform censoring upper bound must be positive."))
    return mean(min.(Float64.(event_time) ./ upper_value, 1.0))
end

function calibrate_uniform_censoring_upper(
    event_time::AbstractVector{<:Real},
    target_censoring_rate::Real;
    tol::Real = 1e-4,
    maxiter::Integer = 100,
)
    target = Float64(target_censoring_rate)
    0 < target < 1 || throw(ArgumentError("target_censoring_rate must be in (0, 1)."))
    finite_times = [Float64(t) for t in event_time if isfinite(Float64(t)) && Float64(t) > 0]
    isempty(finite_times) && throw(ArgumentError("event_time must contain positive finite values."))

    lo = eps(Float64)
    hi = maximum(finite_times)
    while uniform_censoring_rate(finite_times, hi) > target
        hi *= 2
    end

    mid = hi
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        rate = uniform_censoring_rate(finite_times, mid)
        abs(rate - target) <= tol && return mid
        if rate > target
            lo = mid
        else
            hi = mid
        end
    end
    return mid
end

function generate_simulation_data(
    n::Integer,
    p::Integer,
    nonzero::Integer;
    baseline::Symbol = :weibull,
    seed::Integer = 1,
    baseline_rate = nothing,
    baseline_scale = nothing,
    baseline_shape = nothing,
    rng::AbstractRNG = MersenneTwister(seed),
    scenario = :baseline,
    weak_multiplier = 0.35,
    null_correlation = 0.6,
    depth_median = 5000.0,
    depth_log_sd = 0.5,
)
    scenario = resolve_simulation_scenario(scenario)
    params = baseline_parameters(baseline; rate = baseline_rate, scale = baseline_scale, shape = baseline_shape)
    upper = baseline_censoring_upper(baseline)
    process = simulation_event_process(
        rng,
        n,
        p,
        nonzero;
        baseline = baseline,
        baseline_rate = params.rate,
        baseline_shape = params.shape,
        scenario = scenario, weak_multiplier = weak_multiplier, null_correlation = null_correlation,
    )
    censor_time = random_censoring_times(rng, n; upper = upper)
    observed = apply_censoring(process.event_time, censor_time)
    measurement = scenario == :counts ? sequencing_observations(rng, process.X; depth_median, depth_log_sd) :
        (X = process.X, sequencing_counts = nothing, sequencing_depths = nothing, observed_zero_fraction = 0.0)
    return (
        X = measurement.X,
        latent_X = process.X,
        sequencing_counts = measurement.sequencing_counts,
        sequencing_depths = measurement.sequencing_depths,
        observed_zero_fraction = measurement.observed_zero_fraction,
        scenario = scenario,
        paired_nulls = process.paired_nulls,
        latent_covariance = process.latent_covariance,
        time = observed.time,
        event = observed.event,
        beta = process.beta,
        nonzero_index = process.nonzero_index,
        eta = process.eta,
        event_time = process.event_time,
        censor_time = observed.censor_time,
        baseline_rate = params.rate,
        baseline_scale = inv(params.rate),
        baseline_shape = params.shape,
        censoring_distribution = :uniform,
        censoring_upper = upper,
    )
end

"""Independent n=1000 test cohort, shared by methods within a setting/replicate."""
function generate_simulation_test_data(n::Integer, p::Integer, nonzero::Integer;
    seed::Integer = 1, baseline::Symbol = :weibull, kwargs...)
    baseline_id = findfirst(==(baseline), (:weibull, :exponential, :gompertz))
    baseline_id === nothing && throw(ArgumentError("unsupported baseline $baseline"))
    # A separate RNG namespace avoids overlap with any scalar training seed.
    rng = MersenneTwister(UInt32[0x54455354, mod(seed, 2^32), n, p, nonzero, baseline_id])
    sim = generate_simulation_data(1000, p, nonzero; baseline, rng, kwargs...)
    suffix = sim.scenario == :baseline ? "" : ":$(sim.scenario)"
    return merge(sim, (test_seed = "test-v1:$seed:$n:$p:$nonzero:$baseline$suffix",))
end

function add_simulation_test_metrics!(metrics, summary, train, test;
    tie_method = :efron)
    # Saved summaries are already on the unstandardized coefficient scale.
    # Removing training centering only adds a constant to all risk scores.
    data = prepare_cox_data(train.X, train.time, train.event;
        transform = :log, standardize = false, tie_method)
    z_test = _validation_design_matrix(train.X, test.X, data)
    metrics.Cindex_train = copy(metrics.Cindex)
    metrics.Cindex_test = [concordance_index(test.time, test.event,
        z_test * Float64.(summary[!, est == "MAP" ? :map : Symbol(est)]))
        for est in string.(metrics.estimate)]
    metrics.n_test = fill(length(test.time), nrow(metrics))
    metrics.test_seed = fill(test.test_seed, nrow(metrics))
    return metrics
end

function prior_from_name(
    name::AbstractString,
    X::AbstractMatrix,
    time::AbstractVector,
    event::AbstractVector;
    p0 = nothing,
    p0_alpha::Real = 0.05,
    realdata::Bool = false,
    rhs_slab_df::Real = 4.0,
    rhs_slab_s2::Real = 2.0,
    hs_tau_scale::Real = 1.0,
    hs_local_scale::Real = 1.0,
    rhs_tau_scale::Real = 1.0,
    rhs_local_scale::Real = 1.0,
    rhs_pseudo_sigma_df::Real = 3.0,
    rhs_pseudo_sigma_scale::Real = 10.0,
    lasso_lambda = nothing,
    lasso_lambda2_shape::Real = 1.0,
    lasso_lambda2_rate::Real = 0.1,
    sslasso_spike_scale = nothing,
    sslasso_slab_scale = nothing,
    sslasso_theta_a::Real = 1.0,
    sslasso_theta_b::Real = 1.0,
)
    return cox_paper_prior(
        name;
        p0 = p0,
        X = X,
        time = time,
        event = event,
        p0_alpha = p0_alpha,
        realdata = realdata,
        rhs_slab_df = rhs_slab_df,
        rhs_slab_s2 = rhs_slab_s2,
        hs_tau_scale = hs_tau_scale,
        hs_local_scale = hs_local_scale,
        rhs_tau_scale = rhs_tau_scale,
        rhs_local_scale = rhs_local_scale,
        rhs_pseudo_sigma_df = rhs_pseudo_sigma_df,
        rhs_pseudo_sigma_scale = rhs_pseudo_sigma_scale,
        lasso_lambda = lasso_lambda,
        lasso_lambda2_shape = lasso_lambda2_shape,
        lasso_lambda2_rate = lasso_lambda2_rate,
        sslasso_spike_scale = sslasso_spike_scale,
        sslasso_slab_scale = sslasso_slab_scale,
        sslasso_theta_a = sslasso_theta_a,
        sslasso_theta_b = sslasso_theta_b,
    )
end

prior_p0(prior) = hasproperty(prior, :p0) ? getproperty(prior, :p0) : missing

function univariate_screen_table(X, time, event; names = nothing, alpha::Real = 0.05)
    screen = estimate_p0_univariate_cox(X, time, event; alpha = alpha)
    p = length(screen.pvalue)
    vars = names === nothing ? ["x$j" for j in 1:p] : string.(names)
    return DataFrame(
        variable = vars,
        estimate = screen.estimate,
        stderr = screen.stderr,
        zscore = screen.zscore,
        pvalue = screen.pvalue,
        significant = screen.significant,
        converged = screen.converged,
    ), screen
end

function posterior_summary(
    draws::AbstractMatrix;
    names = ["beta[$j]" for j in 1:size(draws, 2)],
    alpha::Real = 0.05,
    map = nothing,
)
    p = size(draws, 2)
    lower_p = alpha / 2
    upper_p = 1 - alpha / 2
    beta_map = map === nothing ? fill(NaN, p) : collect(Float64, map)
    length(beta_map) == p || throw(DimensionMismatch("MAP estimate length must match beta dimension."))
    return DataFrame(
        variable = names,
        mean = vec(mean(draws; dims = 1)),
        median = [median(view(draws, :, j)) for j in 1:p],
        map = beta_map,
        lower = [quantile(view(draws, :, j), lower_p) for j in 1:p],
        upper = [quantile(view(draws, :, j), upper_p) for j in 1:p],
    )
end

function beta_draws_unstandardized(data::CoxPreparedData, draws::AbstractMatrix)
    size(draws, 2) == length(data.col_scale) ||
        throw(DimensionMismatch("draw columns must match the number of covariates."))
    data.standardize || return Matrix{Float64}(draws)
    return Matrix{Float64}(draws) ./ reshape(data.col_scale, 1, :)
end

function beta_vector_unstandardized(data::CoxPreparedData, beta::AbstractVector)
    length(beta) == length(data.col_scale) ||
        throw(DimensionMismatch("beta length must match the number of covariates."))
    data.standardize || return collect(Float64, beta)
    return collect(Float64, beta) ./ data.col_scale
end

function false_discovery(beta::AbstractVector, summary::DataFrame)
    selected = (summary.lower .* summary.upper) .> 0
    truth = beta .!= 0
    fp = sum((.!truth) .& selected)
    tp = sum(truth .& selected)
    fn = sum(truth .& (.!selected))
    tn = sum((.!truth) .& (.!selected))
    fdr = fp + tp == 0 ? 0.0 : fp / (fp + tp)
    return (FP = fp, FN = fn, TP = tp, TN = tn, FDR = fdr)
end

function logmeanexp(v::AbstractVector)
    m = maximum(v)
    return m + log(mean(exp.(v .- m)))
end

function posterior_event_loglik(data::CoxPreparedData, draws::AbstractMatrix)
    n = length(data.event)
    s = size(draws, 1)
    out = Matrix{Float64}(undef, n, s)
    for r in 1:s
        out[:, r] .= cox_loglik_events(data, view(draws, r, :))
    end
    return out
end

function posterior_draw_layout(result, n_total::Integer)
    if result.samples isa MHSamples
        return (chains = 1, draws = n_total)
    elseif result.samples isa MultiMHSamples
        n_chain = length(result.samples.chains)
        n_iter = size(result.samples.chains[1].draws, 1)
        return (chains = n_chain, draws = n_iter)
    end
    try
        value = getfield(result.samples, :value)
        n_iter, _, n_chain = size(value)
        if n_iter * n_chain == n_total
            return (chains = n_chain, draws = n_iter)
        end
    catch
    end
    return (chains = 1, draws = n_total)
end

function _loo_loglik_array(ll_event::AbstractMatrix, n_chains::Integer, n_draws::Integer)
    n_events, n_total = size(ll_event)
    n_chains * n_draws == n_total ||
        throw(DimensionMismatch("LOO chain/draw layout does not match log-likelihood samples."))
    out = Array{Float64}(undef, n_draws, n_chains, n_events)
    @inbounds for c in 1:n_chains
        offset = (c - 1) * n_draws
        for d in 1:n_draws
            out[d, c, :] .= view(ll_event, :, offset + d)
        end
    end
    return out
end

function psis_loo_metrics(ll_event::AbstractMatrix, n_chains::Integer, n_draws::Integer)
    if n_chains * n_draws < 26 || isempty(ll_event)
        return (
            ELPD_LOO = NaN,
            ELPD_LOO_SE = NaN,
            PLOO = NaN,
            PLOO_SE = NaN,
            LOOIC = NaN,
            LOOIC_SE = NaN,
            ParetoK_max = NaN,
            ParetoK_bad = missing,
            PSIS_LOO = NaN,
        )
    end
    try
        loo_result = PosteriorStats.loo(_loo_loglik_array(ll_event, n_chains, n_draws))
        estimates = loo_result.estimates
        pareto_k = vec(collect(loo_result.pointwise.pareto_shape))
        looic = -2 * estimates.elpd
        return (
            ELPD_LOO = estimates.elpd,
            ELPD_LOO_SE = estimates.se_elpd,
            PLOO = estimates.p,
            PLOO_SE = estimates.se_p,
            LOOIC = looic,
            LOOIC_SE = 2 * estimates.se_elpd,
            ParetoK_max = maximum(pareto_k),
            ParetoK_bad = count(>(0.7), pareto_k),
            PSIS_LOO = looic,
        )
    catch err
        @warn "PSIS-LOO calculation failed; LOO metrics will be NaN." exception=(err, catch_backtrace())
        return (
            ELPD_LOO = NaN,
            ELPD_LOO_SE = NaN,
            PLOO = NaN,
            PLOO_SE = NaN,
            LOOIC = NaN,
            LOOIC_SE = NaN,
            ParetoK_max = NaN,
            ParetoK_bad = missing,
            PSIS_LOO = NaN,
        )
    end
end

function fit_metrics(
    data::CoxPreparedData,
    draws::AbstractMatrix,
    beta_hat::AbstractVector;
    chains::Integer = 1,
    draws_per_chain::Integer = size(draws, 1),
    ll_event = nothing,
    loo = nothing,
)
    point_loglik = cox_logpartiallikelihood(data, beta_hat)
    point_risk = data.z * beta_hat
    event_loglik =
        if ll_event === nothing
            ll = posterior_event_loglik(data, draws)
            event_rows = findall(!=(0), data.event)
            ll[event_rows, :]
        else
            ll_event
        end
    loo_metrics = loo === nothing ? psis_loo_metrics(event_loglik, chains, draws_per_chain) : loo
    alog_plik = sum(logmeanexp(view(event_loglik, i, :)) for i in axes(event_loglik, 1))
    mean_ll = vec(mean(event_loglik; dims = 2))
    pw1 = 2 * sum([logmeanexp(view(event_loglik, i, :)) for i in axes(event_loglik, 1)] .- mean_ll)
    pw2 = size(event_loglik, 2) > 1 ? sum(var(view(event_loglik, i, :)) for i in axes(event_loglik, 1)) : NaN
    # Historical LPML column is negative LPML; keep its sign for compatibility.
    lpml = sum(logmeanexp(-view(event_loglik, i, :)) for i in axes(event_loglik, 1))
    sample_total = vec(sum(event_loglik; dims = 1))
    pD = mean(-2 .* sample_total .+ 2) - (-2 * point_loglik + 2)
    return (
        loglik = point_loglik,
        LPML = lpml,
        AIC = -2 * point_loglik + 2 * length(beta_hat),
        BIC = -2 * point_loglik + log(length(data.time)) * length(beta_hat),
        WAIC_pw1 = -2 * alog_plik + 2 * pw1,
        WAIC_pw2 = -2 * alog_plik + 2 * pw2,
        DIC = point_loglik - pD,
        ELPD_LOO = loo_metrics.ELPD_LOO,
        ELPD_LOO_SE = loo_metrics.ELPD_LOO_SE,
        PLOO = loo_metrics.PLOO,
        PLOO_SE = loo_metrics.PLOO_SE,
        LOOIC = loo_metrics.LOOIC,
        LOOIC_SE = loo_metrics.LOOIC_SE,
        ParetoK_max = loo_metrics.ParetoK_max,
        ParetoK_bad = loo_metrics.ParetoK_bad,
        PSIS_LOO = loo_metrics.PSIS_LOO,
        Cindex = concordance_index(data.time, data.event, point_risk),
    )
end

function add_elpd_diffs!(metrics::DataFrame; group_cols = Symbol[])
    :ELPD_LOO in propertynames(metrics) || return metrics
    :ELPD_Diff in propertynames(metrics) || (metrics.ELPD_Diff = fill(NaN, nrow(metrics)))
    valid_group_cols = [c for c in group_cols if c in propertynames(metrics)]
    groups = isempty(valid_group_cols) ? [metrics] : groupby(metrics, valid_group_cols)
    for group in groups
        vals = collect(skipmissing(group.ELPD_LOO))
        vals = vals[isfinite.(vals)]
        isempty(vals) && continue
        best = maximum(vals)
        rows = group isa SubDataFrame ? parentindices(group)[1] : axes(group, 1)
        metrics[rows, :ELPD_Diff] .= group.ELPD_LOO .- best
    end
    return metrics
end

function fit_and_evaluate(
    X::AbstractMatrix,
    time::AbstractVector,
    event::AbstractVector;
    prior_name::AbstractString = "cox-rhs",
    sampler::Symbol = :nuts,
    nuts_backend::Symbol = :turing,
    n_samples::Integer = 1000,
    warmup::Integer = 500,
    thin::Integer = 1,
    chains::Integer = 4,
    multichains::Bool = false,
    target_accept::Real = 0.8,
    max_depth::Integer = 10,
    hmc_n_leapfrog::Integer = 10,
    hmc_step_size = nothing,
    noncentered::Bool = false,
    seed::Integer = 1,
    use_car::Bool = false,
    car_weights = nothing,
    car_distance = :bray,
    car_weight_quantile::Real = 0.75,
    car_precision::Real = 1.0,
    sample_car_strength::Bool = false,
    car_strength_prior_a::Real = 1.0,
    car_strength_prior_b::Real = 1.0,
    car_degree_scale::Symbol = :mean_active,
    car_kappa_init::Real = 0.5,
    p0 = nothing,
    p0_alpha::Real = 0.05,
    constraint::Symbol = :soft_center,
    prox_sum0_lambda::Real = 1e-3,
    prox_l1_lambda::Real = 1e-3,
    prox_root_tol::Real = 1e-10,
    prox_root_maxiter::Integer = 100,
    mala_step_size::Real = 1e-4,
    mala_target_accept::Real = 0.574,
    mala_adapt_every::Integer = 50,
    mala_min_step::Real = 1e-8,
    mala_max_step::Real = 1.0,
    mala_precondition::Bool = false,
    mala_precondition_mode::Symbol = :auto,
    mala_precondition_diag::Bool = false,
    mala_metric_ridge::Real = 1e-6,
    mala_metric_max_cholesky_tries::Integer = 8,
    mala_stepsize_schedule::Symbol = :adaptive_mcmc,
    mala_sgld_c0::Real = 1.0,
    mala_sgld_target_accept::Real = 0.5,
    mala_sgld_accept_low::Real = 0.40,
    mala_sgld_accept_high::Real = 0.60,
    mala_sgld_adapt_c0::Bool = true,
    mala_sgld_adapt_every::Integer = mala_adapt_every,
    mala_sgld_c0_min::Real = 1e-12,
    mala_sgld_c0_max::Real = 1e12,
    prox_block_step_size = nothing,
    prox_block_min_step_size::Real = 1e-8,
    prox_block_max_step_size::Real = 1.0,
    prox_block_adapt_every::Integer = 50,
    prox_block_divergence_threshold::Real = 1000.0,
    prox_block_metric::Symbol = :unit_diag,
    realdata::Bool = false,
    clinical_X = nothing,
    clinical_names = clinical_X === nothing ? String[] : ["clinical[$j]" for j in axes(clinical_X, 2)],
    clinical_prior_df::Real = 3.0,
    clinical_prior_scale::Real = 2.5,
    standardize::Bool = false,
    tie_method::Symbol = :efron,
    rhs_slab_df::Real = 4.0,
    rhs_slab_s2::Real = 2.0,
    hs_tau_scale::Real = 1.0,
    hs_local_scale::Real = 1.0,
    rhs_tau_scale::Real = 1.0,
    rhs_local_scale::Real = 1.0,
    rhs_pseudo_sigma_df::Real = 3.0,
    rhs_pseudo_sigma_scale::Real = 10.0,
    lasso_lambda = nothing,
    lasso_lambda2_shape::Real = 1.0,
    lasso_lambda2_rate::Real = 0.1,
    sslasso_spike_scale = nothing,
    sslasso_slab_scale = nothing,
    sslasso_theta_a::Real = 1.0,
    sslasso_theta_b::Real = 1.0,
    alpha_prior_shape = nothing,
    alpha_prior_scale = nothing,
    alpha_prior_kind = :invgamma,
    init::Symbol = :zero,
    progress::Bool = true,
    approx::Bool = false,
    approx_point::Symbol = :zero,
    compute_metrics::Bool = true,
)
    if realdata && prior_name in ("prox-lasso", "prox-sslasso") && alpha_prior_scale === nothing
        alpha_prior_scale = Float64(size(X, 2) + (clinical_X === nothing ? 0 : size(clinical_X, 2)))
    end
    prior = prior_from_name(
        prior_name,
        X,
        time,
        event;
        p0 = p0,
        p0_alpha = p0_alpha,
        realdata = realdata,
        rhs_slab_df = rhs_slab_df,
        rhs_slab_s2 = rhs_slab_s2,
        hs_tau_scale = hs_tau_scale,
        hs_local_scale = hs_local_scale,
        rhs_tau_scale = rhs_tau_scale,
        rhs_local_scale = rhs_local_scale,
        rhs_pseudo_sigma_df = rhs_pseudo_sigma_df,
        rhs_pseudo_sigma_scale = rhs_pseudo_sigma_scale,
        lasso_lambda = lasso_lambda,
        lasso_lambda2_shape = lasso_lambda2_shape,
        lasso_lambda2_rate = lasso_lambda2_rate,
        sslasso_spike_scale = sslasso_spike_scale,
        sslasso_slab_scale = sslasso_slab_scale,
        sslasso_theta_a = sslasso_theta_a,
        sslasso_theta_b = sslasso_theta_b,
    )
    explicit_nuts_kwargs =
        sampler == :nuts_explicit || (sampler == :nuts && nuts_backend == :explicit) ? (
            max_depth = max_depth,
            prox_block_step_size = prox_block_step_size,
            prox_block_min_step_size = prox_block_min_step_size,
            prox_block_max_step_size = prox_block_max_step_size,
            prox_block_adapt_every = prox_block_adapt_every,
            prox_block_divergence_threshold = prox_block_divergence_threshold,
            prox_block_metric = prox_block_metric,
        ) : NamedTuple()
    result =
        if clinical_X !== nothing
            sampler == :hmc || throw(ArgumentError("joint clinical fitting currently requires HMC"))
            approx && throw(ArgumentError("joint clinical fitting requires --approx false"))
            sample_clinical_cox(X, time, event, clinical_X;
                prior, clinical_names, clinical_prior_df, clinical_prior_scale,
                n_samples, warmup, thin, chains, multichains, seed, target_accept,
                hmc_n_leapfrog, hmc_step_size, noncentered, constraint, prox_sum0_lambda,
                standardize, tie_method, use_car,
                car_weights, car_distance, car_weight_quantile, car_precision, sample_car_strength,
                car_strength_prior_a, car_strength_prior_b, car_degree_scale, car_kappa_init,
                prox_l1_lambda, alpha_prior_shape, alpha_prior_scale, alpha_prior_kind,
                prox_root_tol, prox_root_maxiter, prox_block_step_size, prox_block_min_step_size,
                prox_block_max_step_size, prox_block_adapt_every, prox_block_divergence_threshold,
                prox_block_metric, init, progress)
        elseif sampler in (:nuts, :nuts_turing, :nuts_explicit)
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                nuts_backend = nuts_backend,
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                noncentered = noncentered,
                seed = seed,
                use_car = use_car,
                car_weights = car_weights,
                car_distance = car_weights === nothing ? car_distance : nothing,
                car_weight_quantile = car_weight_quantile,
                car_precision = car_precision,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_degree_scale = car_degree_scale,
                car_kappa_init = car_kappa_init,
                standardize = standardize,
                tie_method = tie_method,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_scale = alpha_prior_scale,
                alpha_prior_kind = alpha_prior_kind,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                init = init,
                progress = progress,
                approx = approx,
                approx_point = approx_point,
                explicit_nuts_kwargs...,
            )
        elseif sampler == :hmc
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                hmc_n_leapfrog = hmc_n_leapfrog,
                hmc_step_size = hmc_step_size,
                noncentered = noncentered,
                seed = seed,
                use_car = use_car,
                car_weights = car_weights,
                car_distance = car_weights === nothing ? car_distance : nothing,
                car_weight_quantile = car_weight_quantile,
                car_precision = car_precision,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_degree_scale = car_degree_scale,
                car_kappa_init = car_kappa_init,
                standardize = standardize,
                tie_method = tie_method,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_scale = alpha_prior_scale,
                alpha_prior_kind = alpha_prior_kind,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                max_depth = max_depth,
                prox_block_step_size = prox_block_step_size,
                prox_block_min_step_size = prox_block_min_step_size,
                prox_block_max_step_size = prox_block_max_step_size,
                prox_block_adapt_every = prox_block_adapt_every,
                prox_block_divergence_threshold = prox_block_divergence_threshold,
                prox_block_metric = prox_block_metric,
                init = init,
                progress = progress,
                approx = approx,
                approx_point = approx_point,
            )
        elseif sampler == :mh || sampler == :mala || sampler == :ula
            if sampler == :mala || sampler == :ula
                sample_bayes_cox(
                    X,
                    time,
                    event;
                    prior = prior,
                    sampler = sampler,
                    n_samples = n_samples,
                    warmup = warmup,
                    thin = thin,
                    chains = chains,
                    multichains = multichains,
                    target_accept = target_accept,
                    seed = seed,
                    use_car = use_car,
                    car_weights = car_weights,
                    car_distance = car_weights === nothing ? car_distance : nothing,
                    car_weight_quantile = car_weight_quantile,
                    car_precision = car_precision,
                    sample_car_strength = sample_car_strength,
                    car_strength_prior_a = car_strength_prior_a,
                    car_strength_prior_b = car_strength_prior_b,
                    car_degree_scale = car_degree_scale,
                    car_kappa_init = car_kappa_init,
                    standardize = standardize,
                    tie_method = tie_method,
                    constraint = constraint,
                    prox_sum0_lambda = prox_sum0_lambda,
                    prox_l1_lambda = prox_l1_lambda,
                    alpha_prior_shape = alpha_prior_shape,
                    alpha_prior_scale = alpha_prior_scale,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                    mala_step_size = mala_step_size,
                    mala_target_accept = mala_target_accept,
                    mala_adapt_every = mala_adapt_every,
                    mala_min_step = mala_min_step,
                    mala_max_step = mala_max_step,
                    mala_precondition = mala_precondition,
                    mala_precondition_mode = mala_precondition_mode,
                    mala_precondition_diag = mala_precondition_diag,
                    mala_metric_ridge = mala_metric_ridge,
                    mala_metric_max_cholesky_tries = mala_metric_max_cholesky_tries,
                    mala_stepsize_schedule = mala_stepsize_schedule,
                    mala_sgld_c0 = mala_sgld_c0,
                    mala_sgld_target_accept = mala_sgld_target_accept,
                    mala_sgld_accept_low = mala_sgld_accept_low,
                    mala_sgld_accept_high = mala_sgld_accept_high,
                    mala_sgld_adapt_c0 = mala_sgld_adapt_c0,
                    mala_sgld_adapt_every = mala_sgld_adapt_every,
                    mala_sgld_c0_min = mala_sgld_c0_min,
                    mala_sgld_c0_max = mala_sgld_c0_max,
                    init = init,
                    progress = progress,
                    approx = approx,
                    approx_point = approx_point,
                )
            else
                sample_bayes_cox(
                    X,
                    time,
                    event;
                    prior = prior,
                    sampler = sampler,
                    n_samples = n_samples,
                    warmup = warmup,
                    thin = thin,
                    chains = chains,
                    multichains = multichains,
                    target_accept = target_accept,
                    seed = seed,
                    use_car = use_car,
                    car_weights = car_weights,
                    car_distance = car_weights === nothing ? car_distance : nothing,
                    car_weight_quantile = car_weight_quantile,
                    car_precision = car_precision,
                    sample_car_strength = sample_car_strength,
                    car_strength_prior_a = car_strength_prior_a,
                    car_strength_prior_b = car_strength_prior_b,
                    car_degree_scale = car_degree_scale,
                    car_kappa_init = car_kappa_init,
                    standardize = standardize,
                    tie_method = tie_method,
                    constraint = constraint,
                    prox_sum0_lambda = prox_sum0_lambda,
                    init = init,
                    progress = progress,
                    approx = approx,
                    approx_point = approx_point,
                )
            end
        elseif sampler == :gibbs
            sample_bayes_cox(
                X,
                time,
                event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                thin = thin,
                chains = chains,
                multichains = multichains,
                target_accept = target_accept,
                seed = seed,
                use_car = use_car,
                car_weights = car_weights,
                car_distance = car_weights === nothing ? car_distance : nothing,
                car_weight_quantile = car_weight_quantile,
                car_precision = car_precision,
                sample_car_strength = sample_car_strength,
                car_strength_prior_a = car_strength_prior_a,
                car_strength_prior_b = car_strength_prior_b,
                car_degree_scale = car_degree_scale,
                car_kappa_init = car_kappa_init,
                standardize = standardize,
                tie_method = tie_method,
                constraint = constraint,
                prox_sum0_lambda = prox_sum0_lambda,
                prox_l1_lambda = prox_l1_lambda,
                alpha_prior_shape = alpha_prior_shape,
                alpha_prior_scale = alpha_prior_scale,
                alpha_prior_kind = alpha_prior_kind,
                prox_root_tol = prox_root_tol,
                prox_root_maxiter = prox_root_maxiter,
                init = init,
                progress = progress,
                approx = approx,
                approx_point = approx_point,
            )
        else
            throw(ArgumentError("sampler must be :nuts, :nuts_turing, :nuts_explicit, :hmc, :mh, :mala, :ula, or :gibbs."))
    end
    model_draws = posterior_beta_draws(result)
    if !compute_metrics
        return (result = result, draws = beta_draws_unstandardized(result.data, model_draws),
                summary = nothing, metrics = nothing, p0 = prior_p0(prior))
    end
    layout = posterior_draw_layout(result, size(model_draws, 1))
    eval_data = clinical_X === nothing ? result.data : BayesCox4Micro.joint_cox_data(result.data, clinical_X)
    eval_draws = clinical_X === nothing ? model_draws : hcat(model_draws, posterior_clinical_draws(result))
    loglik = posterior_event_loglik(eval_data, eval_draws)
    event_rows = findall(!=(0), result.data.event)
    ll_event = loglik[event_rows, :]
    loo = psis_loo_metrics(ll_event, layout.chains, layout.draws)
    names = ["beta[$j]" for j in 1:size(model_draws, 2)]
    beta_map_model = posterior_beta_map(result)
    draws = beta_draws_unstandardized(result.data, model_draws)
    beta_map = beta_vector_unstandardized(result.data, beta_map_model)
    summary = posterior_summary(draws; names = names, map = beta_map)
    beta_mean_model = vec(mean(eval_draws; dims = 1))
    beta_median_model = [median(view(eval_draws, :, j)) for j in 1:size(eval_draws, 2)]
    eval_map = vec(eval_draws[argmax(posterior_logposterior_draws(result)), :])
    metrics_mean = fit_metrics(eval_data, eval_draws, beta_mean_model; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_median = fit_metrics(eval_data, eval_draws, beta_median_model; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_map = fit_metrics(eval_data, eval_draws, eval_map; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    diagnostics = sampler_diagnostics(result)
    metrics = DataFrame(
        estimate = ["mean", "median", "MAP"],
        loglik = [metrics_mean.loglik, metrics_median.loglik, metrics_map.loglik],
        LPML = [metrics_mean.LPML, metrics_median.LPML, metrics_map.LPML],
        AIC = [metrics_mean.AIC, metrics_median.AIC, metrics_map.AIC],
        BIC = [metrics_mean.BIC, metrics_median.BIC, metrics_map.BIC],
        WAIC_pw1 = [metrics_mean.WAIC_pw1, metrics_median.WAIC_pw1, metrics_map.WAIC_pw1],
        WAIC_pw2 = [metrics_mean.WAIC_pw2, metrics_median.WAIC_pw2, metrics_map.WAIC_pw2],
        DIC = [metrics_mean.DIC, metrics_median.DIC, metrics_map.DIC],
        ELPD_LOO = [metrics_mean.ELPD_LOO, metrics_median.ELPD_LOO, metrics_map.ELPD_LOO],
        ELPD_LOO_SE = [metrics_mean.ELPD_LOO_SE, metrics_median.ELPD_LOO_SE, metrics_map.ELPD_LOO_SE],
        PLOO = [metrics_mean.PLOO, metrics_median.PLOO, metrics_map.PLOO],
        PLOO_SE = [metrics_mean.PLOO_SE, metrics_median.PLOO_SE, metrics_map.PLOO_SE],
        LOOIC = [metrics_mean.LOOIC, metrics_median.LOOIC, metrics_map.LOOIC],
        LOOIC_SE = [metrics_mean.LOOIC_SE, metrics_median.LOOIC_SE, metrics_map.LOOIC_SE],
        ParetoK_max = [metrics_mean.ParetoK_max, metrics_median.ParetoK_max, metrics_map.ParetoK_max],
        ParetoK_bad = [metrics_mean.ParetoK_bad, metrics_median.ParetoK_bad, metrics_map.ParetoK_bad],
        PSIS_LOO = [metrics_mean.PSIS_LOO, metrics_median.PSIS_LOO, metrics_map.PSIS_LOO],
        Cindex = [metrics_mean.Cindex, metrics_median.Cindex, metrics_map.Cindex],
        warmup_seconds = fill(diagnostics.warmup_seconds, 3),
        sample_seconds = fill(diagnostics.sample_seconds, 3),
        total_seconds = fill(diagnostics.total_seconds, 3),
        allocated_mb = fill(diagnostics.allocated_bytes / 2.0^20, 3),
        gc_seconds = fill(diagnostics.gc_seconds, 3),
        gc_percent = fill(diagnostics.gc_percent, 3),
    )
    return (result = result, draws = draws, summary = summary, metrics = metrics, p0 = prior_p0(prior))
end

function _plot_path_with_ext(path::AbstractString, ext::AbstractString)
    root, _ = splitext(path)
    return root * ext
end

function _save_plots_vector(plot_obj, path::AbstractString; save_eps::Bool = true)
    pdf_path = _plot_path_with_ext(path, ".pdf")
    ensure_dir(dirname(pdf_path))
    Plots.savefig(plot_obj, pdf_path)
    if save_eps
        eps_path = _plot_path_with_ext(path, ".eps")
        open(eps_path, "w") do io
            Base.invokelatest(show, io, MIME("application/postscript"), plot_obj)
        end
    end
    return pdf_path
end

function _coefficient_diagnostic_title(summary::DataFrame, estimate::Symbol, estimates::AbstractVector, truth)
    truth === nothing && return ""
    truthv = Float64.(truth)
    length(truthv) == length(estimates) ||
        throw(DimensionMismatch("truth length must match summary row count."))
    mse = mean((Float64.(estimates) .- truthv) .^ 2)
    fd = false_discovery(truthv, summary)
    estimate_name = estimate == :map ? "MAP" : String(estimate)
    return @sprintf(
        "MSE(%s)=%.4f | FDR=%.4f (FP=%d, FN=%d, TP=%d, TN=%d)",
        estimate_name,
        mse,
        fd.FDR,
        fd.FP,
        fd.FN,
        fd.TP,
        fd.TN,
    )
end

function coefficient_plot(
    summary::DataFrame,
    path::AbstractString;
    truth = nothing,
    title = "Coefficient estimates",
    estimate::Symbol = :mean,
)
    x = 1:nrow(summary)
    estimate in propertynames(summary) ||
        throw(ArgumentError("summary does not contain estimate column $estimate."))
    (:lower in propertynames(summary) && :upper in propertynames(summary)) ||
        throw(ArgumentError("summary must contain lower and upper columns."))
    y = Float64.(summary[!, estimate])
    lower = Float64.(summary.lower)
    upper = Float64.(summary.upper)
    estimate_label = estimate == :map ? "MAP" : "posterior $(String(estimate))"
    diagnostic_title = _coefficient_diagnostic_title(summary, estimate, y, truth)
    plot_title = isempty(diagnostic_title) ? title : "$title\n$diagnostic_title"
    y_values = vcat(lower, upper, y, [0.0], truth === nothing ? Float64[] : Float64.(truth))
    ymin, ymax = extrema(y_values)
    pad = max(0.12 * (ymax - ymin), 0.15)
    ymin -= pad
    ymax += pad
    try
        Plots.gr()
    catch
    end
    p = Plots.scatter(
        x,
        y;
        yerror = (y .- lower, upper .- y),
        label = estimate_label,
        xlabel = "Taxon index",
        ylabel = "Coefficient",
        title = plot_title,
        ylims = (ymin, ymax),
        color = :steelblue,
        markerstrokecolor = :steelblue,
        markersize = 3,
        linewidth = 1.2,
        legend = :topright,
        grid = true,
        size = (1000, 520),
        titlefontsize = 10,
        guidefontsize = 10,
        tickfontsize = 8,
        legendfontsize = 8,
        left_margin = 9Plots.mm,
        right_margin = 7Plots.mm,
        bottom_margin = 9Plots.mm,
        top_margin = 9Plots.mm,
    )
    Plots.hline!(p, [0.0]; label = "", color = :gray, linestyle = :dash, linewidth = 1)
    if truth !== nothing
        truthv = Float64.(truth)
        length(truthv) == nrow(summary) ||
            throw(DimensionMismatch("truth length must match summary row count."))
        Plots.scatter!(
            p,
            x,
            truthv;
            label = "true",
            color = :firebrick,
            markerstrokecolor = :firebrick,
            markersize = 3,
        )
    end
    return _save_plots_vector(p, path)
end

function _finite_mean(v)
    vals = Float64[]
    for x in skipmissing(v)
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    return isempty(vals) ? NaN : mean(vals)
end

function _metric_values(rows::DataFrame, metric::Symbol)
    agg = combine(groupby(rows, [:prior, :estimate]), metric => _finite_mean => :value)
    priors = sort(unique(agg.prior))
    estimate_order = ["mean", "median", "MAP"]
    estimates = sort(unique(agg.estimate); by = e -> begin
        pos = findfirst(==(string(e)), estimate_order)
        pos === nothing ? length(estimate_order) + 1 : pos
    end)
    return agg, priors, estimates
end

function _metric_panel_plot(rows::DataFrame, metric::Symbol; title::AbstractString)
    agg, priors, estimates = _metric_values(rows, metric)
    p = Plots.plot(;
        title = title,
        xlabel = "",
        ylabel = "",
        legend = false,
        grid = true,
        gridalpha = 0.22,
        framestyle = :box,
        titlefontsize = 10,
        tickfontsize = 7,
        guidefontsize = 8,
        left_margin = 6Plots.mm,
        right_margin = 3Plots.mm,
        bottom_margin = 7Plots.mm,
        top_margin = 5Plots.mm,
    )
    isempty(priors) && return p
    vals = [Float64(v) for v in agg.value if isfinite(Float64(v))]
    isempty(vals) && return p
    xbase = collect(1:length(priors))
    n_est = max(length(estimates), 1)
    bar_w = min(0.72 / n_est, 0.24)
    offsets = collect(range(-bar_w * (n_est - 1) / 2, bar_w * (n_est - 1) / 2; length = n_est))
    palette = ["#2F5D8C", "#5B8FC1", "#A9CBE8", "#6BAF92", "#B8D8BA"]
    all_label_y = Float64[]
    for (k, est) in enumerate(estimates)
        xs = Float64[]
        ys = Float64[]
        for (i, prior) in enumerate(priors)
            idx = findfirst((agg.prior .== prior) .& (agg.estimate .== est))
            idx === nothing && continue
            value = agg.value[idx]
            isfinite(value) || continue
            push!(xs, xbase[i] + offsets[k])
            push!(ys, Float64(value))
        end
        isempty(xs) && continue
        Plots.bar!(
            p,
            xs,
            ys;
            bar_width = bar_w,
            label = string(est),
            color = palette[mod1(k, length(palette))],
            linecolor = :white,
            linewidth = 0.6,
            fillalpha = 0.92,
        )
        span = max(maximum(vals) - minimum(vals), 1e-6)
        offset = max(0.025 * span, 1e-4)
        for (x, y) in zip(xs, ys)
            label_y = y >= 0 ? y + offset : y - offset
            push!(all_label_y, label_y)
            valign = y >= 0 ? :bottom : :top
            Plots.annotate!(p, x, label_y, Plots.text(@sprintf("%.2f", y), 6, :center, valign))
        end
    end
    y_candidates = vcat(vals, [0.0], all_label_y)
    is_cindex = startswith(string(metric), "Cindex")
    ymin = is_cindex ? 0.0 : minimum(y_candidates)
    ymax = is_cindex ? max(1.0, maximum(y_candidates)) : maximum(y_candidates)
    span = max(ymax - ymin, 1e-6)
    ymin -= is_cindex ? 0.0 : 0.12 * span
    ymax += 0.18 * span
    labels = [length(string(prior)) > 10 ? string(prior)[1:10] : string(prior) for prior in priors]
    Plots.plot!(
        p;
        xticks = (xbase, labels),
        xrotation = 35,
        ylims = (ymin, ymax),
        legend = :topright,
    )
    return p
end

function metrics_plot(metrics::DataFrame, path::AbstractString; title = "Model metrics")
    plot_rows = copy(metrics)
    plot_rows.prior = string.(plot_rows.prior)
    plot_rows.estimate = string.(plot_rows.estimate)
    for col in (:prior, :estimate, :WAIC_pw1, :LOOIC, :LPML, :Cindex)
        col in propertynames(plot_rows) ||
            throw(ArgumentError("metrics table does not contain required column $col."))
    end
    try
        Plots.gr()
    catch
    end
    panels = (
        _metric_panel_plot(plot_rows, :WAIC_pw1; title = "WAIC"),
        _metric_panel_plot(plot_rows, :LOOIC; title = "LOOIC"),
        _metric_panel_plot(plot_rows, :LPML; title = "LPML"),
        _metric_panel_plot(plot_rows, :Cindex; title = "Training C-index"),
    )
    has_test = :Cindex_test in propertynames(plot_rows)
    has_test && (panels = (panels...,
        _metric_panel_plot(plot_rows, :Cindex_test; title = "Test C-index (n=1000)")))
    p = Plots.plot(
        panels...;
        layout = has_test ? (3, 2) : (2, 2),
        plot_title = title,
        size = (1200, 850),
        plot_titlefontsize = 13,
    )
    return _save_plots_vector(p, path)
end

function write_table(path::AbstractString, table)
    ensure_dir(dirname(path))
    CSV.write(path, table)
    return path
end

file_complete(path::AbstractString) = isfile(path) && filesize(path) > 0

function simulation_samples_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "samples_wide.csv")) ||
           file_complete(joinpath(run_dir, "result_bundle.jld2"))
end

function simulation_metric_outputs_complete(run_dir::AbstractString)
    return file_complete(joinpath(run_dir, "posterior_summary.csv")) &&
           file_complete(joinpath(run_dir, "metrics.csv")) &&
           file_complete(joinpath(run_dir, "false_discovery.csv"))
end

simulation_run_complete(run_dir::AbstractString) =
    simulation_samples_complete(run_dir) && simulation_metric_outputs_complete(run_dir)

simulation_run_recoverable(run_dir::AbstractString) =
    simulation_samples_complete(run_dir) && file_complete(joinpath(run_dir, "posterior_summary.csv"))

function mark_simulation_run_complete(run_dir::AbstractString)
    ensure_dir(run_dir)
    open(joinpath(run_dir, "complete.txt"), "w") do io
        write(io, "complete\n")
    end
    return joinpath(run_dir, "complete.txt")
end

function read_run_metadata(run_dir::AbstractString)
    path = joinpath(run_dir, "metadata.json")
    file_complete(path) || return Dict{String,Any}()
    try
        # Metadata is small and may be updated on resume. Avoid a memory-mapped
        # file handle, which can prevent rewriting the file on Windows.
        return Dict{String,Any}(JSON.parse(read(path, String)))
    catch err
        @warn "Could not read metadata.json from $run_dir." exception=(err, catch_backtrace())
        return Dict{String,Any}()
    end
end

function _metadata_number(run_dir::AbstractString, key::AbstractString, default)
    meta = read_run_metadata(run_dir)
    haskey(meta, key) || return default
    value = meta[key]
    value isa Number && return value
    value isa AbstractString || return default
    try
        return parse(Float64, value)
    catch
        return default
    end
end

function _metadata_string(run_dir::AbstractString, key::AbstractString, default::AbstractString)
    meta = read_run_metadata(run_dir)
    haskey(meta, key) || return default
    value = meta[key]
    value === nothing && return default
    value === missing && return default
    return string(value)
end

# Generic metadata value comparison (moved from run_simulation.jl)
function metadata_value_matches(actual, expected)
    actual === missing && return expected === missing
    actual === nothing && return expected === nothing
    if expected isa Number
        actual_number =
            if actual isa Number
                Float64(actual)
            elseif actual isa AbstractString
                try
                    parse(Float64, actual)
                catch
                    return false
                end
            else
                return false
            end
        return isapprox(actual_number, Float64(expected); rtol = 1e-10, atol = 1e-10)
    end
    return string(actual) == string(expected)
end

# ===== Type-aware metadata comparison for resume logic =====

const COMMON_SAMPLER_METADATA_KEYS = [
    "sampler", "samples", "n_samples", "warmup", "chains"
]

const HMC_SAMPLER_METADATA_KEYS = [
    "nuts_backend", "target_accept", "max_depth",
    "hmc_n_leapfrog", "hmc_step_size", "noncentered"
]

const MALA_SAMPLER_METADATA_KEYS = [
    "mala_step_size", "mala_target_accept", "mala_adapt_every",
    "mala_min_step", "mala_max_step", "mala_precondition",
    "mala_precondition_mode", "mala_precondition_diag",
    "mala_metric_ridge", "mala_metric_max_cholesky_tries",
    "mala_stepsize_schedule", "mala_sgld_c0", "mala_sgld_target_accept",
    "mala_sgld_accept_low", "mala_sgld_accept_high", "mala_sgld_adapt_c0",
    "mala_sgld_adapt_every", "mala_sgld_c0_min", "mala_sgld_c0_max"
]

const PROX_BLOCK_SAMPLER_METADATA_KEYS = [
    "prox_block_step_size", "prox_block_min_step_size",
    "prox_block_max_step_size", "prox_block_adapt_every",
    "prox_block_divergence_threshold", "prox_block_metric"
]

const HS_PRIOR_METADATA_KEYS = ["hs_tau_scale", "hs_local_scale"]

const RHS_PRIOR_METADATA_KEYS = [
    "rhs_tau_scale", "rhs_local_scale", "rhs_slab_df",
    "rhs_slab_s2", "rhs_pseudo_sigma_df", "rhs_pseudo_sigma_scale"
]

const LASSO_PRIOR_METADATA_KEYS = ["lasso_lambda", "lasso_lambda2_shape", "lasso_lambda2_rate"]

const SSLASSO_PRIOR_METADATA_KEYS = [
    "sslasso_spike_scale", "sslasso_slab_scale",
    "sslasso_theta_a", "sslasso_theta_b"
]

const PROX_PRIOR_METADATA_KEYS = [
    "prox_l1_lambda", "alpha_prior_shape", "alpha_prior_scale",
    "alpha_prior_kind", "prox_root_tol", "prox_root_maxiter"
]

const CAR_METADATA_KEYS = [
    "use_car", "car_distance", "car_weight_quantile", "car_precision",
    "sample_car_strength", "car_strength_prior_a", "car_strength_prior_b",
    "car_degree_scale", "car_kappa_init"
]

const ALWAYS_COMPARED_METADATA_KEYS = [
    "baseline_rate", "baseline_scale", "baseline_shape",
    "censoring_distribution", "censoring_upper",
    "p0", "constraint",
    "standardize", "tie_method", "seed"
]

_is_hs_family(prior::AbstractString) = occursin("hs", lowercase(prior))

function _relevant_prior_keys(prior::AbstractString)
    keys = String[]
    if prior in ("hs", "cox-hs")
        append!(keys, HS_PRIOR_METADATA_KEYS)
    end
    if prior in ("rhs", "new-rhs", "est-rhs", "cox-rhs")
        append!(keys, RHS_PRIOR_METADATA_KEYS)
    end
    if prior == "lasso"
        append!(keys, LASSO_PRIOR_METADATA_KEYS)
    end
    if prior == "sslasso"
        append!(keys, SSLASSO_PRIOR_METADATA_KEYS)
    end
    if prior in ("prox-lasso", "prox-sslasso")
        append!(keys, PROX_PRIOR_METADATA_KEYS)
    end
    return keys
end

function _relevant_sampler_keys(sampler::AbstractString)
    keys = copy(COMMON_SAMPLER_METADATA_KEYS)
    if sampler in ("nuts", "hmc")
        append!(keys, HMC_SAMPLER_METADATA_KEYS)
    elseif sampler == "mala"
        append!(keys, MALA_SAMPLER_METADATA_KEYS)
    elseif sampler == "prox_block"
        append!(keys, PROX_BLOCK_SAMPLER_METADATA_KEYS)
    end
    return keys
end

function simulation_metadata_matches_typed(run_dir::AbstractString, expected::AbstractDict)
    saved = read_run_metadata(run_dir)
    isempty(saved) && return false
    # Old runs did not record init and the matrix API silently dropped it.
    # Do not reuse those draws after fixing initialization, or reuse a run
    # produced from a different requested starting strategy.
    if haskey(expected, "init")
        haskey(saved, "init") && metadata_value_matches(saved["init"], expected["init"]) || return false
    end
    if haskey(expected, "zero_handling")
        get(saved, "zero_handling", nothing) == expected["zero_handling"] || return false
    end
    # Legacy runs are baseline only; never reuse them for a new DGP merely
    # because old metadata did not know about the new scenario keys.
    get(saved, "scenario", "baseline") == get(expected, "scenario", "baseline") || return false
    get(saved, "car_graph", "abundance") == get(expected, "car_graph", "abundance") || return false
    if get(expected, "scenario", "baseline") == "phylo_bm"
        for key in BROWNIAN_DGP_METADATA_KEYS
            haskey(expected, key) && haskey(saved, key) &&
                metadata_value_matches(saved[key], expected[key]) || return false
        end
    end
    for key in ("weak_multiplier", "null_correlation", "correlated_null_indices", "depth_median", "depth_log_sd")
        haskey(expected, key) || continue
        haskey(saved, key) && metadata_value_matches(saved[key], expected[key]) || return false
    end

    # Only compare keys present in BOTH saved and expected
    function _match(key)
        haskey(expected, key) || return true
        haskey(saved, key) || return true
        return metadata_value_matches(saved[key], expected[key])
    end

    # 1. Always-compared parameters
    for key in ALWAYS_COMPARED_METADATA_KEYS
        _match(key) || return false
    end

    # 2. Sampler: first check type, then type-specific params
    _match("sampler") || return false
    for key in _relevant_sampler_keys(string(get(expected, "sampler", "")))
        _match(key) || return false
    end

    # 3. Prior: first check type, then type-specific params
    _match("prior") || return false
    for key in _relevant_prior_keys(string(get(expected, "prior", "")))
        _match(key) || return false
    end

    # 4. CAR params: only for hs-family priors
    if _is_hs_family(string(get(expected, "prior", "")))
        for key in CAR_METADATA_KEYS
            _match(key) || return false
        end
    end

    return true
end

function _metadata_bool(run_dir::AbstractString, key::AbstractString, default::Bool)
    meta = read_run_metadata(run_dir)
    haskey(meta, key) || return default
    value = meta[key]
    value isa Bool && return value
    value isa Number && return value != 0
    value isa AbstractString || return default
    return lowercase(value) in ("true", "1", "yes", "y")
end

function _beta_index(name)
    m = match(r"^beta\[(\d+)\]$", string(name))
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function ensure_simulation_summary(summary::DataFrame, beta::AbstractVector)
    out = copy(summary)
    if !(:variable in propertynames(out))
        if :parameters in propertynames(out)
            rename!(out, :parameters => :variable)
        else
            out[!, :variable] = ["beta[$j]" for j in eachindex(beta)]
        end
    end
    idx = [_beta_index(v) for v in out.variable]
    if all(!isnothing, idx)
        out[!, :_beta_index] = Int.(idx)
        out = sort(out, :_beta_index)
        select!(out, Not(:_beta_index))
    end
    if nrow(out) != length(beta)
        keep = [_beta_index(v) !== nothing for v in out.variable]
        if any(keep)
            out = out[keep, :]
            idx = [_beta_index(v) for v in out.variable]
            out[!, :_beta_index] = Int.(idx)
            out = sort(out, :_beta_index)
            select!(out, Not(:_beta_index))
        end
    end
    nrow(out) == length(beta) ||
        throw(ArgumentError("posterior summary row count $(nrow(out)) does not match beta length $(length(beta))."))
    if !(:lower in propertynames(out)) && :q025 in propertynames(out)
        rename!(out, :q025 => :lower)
    end
    if !(:upper in propertynames(out)) && :q975 in propertynames(out)
        rename!(out, :q975 => :upper)
    end
    if !(:median in propertynames(out)) && Symbol("50.0%") in propertynames(out)
        rename!(out, Symbol("50.0%") => :median)
    end
    if !(:map in propertynames(out))
        out[!, :map] = :median in propertynames(out) ? Float64.(out[!, :median]) : fill(NaN, nrow(out))
    end
    out[!, :variable] = ["beta[$j]" for j in eachindex(beta)]
    out[!, :true_beta] = collect(Float64, beta)
    out[!, :nonzero] = beta .!= 0
    return out
end

function _saved_samples_wide(run_dir::AbstractString)
    wide_path = joinpath(run_dir, "samples_wide.csv")
    if file_complete(wide_path)
        return CSV.read(wide_path, DataFrame)
    end
    if file_complete(joinpath(run_dir, "result_bundle.jld2"))
        chains = BayesCox4Micro._load_chains_from_run(run_dir)
        return chains_to_wide_dataframe(chains)
    end
    throw(ArgumentError("no saved samples found in $run_dir"))
end

function saved_beta_draws_and_layout(run_dir::AbstractString, p::Integer)
    wide = _saved_samples_wide(run_dir)
    if all(c -> c in propertynames(wide), (:chain, :iter))
        wide = sort(wide, [:chain, :iter])
    end
    beta_cols = Symbol.("beta[$j]" for j in 1:p)
    missing_cols = [c for c in beta_cols if !(c in propertynames(wide))]
    isempty(missing_cols) ||
        throw(ArgumentError("saved samples in $run_dir are missing beta columns: $(join(string.(missing_cols), ", "))"))
    draws = Matrix{Float64}(wide[:, beta_cols])
    if :chain in propertynames(wide)
        chains = sort(unique(Int.(wide.chain)))
        counts = [count(==(c), Int.(wide.chain)) for c in chains]
        length(unique(counts)) == 1 ||
            throw(ArgumentError("saved sample chains in $run_dir have unequal lengths."))
        return draws, (chains = length(chains), draws = first(counts))
    end
    return draws, (chains = 1, draws = nrow(wide))
end

function _diagnostic_number(run_dir::AbstractString, key::AbstractString, default = NaN)
    path = joinpath(run_dir, "diagnostics.json")
    file_complete(path) || return default
    try
        diag = JSON.parsefile(path)
        haskey(diag, key) || return default
        value = diag[key]
        value isa Number && return Float64(value)
        value isa AbstractString && return parse(Float64, value)
    catch
    end
    return default
end

function metrics_from_saved_samples(run_dir::AbstractString, sim, summary::DataFrame)
    draws, layout = saved_beta_draws_and_layout(run_dir, length(sim.beta))
    tie_method = Symbol(_metadata_string(run_dir, "tie_method", "efron"))
    data = prepare_cox_data(
        sim.X,
        sim.time,
        sim.event;
        transform = :log,
        standardize = false,
        tie_method = tie_method,
    )
    loglik = posterior_event_loglik(data, draws)
    event_rows = findall(!=(0), data.event)
    ll_event = loglik[event_rows, :]
    loo = psis_loo_metrics(ll_event, layout.chains, layout.draws)
    beta_mean = Float64.(summary[!, :mean])
    beta_median = Float64.(summary[!, :median])
    beta_map = Float64.(summary[!, :map])
    metrics_mean = fit_metrics(data, draws, beta_mean; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_median = fit_metrics(data, draws, beta_median; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    metrics_map = fit_metrics(data, draws, beta_map; chains = layout.chains, draws_per_chain = layout.draws, ll_event = ll_event, loo = loo)
    return DataFrame(
        estimate = ["mean", "median", "MAP"],
        loglik = [metrics_mean.loglik, metrics_median.loglik, metrics_map.loglik],
        LPML = [metrics_mean.LPML, metrics_median.LPML, metrics_map.LPML],
        AIC = [metrics_mean.AIC, metrics_median.AIC, metrics_map.AIC],
        BIC = [metrics_mean.BIC, metrics_median.BIC, metrics_map.BIC],
        WAIC_pw1 = [metrics_mean.WAIC_pw1, metrics_median.WAIC_pw1, metrics_map.WAIC_pw1],
        WAIC_pw2 = [metrics_mean.WAIC_pw2, metrics_median.WAIC_pw2, metrics_map.WAIC_pw2],
        DIC = [metrics_mean.DIC, metrics_median.DIC, metrics_map.DIC],
        ELPD_LOO = [metrics_mean.ELPD_LOO, metrics_median.ELPD_LOO, metrics_map.ELPD_LOO],
        ELPD_LOO_SE = [metrics_mean.ELPD_LOO_SE, metrics_median.ELPD_LOO_SE, metrics_map.ELPD_LOO_SE],
        PLOO = [metrics_mean.PLOO, metrics_median.PLOO, metrics_map.PLOO],
        PLOO_SE = [metrics_mean.PLOO_SE, metrics_median.PLOO_SE, metrics_map.PLOO_SE],
        LOOIC = [metrics_mean.LOOIC, metrics_median.LOOIC, metrics_map.LOOIC],
        LOOIC_SE = [metrics_mean.LOOIC_SE, metrics_median.LOOIC_SE, metrics_map.LOOIC_SE],
        ParetoK_max = [metrics_mean.ParetoK_max, metrics_median.ParetoK_max, metrics_map.ParetoK_max],
        ParetoK_bad = [metrics_mean.ParetoK_bad, metrics_median.ParetoK_bad, metrics_map.ParetoK_bad],
        PSIS_LOO = [metrics_mean.PSIS_LOO, metrics_median.PSIS_LOO, metrics_map.PSIS_LOO],
        Cindex = [metrics_mean.Cindex, metrics_median.Cindex, metrics_map.Cindex],
        warmup_seconds = fill(_diagnostic_number(run_dir, "warmup_seconds", NaN), 3),
        sample_seconds = fill(_diagnostic_number(run_dir, "sample_seconds", NaN), 3),
        total_seconds = fill(_diagnostic_number(run_dir, "total_seconds", NaN), 3),
        allocated_mb = fill(_diagnostic_number(run_dir, "allocated_mb", NaN), 3),
        gc_seconds = fill(_diagnostic_number(run_dir, "gc_seconds", NaN), 3),
        gc_percent = fill(_diagnostic_number(run_dir, "gc_percent", NaN), 3),
    )
end

function annotate_simulation_metrics(metrics::DataFrame, sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    out = copy(metrics)
    mse = Dict(
        "mean" => mean((Float64.(summary[!, :mean]) .- sim.beta) .^ 2),
        "median" => mean((Float64.(summary[!, :median]) .- sim.beta) .^ 2),
        "MAP" => mean((Float64.(summary[!, :map]) .- sim.beta) .^ 2),
    )
    out[!, :prior] = fill(prior, nrow(out))
    out[!, :rep] = fill(rep, nrow(out))
    out[!, :n] = fill(n, nrow(out))
    out[!, :p] = fill(p, nrow(out))
    out[!, :nonzero_count] = fill(nonzero, nrow(out))
    out[!, :p0] = fill(p0, nrow(out))
    out[!, :baseline] = fill(string(baseline), nrow(out))
    out[!, :scenario] = fill(string(get(sim, :scenario, :baseline)), nrow(out))
    out[!, :MSE] = [get(mse, string(e), NaN) for e in out.estimate]
    add_simulation_coverage!(out, summary; beta = sim.beta)
    return out
end

function simulation_false_discovery_table(sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    fd = false_discovery(sim.beta, summary)
    return DataFrame(
        rep = [rep],
        n = [n],
        p = [p],
        nonzero_count = [nonzero],
        p0 = [p0],
        baseline = [string(baseline)],
        scenario = [string(get(sim, :scenario, :baseline))],
        prior = [prior],
        FP = [fd.FP],
        FN = [fd.FN],
        TP = [fd.TP],
        TN = [fd.TN],
        FDR = [fd.FDR],
    )
end

function load_or_recover_simulation_run(
    run_dir::AbstractString,
    sim;
    rep,
    n,
    p,
    nonzero,
    baseline,
    prior,
    p0,
    test_sim = nothing,
)
    summary = ensure_simulation_summary(CSV.read(joinpath(run_dir, "posterior_summary.csv"), DataFrame), sim.beta)
    saved_p0 = _metadata_number(run_dir, "p0", p0)
    metrics_path = joinpath(run_dir, "metrics.csv")
    if file_complete(metrics_path) && !file_complete(joinpath(run_dir, "complete.txt"))
        # Interrupted new runs may have metrics written before an incomplete
        # sample file. Validate saved samples before reusing such a run.
        saved_beta_draws_and_layout(run_dir, length(sim.beta))
    end
    metrics =
        if file_complete(metrics_path)
            loaded = CSV.read(metrics_path, DataFrame)
            p0_value = :p0 in propertynames(loaded) && nrow(loaded) > 0 ? loaded.p0[1] : saved_p0
            annotate_simulation_metrics(loaded, sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = p0_value, baseline = baseline, prior = prior)
        else
            recovered = metrics_from_saved_samples(run_dir, sim, summary)
            annotated = annotate_simulation_metrics(recovered, sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = saved_p0, baseline = baseline, prior = prior)
            annotated
        end

    if test_sim !== nothing
        add_simulation_test_metrics!(metrics, summary, sim, test_sim;
            tie_method = Symbol(_metadata_string(run_dir, "tie_method", "efron")))
    end
    write_table(metrics_path, metrics)
    fd_path = joinpath(run_dir, "false_discovery.csv")
    p0_value = :p0 in propertynames(metrics) && nrow(metrics) > 0 ? metrics.p0[1] : saved_p0
    fd = simulation_false_discovery_table(sim, summary; rep = rep, n = n, p = p, nonzero = nonzero, p0 = p0_value, baseline = baseline, prior = prior)
    if !file_complete(fd_path)
        write_table(fd_path, fd)
    end
    mark_simulation_run_complete(run_dir)
    return (summary = summary, metrics = metrics, false_discovery = fd)
end

include(joinpath(@__DIR__, "loo_evaluation.jl"))
