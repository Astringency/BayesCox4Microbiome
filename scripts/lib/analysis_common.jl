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
using Random
using Statistics



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
)
const NUTS_SAMPLER_CLI_KEYS = (
    "nuts-backend",
    "max-depth",
)
const HMC_SAMPLER_CLI_KEYS = (
    "hmc-n-leapfrog",
    "hmc-step-size",
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
        prox_block_step_size = optional_float_arg(args, "prox-block-step-size"),
        prox_block_min_step_size = arg(args, "prox-block-min-step-size", 1e-8),
        prox_block_max_step_size = arg(args, "prox-block-max-step-size", 1.0),
        prox_block_adapt_every = arg(args, "prox-block-adapt-every", 50),
        prox_block_divergence_threshold = arg(args, "prox-block-divergence-threshold", 1000.0),
        prox_block_metric = Symbol(arg(args, "prox-block-metric", "unit_diag")),
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
    get(saved,"rhs_scale_calibration","none")==get(expected,"rhs_scale_calibration","none") || return false
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
    path = joinpath(run_dir, "diagnostics_summary.csv")
    file_complete(path) || return default
    table = read_diagnostics(run_dir; category = "runtime")
    col = Symbol(key)
    isempty(table) || !(col in propertynames(table)) ? default :
        (value = first(table[!, col]); value isa Number ? Float64(value) : default)
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
    estimates = (mean = Float64.(summary.mean), median = Float64.(summary.median), MAP = Float64.(summary.map))
    metrics = posterior_fit_metrics(data, draws, estimates; chains = layout.chains, draws_per_chain = layout.draws)
    for key in ("warmup_seconds", "sample_seconds", "total_seconds", "allocated_mb", "gc_seconds", "gc_percent")
        metrics[!, Symbol(key)] = fill(_diagnostic_number(run_dir, key), nrow(metrics))
    end
    return metrics
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

include(joinpath(@__DIR__, "simulation", "data.jl"))
include(joinpath(@__DIR__, "simulation", "scenarios.jl"))
include(joinpath(@__DIR__, "plotting", "analysis_plots.jl"))
