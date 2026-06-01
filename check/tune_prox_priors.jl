#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    Base.active_project() == project || Pkg.activate(dirname(project))
    return nothing
end

_activate_repo_project()

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using Dates
using MCMCChains

const PROX_PRIORS = ("prox-lasso", "prox-sslasso")
const PROX_SAMPLERS = (:hmc, :nuts, :mala)

const RESULT_COLUMNS = [
    :run_id,
    :created_at,
    :n,
    :p,
    :nonzero,
    :events,
    :event_rate,
    :seed,
    :prior,
    :sampler,
    :nuts_backend,
    :constraint,
    :prox_l1_lambda,
    :prox_sum0_lambda,
    :alpha_prior_shape,
    :alpha_prior_rate,
    :spike_scale,
    :slab_scale,
    :theta_a,
    :theta_b,
    :target_accept,
    :hmc_step_size,
    :hmc_n_leapfrog,
    :max_depth,
    :mala_step_size,
    :mala_target_accept,
    :mala_adapt_every,
    :mala_min_step,
    :mala_max_step,
    :mala_precondition,
    :mala_precondition_mode,
    :n_samples,
    :warmup,
    :chains,
    :thin,
    :acceptance_rate,
    :warmup_acceptance_rate,
    :final_step_mean,
    :min_ess_beta,
    :min_tail_ess_beta,
    :max_rhat_beta,
    :ess_eta,
    :tail_ess_eta,
    :rhat_eta,
    :beta_mse,
    :nonzero_mae,
    :zero_mean_abs,
    :max_abs_nonzero_mean,
    :topk_recovery,
    :topk_recovery_count,
    :sign_recovery,
    :sign_recovery_median,
    :sum_beta_mean,
    :sum_beta_sd,
    :logpost_mean,
    :logpost_sd,
    :logpost_min,
    :logpost_max,
    :all_zero_shrinkage,
    :divergences,
    :abnormal_transitions,
    :status,
    :notes,
]

function option(args, keys::AbstractVector{<:AbstractString}, default::AbstractString)
    for key in keys
        haskey(args, key) && return args[key]
    end
    return default
end

option(args, key::AbstractString, default::AbstractString) = option(args, [key], default)

function has_any_key(args, keys)
    return any(key -> haskey(args, key), keys)
end

function split_tokens(raw::AbstractString)
    return [strip(x) for x in split(raw, ",") if !isempty(strip(x))]
end

function parse_int_list(args, keys, default::AbstractString)
    return parse.(Int, split_tokens(option(args, keys, default)))
end

function parse_float_list(args, keys, default::AbstractString)
    return parse.(Float64, split_tokens(option(args, keys, default)))
end

function parse_string_list(args, keys, default::AbstractString)
    return split_tokens(option(args, keys, default))
end

function parse_symbol_list(args, keys, default::AbstractString)
    return Symbol.(split_tokens(option(args, keys, default)))
end

function parse_bool_option(args, key::AbstractString, default::Bool)
    return lowercase(option(args, key, string(default))) in ("1", "true", "yes", "y")
end

function parse_optional_float(args, key::AbstractString)
    haskey(args, key) || return nothing
    val = strip(args[key])
    if isempty(val) || lowercase(val) in ("nothing", "none", "missing", "auto")
        return nothing
    end
    return parse(Float64, val)
end

function parse_rate_token(token::AbstractString, p::Integer)
    raw = lowercase(replace(strip(token), "*" => ""))
    raw == "p" && return Float64(p)
    if endswith(raw, "p")
        coef = raw[1:(end - 1)]
        return (isempty(coef) ? 1.0 : parse(Float64, coef)) * p
    end
    return parse(Float64, raw)
end

function parse_rate_list(args, keys, default::AbstractString, p::Integer)
    return [parse_rate_token(tok, p) for tok in split_tokens(option(args, keys, default))]
end

function result_path(outdir::AbstractString, prior::AbstractString)
    stem = replace(prior, "-" => "_")
    return joinpath(outdir, "$(stem)_tuning_results.csv")
end

function safe_token(x)
    return replace(string(x), r"[^A-Za-z0-9_.=-]+" => "_")
end

function run_id(cfg)
    return join(
        safe_token.([
            cfg.prior,
            cfg.sampler,
            "n$(cfg.n)",
            "p$(cfg.p)",
            "k$(cfg.nonzero)",
            "seed$(cfg.seed)",
            cfg.constraint,
            "l1$(cfg.prox_l1_lambda)",
            "s0$(cfg.prox_sum0_lambda)",
            "ash$(cfg.alpha_prior_shape)",
            "art$(cfg.alpha_prior_rate)",
            "sp$(cfg.spike_scale)",
            "sl$(cfg.slab_scale)",
            "ta$(cfg.theta_a)",
            "tb$(cfg.theta_b)",
        ]),
        "__",
    )
end

function append_result!(path::AbstractString, row::Dict{Symbol,Any})
    mkpath(dirname(path))
    complete = Dict{Symbol,Any}(col => get(row, col, missing) for col in RESULT_COLUMNS)
    newdf = DataFrame(Dict(k => [v] for (k, v) in complete))
    if isfile(path)
        old = CSV.read(path, DataFrame)
        append!(old, newdf; cols = :union)
        select!(old, unique(vcat(RESULT_COLUMNS, propertynames(old))))
        CSV.write(path, old)
    else
        CSV.write(path, newdf[:, RESULT_COLUMNS])
    end
    return path
end

function write_coefficients!(outdir::AbstractString, cfg, beta_true, beta_mean, beta_median, beta_sd)
    dir = joinpath(outdir, "prox_tuning_details")
    mkpath(dir)
    rid = run_id(cfg)
    p = length(beta_true)
    df = DataFrame(
        run_id = fill(rid, p),
        variable = ["beta[$j]" for j in 1:p],
        index = collect(1:p),
        true_beta = beta_true,
        nonzero = beta_true .!= 0.0,
        posterior_mean = beta_mean,
        posterior_median = beta_median,
        posterior_sd = beta_sd,
        mean_error = beta_mean .- beta_true,
        median_error = beta_median .- beta_true,
    )
    path = joinpath(dir, "$(rid)_coefficients.csv")
    CSV.write(path, df)
    return path
end

function mhsample_acceptance(result)
    if result.samples isa MHSamples
        return (
            acceptance = result.samples.acceptance_rate,
            warmup_acceptance = result.samples.warmup_acceptance_rate,
            final_step_mean = mean(result.samples.final_step),
        )
    elseif result.samples isa MultiMHSamples
        chains = result.samples.chains
        return (
            acceptance = mean([s.acceptance_rate for s in chains]),
            warmup_acceptance = mean([s.warmup_acceptance_rate for s in chains]),
            final_step_mean = mean([mean(s.final_step) for s in chains]),
        )
    end
    return (acceptance = missing, warmup_acceptance = missing, final_step_mean = missing)
end

function _param_column(df::DataFrame)
    :parameters in propertynames(df) && return :parameters
    :parameter in propertynames(df) && return :parameter
    :name in propertynames(df) && return :name
    return nothing
end

function _numeric_values(df::DataFrame, col::Symbol)
    col in propertynames(df) || return Float64[]
    vals = Float64[]
    for x in df[!, col]
        if !(ismissing(x))
            xf = Float64(x)
            isfinite(xf) && push!(vals, xf)
        end
    end
    return vals
end

function _first_existing_col(df::DataFrame, cols)
    for col in cols
        col in propertynames(df) && return col
    end
    return nothing
end

function chain_diagnostics(chains)
    stats = try
        DataFrame(MCMCChains.summarystats(chains))
    catch err
        @warn "MCMCChains.summarystats failed; ESS/Rhat columns will be missing." exception=(err, catch_backtrace())
        DataFrame()
    end
    isempty(stats) && return Dict{Symbol,Any}()
    pcol = _param_column(stats)
    pcol === nothing && return Dict{Symbol,Any}()
    params = string.(stats[!, pcol])
    beta_rows = stats[occursin.(r"^beta\[\d+\]$", params), :]
    eta_rows = stats[params .== "eta", :]
    ess_col = _first_existing_col(stats, (:ess_bulk, :ess, :n_eff))
    tail_col = _first_existing_col(stats, (:ess_tail, :tail_ess))
    rhat_col = _first_existing_col(stats, (:rhat, :r_hat))
    beta_ess = ess_col === nothing ? Float64[] : _numeric_values(beta_rows, ess_col)
    beta_tail = tail_col === nothing ? Float64[] : _numeric_values(beta_rows, tail_col)
    beta_rhat = rhat_col === nothing ? Float64[] : _numeric_values(beta_rows, rhat_col)
    eta_ess = ess_col === nothing ? Float64[] : _numeric_values(eta_rows, ess_col)
    eta_tail = tail_col === nothing ? Float64[] : _numeric_values(eta_rows, tail_col)
    eta_rhat = rhat_col === nothing ? Float64[] : _numeric_values(eta_rows, rhat_col)
    return Dict{Symbol,Any}(
        :min_ess_beta => isempty(beta_ess) ? missing : minimum(beta_ess),
        :min_tail_ess_beta => isempty(beta_tail) ? missing : minimum(beta_tail),
        :max_rhat_beta => isempty(beta_rhat) ? missing : maximum(beta_rhat),
        :ess_eta => isempty(eta_ess) ? missing : minimum(eta_ess),
        :tail_ess_eta => isempty(eta_tail) ? missing : minimum(eta_tail),
        :rhat_eta => isempty(eta_rhat) ? missing : maximum(eta_rhat),
    )
end

function finite_or_missing(x)
    x === missing && return missing
    xf = Float64(x)
    return isfinite(xf) ? xf : missing
end

function bool_status(x::Bool)
    return x ? "true" : "false"
end

function summarize_fit(result, sim, cfg, outdir::AbstractString)
    beta_draws_model = posterior_beta_draws(result)
    beta_draws = beta_draws_unstandardized(result.data, beta_draws_model)
    beta_true = collect(Float64, sim.beta)
    p = length(beta_true)
    nz = findall(!iszero, beta_true)
    zidx = findall(iszero, beta_true)
    k = length(nz)
    beta_mean = vec(mean(beta_draws; dims = 1))
    beta_median = [median(view(beta_draws, :, j)) for j in 1:p]
    beta_sd = [std(view(beta_draws, :, j)) for j in 1:p]
    write_coefficients!(outdir, cfg, beta_true, beta_mean, beta_median, beta_sd)

    beta_mse = mean((beta_mean .- beta_true) .^ 2)
    nonzero_mae = isempty(nz) ? missing : mean(abs.(beta_mean[nz] .- beta_true[nz]))
    zero_mean_abs = isempty(zidx) ? 0.0 : mean(abs.(beta_mean[zidx]))
    top = partialsortperm(abs.(beta_mean), 1:k; rev = true)
    topk_count = length(intersect(Set(top), Set(nz)))
    topk_recovery = topk_count == k
    sign_recovery = all(j -> sign(beta_mean[j]) == sign(beta_true[j]), nz)
    sign_recovery_median = all(j -> sign(beta_median[j]) == sign(beta_true[j]), nz)
    sum_beta = vec(sum(beta_draws; dims = 2))
    logpost = posterior_logposterior_draws(result)
    max_abs_nz = isempty(nz) ? 0.0 : maximum(abs.(beta_mean[nz]))
    truth_scale = isempty(nz) ? 1.0 : median(abs.(beta_true[nz]))
    shrink = max_abs_nz < min(0.2, 0.25 * truth_scale)

    chains = result_to_chains(result)
    diag = chain_diagnostics(chains)
    accept = mhsample_acceptance(result)
    notes = String[]
    if cfg.sampler in (:hmc, :nuts)
        push!(notes, "transition divergence counts are not exposed by CoxSamplerResult")
    end
    if shrink
        push!(notes, "all nonzero beta posterior means are near zero")
    end
    if !(topk_recovery)
        push!(notes, "top-k recovery failed")
    end
    if !(sign_recovery)
        push!(notes, "sign recovery failed")
    end

    max_rhat_beta = get(diag, :max_rhat_beta, missing)
    min_ess_beta = get(diag, :min_ess_beta, missing)
    acc = accept.acceptance
    rhat_ok = max_rhat_beta === missing || Float64(max_rhat_beta) <= cfg.rhat_threshold
    ess_floor = min(cfg.ess_min, cfg.n_samples * cfg.chains)
    ess_ok = min_ess_beta === missing || Float64(min_ess_beta) >= ess_floor
    accept_ok =
        acc === missing ? true :
        cfg.sampler == :mala ? 0.15 <= Float64(acc) <= 0.95 :
        0.20 <= Float64(acc) <= 1.00
    diag_ok = rhat_ok && ess_ok && accept_ok
    recovery_ok = topk_recovery && sign_recovery && !shrink
    status =
        shrink ? "shrink_to_zero" :
        !diag_ok ? "diagnostic_fail" :
        recovery_ok ? "success" :
        "recovery_fail"
    if !rhat_ok
        push!(notes, "max beta Rhat exceeds threshold $(cfg.rhat_threshold)")
    end
    if !ess_ok
        push!(notes, "min beta ESS below threshold $ess_floor")
    end
    if !accept_ok
        push!(notes, "acceptance rate outside broad target range")
    end

    row = Dict{Symbol,Any}(
        :run_id => run_id(cfg),
        :created_at => string(now()),
        :n => cfg.n,
        :p => cfg.p,
        :nonzero => cfg.nonzero,
        :events => count(!=(0), sim.event),
        :event_rate => mean(sim.event .!= 0),
        :seed => cfg.seed,
        :prior => cfg.prior,
        :sampler => string(cfg.sampler),
        :nuts_backend => string(cfg.nuts_backend),
        :constraint => string(cfg.constraint),
        :prox_l1_lambda => cfg.prox_l1_lambda,
        :prox_sum0_lambda => cfg.prox_sum0_lambda,
        :alpha_prior_shape => cfg.alpha_prior_shape,
        :alpha_prior_rate => cfg.alpha_prior_rate,
        :spike_scale => cfg.spike_scale,
        :slab_scale => cfg.slab_scale,
        :theta_a => cfg.theta_a,
        :theta_b => cfg.theta_b,
        :target_accept => cfg.target_accept,
        :hmc_step_size => cfg.hmc_step_size === nothing ? missing : cfg.hmc_step_size,
        :hmc_n_leapfrog => cfg.hmc_n_leapfrog,
        :max_depth => cfg.max_depth,
        :mala_step_size => cfg.mala_step_size,
        :mala_target_accept => cfg.mala_target_accept,
        :mala_adapt_every => cfg.mala_adapt_every,
        :mala_min_step => cfg.mala_min_step,
        :mala_max_step => cfg.mala_max_step,
        :mala_precondition => cfg.mala_precondition,
        :mala_precondition_mode => string(cfg.mala_precondition_mode),
        :n_samples => cfg.n_samples,
        :warmup => cfg.warmup,
        :chains => cfg.chains,
        :thin => cfg.thin,
        :acceptance_rate => accept.acceptance,
        :warmup_acceptance_rate => accept.warmup_acceptance,
        :final_step_mean => accept.final_step_mean,
        :min_ess_beta => get(diag, :min_ess_beta, missing),
        :min_tail_ess_beta => get(diag, :min_tail_ess_beta, missing),
        :max_rhat_beta => get(diag, :max_rhat_beta, missing),
        :ess_eta => get(diag, :ess_eta, missing),
        :tail_ess_eta => get(diag, :tail_ess_eta, missing),
        :rhat_eta => get(diag, :rhat_eta, missing),
        :beta_mse => beta_mse,
        :nonzero_mae => nonzero_mae,
        :zero_mean_abs => zero_mean_abs,
        :max_abs_nonzero_mean => max_abs_nz,
        :topk_recovery => topk_recovery,
        :topk_recovery_count => topk_count,
        :sign_recovery => sign_recovery,
        :sign_recovery_median => sign_recovery_median,
        :sum_beta_mean => mean(sum_beta),
        :sum_beta_sd => std(sum_beta),
        :logpost_mean => mean(logpost),
        :logpost_sd => std(logpost),
        :logpost_min => minimum(logpost),
        :logpost_max => maximum(logpost),
        :all_zero_shrinkage => shrink,
        :divergences => missing,
        :abnormal_transitions => missing,
        :status => status,
        :notes => join(notes, "; "),
    )
    return row
end

function error_row(cfg, sim, err)
    return Dict{Symbol,Any}(
        :run_id => run_id(cfg),
        :created_at => string(now()),
        :n => cfg.n,
        :p => cfg.p,
        :nonzero => cfg.nonzero,
        :events => sim === nothing ? missing : count(!=(0), sim.event),
        :event_rate => sim === nothing ? missing : mean(sim.event .!= 0),
        :seed => cfg.seed,
        :prior => cfg.prior,
        :sampler => string(cfg.sampler),
        :nuts_backend => string(cfg.nuts_backend),
        :constraint => string(cfg.constraint),
        :prox_l1_lambda => cfg.prox_l1_lambda,
        :prox_sum0_lambda => cfg.prox_sum0_lambda,
        :alpha_prior_shape => cfg.alpha_prior_shape,
        :alpha_prior_rate => cfg.alpha_prior_rate,
        :spike_scale => cfg.spike_scale,
        :slab_scale => cfg.slab_scale,
        :theta_a => cfg.theta_a,
        :theta_b => cfg.theta_b,
        :target_accept => cfg.target_accept,
        :hmc_step_size => cfg.hmc_step_size === nothing ? missing : cfg.hmc_step_size,
        :hmc_n_leapfrog => cfg.hmc_n_leapfrog,
        :max_depth => cfg.max_depth,
        :mala_step_size => cfg.mala_step_size,
        :mala_target_accept => cfg.mala_target_accept,
        :mala_adapt_every => cfg.mala_adapt_every,
        :mala_min_step => cfg.mala_min_step,
        :mala_max_step => cfg.mala_max_step,
        :mala_precondition => cfg.mala_precondition,
        :mala_precondition_mode => string(cfg.mala_precondition_mode),
        :n_samples => cfg.n_samples,
        :warmup => cfg.warmup,
        :chains => cfg.chains,
        :thin => cfg.thin,
        :status => "error",
        :notes => "$(typeof(err)): $(sprint(showerror, err))",
    )
end

function theta_grid(args, p::Integer, nonzero::Integer)
    explicit = has_any_key(args, ("sslasso-theta-a", "theta-a", "sslasso-theta-b", "theta-b"))
    if explicit
        a = parse_float_list(args, ["sslasso-theta-a", "theta-a"], "1.0")
        b = parse_float_list(args, ["sslasso-theta-b", "theta-b"], "1.0")
        return a, b
    end
    expected = option(args, "expected-model-size", "truth")
    m =
        lowercase(expected) == "truth" ? Float64(nonzero) :
        lowercase(expected) in ("none", "flat") ? 1.0 :
        parse(Float64, expected)
    return [max(1.0, m)], [max(1.0, p - m)]
end

function preset_defaults(args, prior::AbstractString, key::AbstractString, fallback::AbstractString)
    preset = lowercase(option(args, "preset", "single"))
    preset == "single" && return fallback
    if prior == "prox-lasso"
        if key == "prox-l1-lambda"
            return preset == "quick" ? "1e-3,1e-4" : "1e-2,1e-3,1e-4"
        elseif key == "prox-sum0-lambda"
            return preset == "quick" ? "1e-2,1e-3" : "1e-2,1e-3,1e-4"
        elseif key == "alpha-prior-shape"
            return preset == "quick" ? "1.0" : "1.0,2.0"
        elseif key == "alpha-prior-rate"
            return preset == "quick" ? "0.5p,p" : "0.5p,p,2p,5p"
        end
    elseif prior == "prox-sslasso"
        if key == "prox-l1-lambda"
            return preset == "quick" ? "1e-3,1e-4" : "1e-2,1e-3,1e-4"
        elseif key == "prox-sum0-lambda"
            return preset == "quick" ? "1e-2,1e-3" : "1e-2,1e-3,1e-4"
        elseif key == "sslasso-spike-scale"
            return preset == "quick" ? "0.1,0.2" : "0.05,0.1,0.2,0.5"
        elseif key == "sslasso-slab-scale"
            return preset == "quick" ? "2.0,5.0" : "1.0,2.0,5.0,10.0"
        end
    end
    return fallback
end

function build_configs(args)
    ns = parse_int_list(args, ["n"], "1000")
    ps = parse_int_list(args, ["p"], "10")
    nonzeros = parse_int_list(args, ["nonzero", "k"], "2")
    priors = parse_string_list(args, ["prior", "priors"], "prox-lasso")
    samplers = parse_symbol_list(args, ["sampler", "samplers"], "hmc")
    constraints = parse_symbol_list(args, ["constraint", "constraints"], "prox")
    seeds = parse_int_list(args, ["seed", "seeds"], "1")
    n_samples = parse(Int, option(args, ["n-samples", "samples"], "300"))
    warmup = parse(Int, option(args, "warmup", "300"))
    chains = parse(Int, option(args, "chains", "2"))
    thin = parse(Int, option(args, "thin", "1"))
    nuts_backend = Symbol(option(args, "nuts-backend", "explicit"))
    target_accepts = parse_float_list(args, ["target-accept"], "0.9")
    hmc_steps = parse_int_list(args, ["hmc-n-leapfrog"], "20")
    hmc_step_size = parse_optional_float(args, "hmc-step-size")
    max_depths = parse_int_list(args, ["max-depth"], "10")
    mala_step_sizes = parse_float_list(args, ["mala-step-size"], "1e-5")
    mala_target_accepts = parse_float_list(args, ["mala-target-accept"], "0.574")
    mala_adapt_every = parse(Int, option(args, "mala-adapt-every", "50"))
    mala_min_step = parse(Float64, option(args, "mala-min-step", "1e-8"))
    mala_max_step = parse(Float64, option(args, "mala-max-step", "1e-2"))
    mala_precondition = parse_bool_option(args, "mala-precondition", true)
    mala_precondition_mode = Symbol(option(args, "mala-precondition-mode", "auto"))
    rhat_threshold = parse(Float64, option(args, "rhat-threshold", "1.10"))
    ess_min = parse(Float64, option(args, "ess-min", "20"))
    prox_root_tol = parse(Float64, option(args, "prox-root-tol", "1e-10"))
    prox_root_maxiter = parse(Int, option(args, "prox-root-maxiter", "100"))
    pseudocount = parse(Float64, option(args, "pseudocount", "0.0"))
    standardize = parse_bool_option(args, "standardize", false)
    progress = parse_bool_option(args, "progress", false)
    baseline = Symbol(option(args, "baseline", "weibull"))
    max_runs = parse(Int, option(args, "max-runs", string(typemax(Int))))

    configs = NamedTuple[]
    for n in ns, p in ps, nonzero in nonzeros
        nonzero <= p || continue
        for prior in priors
            prior in PROX_PRIORS || throw(ArgumentError("prior must be one of $(join(PROX_PRIORS, ", "))"))
            prox_l1s = parse_float_list(
                args,
                ["prox-l1-lambda"],
                preset_defaults(args, prior, "prox-l1-lambda", "1e-3"),
            )
            prox_sum0s = parse_float_list(
                args,
                ["prox-sum0-lambda"],
                preset_defaults(args, prior, "prox-sum0-lambda", "1e-3"),
            )
            alpha_shapes = parse_float_list(
                args,
                ["alpha-prior-shape"],
                preset_defaults(args, prior, "alpha-prior-shape", "1.0"),
            )
            alpha_rates = parse_rate_list(
                args,
                ["alpha-prior-rate"],
                preset_defaults(args, prior, "alpha-prior-rate", "p"),
                p,
            )
            spike_scales =
                prior == "prox-sslasso" ?
                parse_float_list(args, ["sslasso-spike-scale", "spike-scale"], preset_defaults(args, prior, "sslasso-spike-scale", "0.1")) :
                [missing]
            slab_scales =
                prior == "prox-sslasso" ?
                parse_float_list(args, ["sslasso-slab-scale", "slab-scale"], preset_defaults(args, prior, "sslasso-slab-scale", "5.0")) :
                [missing]
            theta_as, theta_bs = prior == "prox-sslasso" ? theta_grid(args, p, nonzero) : ([missing], [missing])
            for seed in seeds, sampler in samplers, constraint in constraints
                sampler in PROX_SAMPLERS || throw(ArgumentError("sampler must be one of :hmc, :nuts, :mala"))
                for prox_l1 in prox_l1s, prox_sum0 in prox_sum0s, alpha_shape in alpha_shapes, alpha_rate in alpha_rates
                    for spike in spike_scales, slab in slab_scales, theta_a in theta_as, theta_b in theta_bs
                        for target_accept in target_accepts, hmc_n in hmc_steps, max_depth in max_depths, mala_step in mala_step_sizes, mala_ta in mala_target_accepts
                            push!(configs, (
                                n = n,
                                p = p,
                                nonzero = nonzero,
                                prior = prior,
                                sampler = sampler,
                                seed = seed,
                                baseline = baseline,
                                constraint = constraint,
                                prox_l1_lambda = prox_l1,
                                prox_sum0_lambda = prox_sum0,
                                alpha_prior_shape = alpha_shape,
                                alpha_prior_rate = alpha_rate,
                                spike_scale = spike,
                                slab_scale = slab,
                                theta_a = theta_a,
                                theta_b = theta_b,
                                n_samples = n_samples,
                                warmup = warmup,
                                chains = chains,
                                thin = thin,
                                nuts_backend = nuts_backend,
                                target_accept = target_accept,
                                hmc_step_size = hmc_step_size,
                                hmc_n_leapfrog = hmc_n,
                                max_depth = max_depth,
                                mala_step_size = mala_step,
                                mala_target_accept = mala_ta,
                                mala_adapt_every = mala_adapt_every,
                                mala_min_step = mala_min_step,
                                mala_max_step = mala_max_step,
                                mala_precondition = mala_precondition,
                                mala_precondition_mode = mala_precondition_mode,
                                rhat_threshold = rhat_threshold,
                                ess_min = ess_min,
                                prox_root_tol = prox_root_tol,
                                prox_root_maxiter = prox_root_maxiter,
                                pseudocount = pseudocount,
                                standardize = standardize,
                                progress = progress,
                            ))
                            length(configs) >= max_runs && return configs
                        end
                    end
                end
            end
        end
    end
    return configs
end

function fit_config(cfg, outdir::AbstractString)
    sim = generate_simulation_data(cfg.n, cfg.p, cfg.nonzero; baseline = cfg.baseline, seed = cfg.seed)
    prior =
        if cfg.prior == "prox-lasso"
            BayesianLassoPrior(proximal = true)
        else
            SpikeSlabLassoPrior(
                proximal = true,
                spike_scale = Float64(cfg.spike_scale),
                slab_scale = Float64(cfg.slab_scale),
                theta_a = Float64(cfg.theta_a),
                theta_b = Float64(cfg.theta_b),
            )
        end
    common_kwargs = (
        prior = prior,
        sampler = cfg.sampler,
        nuts_backend = cfg.nuts_backend,
        n_samples = cfg.n_samples,
        warmup = cfg.warmup,
        thin = cfg.thin,
        chains = cfg.chains,
        multichains = false,
        seed = cfg.seed,
        use_car = false,
        progress = cfg.progress,
        pseudocount = cfg.pseudocount,
        standardize = cfg.standardize,
        target_accept = cfg.target_accept,
        constraint = cfg.constraint,
        prox_l1_lambda = cfg.prox_l1_lambda,
        prox_sum0_lambda = cfg.prox_sum0_lambda,
        alpha_prior_shape = cfg.alpha_prior_shape,
        alpha_prior_rate = cfg.alpha_prior_rate,
        prox_root_tol = cfg.prox_root_tol,
        prox_root_maxiter = cfg.prox_root_maxiter,
    )
    sampler_kwargs =
        cfg.sampler == :hmc ? (
            hmc_step_size = cfg.hmc_step_size,
            hmc_n_leapfrog = cfg.hmc_n_leapfrog,
        ) :
        cfg.sampler == :nuts ? (
            max_depth = cfg.max_depth,
        ) :
        cfg.sampler == :mala ? (
            mala_step_size = cfg.mala_step_size,
            mala_target_accept = cfg.mala_target_accept,
            mala_adapt_every = cfg.mala_adapt_every,
            mala_min_step = cfg.mala_min_step,
            mala_max_step = cfg.mala_max_step,
            mala_precondition = cfg.mala_precondition,
            mala_precondition_mode = cfg.mala_precondition_mode,
        ) :
        NamedTuple()
    result = sample_bayes_cox(
        sim.X,
        sim.time,
        sim.event;
        common_kwargs...,
        sampler_kwargs...,
    )
    return summarize_fit(result, sim, cfg, outdir)
end

function main()
    args = parse_cli_args()
    if haskey(args, "help") || haskey(args, "h")
        println("""
        Usage:
          julia +1.11.7 --project=. check/tune_prox_priors.jl [options]

        Common options:
          --n 1000 --p 10 --nonzero 2
          --priors prox-lasso,prox-sslasso
          --samplers hmc,nuts,mala
          --constraint prox
          --prox-l1-lambda 1e-2,1e-3,1e-4
          --prox-sum0-lambda 1e-2,1e-3
          --alpha-prior-shape 1.0
          --alpha-prior-rate p
          --sslasso-spike-scale 0.2
          --sslasso-slab-scale 5.0
          --expected-model-size truth
          --samples 300 --warmup 300 --chains 2
          --outdir check/results

        Presets:
          --preset single   one conservative configuration
          --preset quick    small staged grid
          --preset full     recommended starting grid
        """)
        return nothing
    end
    valid = [
        "help",
        "h",
        "outdir",
        "preset",
        "max-runs",
        "n",
        "p",
        "nonzero",
        "k",
        "prior",
        "priors",
        "sampler",
        "samplers",
        "seed",
        "seeds",
        "baseline",
        "constraint",
        "constraints",
        "prox-l1-lambda",
        "prox-sum0-lambda",
        "alpha-prior-shape",
        "alpha-prior-rate",
        "prox-root-tol",
        "prox-root-maxiter",
        "sslasso-spike-scale",
        "spike-scale",
        "sslasso-slab-scale",
        "slab-scale",
        "sslasso-theta-a",
        "theta-a",
        "sslasso-theta-b",
        "theta-b",
        "expected-model-size",
        "n-samples",
        "samples",
        "warmup",
        "chains",
        "thin",
        "nuts-backend",
        "target-accept",
        "hmc-step-size",
        "hmc-n-leapfrog",
        "max-depth",
        "mala-step-size",
        "mala-target-accept",
        "mala-adapt-every",
        "mala-min-step",
        "mala-max-step",
        "mala-precondition",
        "mala-precondition-mode",
        "rhat-threshold",
        "ess-min",
        "pseudocount",
        "standardize",
        "progress",
    ]
    require_known_args!(args, valid)
    outdir = option(args, "outdir", joinpath("check", "results"))
    mkpath(outdir)
    configs = build_configs(args)
    println("Running $(length(configs)) proximal tuning configuration(s).")
    for (i, cfg) in enumerate(configs)
        println("[$i/$(length(configs))] $(run_id(cfg))")
        sim_for_error = nothing
        row = try
            fit_config(cfg, outdir)
        catch err
            @warn "Configuration failed." run_id = run_id(cfg) exception=(err, catch_backtrace())
            error_row(cfg, sim_for_error, err)
        end
        append_result!(result_path(outdir, cfg.prior), row)
        combined = joinpath(outdir, "prox_priors_tuning_results.csv")
        append_result!(combined, row)
        println("    status=$(row[:status]) beta_mse=$(get(row, :beta_mse, missing)) notes=$(get(row, :notes, missing))")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
