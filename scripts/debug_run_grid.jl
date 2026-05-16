#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using CSV
using DataFrames
using Dates
using LinearAlgebra
using MCMCChains
using Statistics

function split_floats(s::AbstractString)
    return parse.(Float64, strip.(split(s, ",")))
end

function split_bools(s::AbstractString)
    return [lowercase(strip(x)) in ("true", "1", "yes", "y") for x in split(s, ",")]
end

function split_sample_warmups(s::AbstractString)
    pairs = NamedTuple[]
    for raw in split(s, ",")
        value = strip(raw)
        isempty(value) && continue
        parts = split(value, r"[:/]")
        length(parts) == 2 || throw(ArgumentError("sample-warmup entries must look like 500:200."))
        push!(pairs, (samples = parse(Int, strip(parts[1])), warmup = parse(Int, strip(parts[2]))))
    end
    isempty(pairs) && throw(ArgumentError("at least one sample-warmup pair is required."))
    return pairs
end

function append_row!(path::AbstractString, row)
    mkpath(dirname(path))
    exists = isfile(path) && filesize(path) > 0
    CSV.write(path, DataFrame([row]); append = exists, writeheader = !exists)
    return path
end

function finite_col(rows::DataFrame, col::Symbol, reducer, default = missing)
    col in propertynames(rows) || return default
    vals = Float64[]
    for x in skipmissing(rows[!, col])
        x isa Number || continue
        xf = Float64(x)
        isfinite(xf) && push!(vals, xf)
    end
    isempty(vals) ? default : reducer(vals)
end

function beta_chain_diagnostics(result::CoxSamplerResult)
    chains = result_to_chains(result)
    summary = DataFrame(MCMCChains.summarystats(chains))
    param_col =
        :parameters in propertynames(summary) ? :parameters :
        :parameter in propertynames(summary) ? :parameter :
        nothing
    beta_rows = summary
    if param_col !== nothing
        beta_rows = summary[occursin.(r"^beta\[\d+\]$", string.(summary[!, param_col])), :]
    end

    wide = chains_to_wide_dataframe(chains)
    div_cols = [
        c for c in propertynames(wide)
        if occursin("diverg", lowercase(String(c))) || occursin("numerical_error", lowercase(String(c)))
    ]
    divergences = 0
    for c in div_cols
        divergences += count(x -> x isa Number && Float64(x) != 0.0, skipmissing(wide[!, c]))
    end

    return (
        Rhat_max = finite_col(beta_rows, :rhat, maximum),
        ESS_bulk_min = finite_col(beta_rows, :ess_bulk, minimum, finite_col(beta_rows, :ess, minimum)),
        ESS_tail_min = finite_col(beta_rows, :ess_tail, minimum),
        divergences = divergences,
    )
end

function resolve_debug_p0(prior::AbstractString, sim, p0_arg, nonzero::Integer, p0_alpha::Real, pseudocount::Real)
    requested =
        p0_arg === :truth ? nonzero :
        p0_arg === nothing ? nothing :
        p0_arg
    if requested !== nothing
        return requested
    end
    p0_value = prior_p0(prior_from_name(
        prior,
        sim.X,
        sim.time,
        sim.event;
        p0 = nothing,
        p0_alpha = p0_alpha,
        realdata = false,
        pseudocount = pseudocount,
    ))
    return p0_value === missing ? nothing : p0_value
end

function mean_metric(metrics::DataFrame, col::Symbol)
    rows = metrics[metrics.estimate .== "mean", :]
    nrow(rows) == 0 && return missing
    return rows[1, col]
end

function debug_row_from_fit(; sim, data, fit, summary, rep, n, p, nonzero, baseline, prior,
    sampler, nuts_backend, samples, warmup, chains, use_car, car_distance, pseudocount,
    p0, tau_scale_mode, seed)
    beta_hat = Float64.(summary.mean)
    truth = Float64.(sim.beta)
    nz = truth .!= 0.0
    zc = .!nz
    fd = false_discovery(truth, summary)
    diag = beta_chain_diagnostics(fit.result)
    ll_true = cox_logpartiallikelihood(data, truth)
    ll_zero = cox_logpartiallikelihood(data, zeros(p))
    return (
        created_at = string(now()),
        rep = rep,
        seed = seed,
        baseline = string(baseline),
        n = n,
        p = p,
        nonzero = nonzero,
        prior = prior,
        sampler = string(sampler),
        nuts_backend = string(nuts_backend),
        samples = samples,
        warmup = warmup,
        chains = chains,
        p0 = p0,
        constraint = "soft_center",
        use_car = use_car,
        car_distance = string(car_distance),
        pseudocount = pseudocount,
        tau_scale_mode = tau_scale_mode,
        event_rate = mean(sim.event .!= 0),
        censor_rate = mean(sim.event .== 0),
        ll_true_minus_zero = ll_true - ll_zero,
        mse_zero = mean(truth .^ 2),
        mse_mean = mean((beta_hat .- truth) .^ 2),
        mse_nonzero = mean((beta_hat[nz] .- truth[nz]) .^ 2),
        mse_zero_coeffs = mean(beta_hat[zc] .^ 2),
        beta_hat_norm = norm(beta_hat),
        sum_beta_hat = sum(beta_hat),
        FP = fd.FP,
        FN = fd.FN,
        TP = fd.TP,
        TN = fd.TN,
        FDR = fd.FDR,
        Cindex = mean_metric(fit.metrics, :Cindex),
        Rhat_max = diag.Rhat_max,
        ESS_bulk_min = diag.ESS_bulk_min,
        ESS_tail_min = diag.ESS_tail_min,
        divergences = diag.divergences,
        git_commit_hash = BayesCox4Micro._git_commit_hash(),
        error = "",
    )
end

function main()
    args = parse_cli_args()
    outdir = arg(args, "outdir", joinpath("results", "debug_grid"))
    outpath = joinpath(outdir, "debug_grid_metrics.csv")
    reps = arg(args, "reps", 1)
    baselines = split_symbols(arg(args, "baseline", "weibull,exponential,gompertz"))
    ns = split_ints(arg(args, "n", "200,400"))
    ps = split_ints(arg(args, "p", "50,150"))
    nonzeros = split_ints(arg(args, "nonzero", "6,12"))
    priors = split_strings(arg(args, "priors", "cox-rhs"))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    nuts_backends = split_symbols(arg(args, "nuts-backends", arg(args, "nuts-backend", "turing")))
    sample_warmups = split_sample_warmups(arg(args, "sample-warmup", "$(arg(args, "samples", 500)):$(arg(args, "warmup", 200))"))
    chains = arg(args, "chains", 4)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 1)
    use_cars = split_bools(arg(args, "use-car", "true"))
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    pseudocounts = split_floats(arg(args, "pseudocount", "0.0"))
    tau_modes = split_strings(arg(args, "tau-scale-mode", get(ENV, "BAYESCOX_TAU_CAUCHY_SCALE", "tau0_squared")))
    p0_arg = parse_optional_p0(args; default = :truth)
    p0_alpha = arg(args, "p0-alpha", 0.05)
    progress = arg(args, "progress", false)
    resume = false

    mkpath(outdir)
    @info "Debug grid forces resume=false." resume outpath

    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        sim_seed = seed0 + rep - 1
        sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = sim_seed)
        for prior in priors, nuts_backend in nuts_backends, sw in sample_warmups,
            use_car_requested in use_cars, pseudocount in pseudocounts, tau_mode in tau_modes

            actual_use_car = use_car_requested && occursin("hs", lowercase(prior))
            old_tau = get(ENV, "BAYESCOX_TAU_CAUCHY_SCALE", nothing)
            ENV["BAYESCOX_TAU_CAUCHY_SCALE"] = tau_mode
            try
                p0 = resolve_debug_p0(prior, sim, p0_arg, nonzero, p0_alpha, pseudocount)
                data = prepare_cox_data(sim.X, sim.time, sim.event; transform = :log, pseudocount = pseudocount)
                fit = fit_and_evaluate(
                    sim.X,
                    sim.time,
                    sim.event;
                    prior_name = prior,
                    sampler = sampler,
                    nuts_backend = nuts_backend,
                    n_samples = sw.samples,
                    warmup = sw.warmup,
                    chains = chains,
                    target_accept = target_accept,
                    seed = sim_seed,
                    use_car = actual_use_car,
                    car_distance = car_distance,
                    p0 = p0,
                    p0_alpha = p0_alpha,
                    pseudocount = pseudocount,
                    progress = progress,
                )
                summary = ensure_simulation_summary(fit.summary, sim.beta)
                row = debug_row_from_fit(
                    sim = sim,
                    data = data,
                    fit = fit,
                    summary = summary,
                    rep = rep,
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    baseline = baseline,
                    prior = prior,
                    sampler = sampler,
                    nuts_backend = nuts_backend,
                    samples = sw.samples,
                    warmup = sw.warmup,
                    chains = chains,
                    use_car = actual_use_car,
                    car_distance = car_distance,
                    pseudocount = pseudocount,
                    p0 = p0,
                    tau_scale_mode = tau_mode,
                    seed = sim_seed,
                )
                append_row!(outpath, row)
                println("wrote debug row: baseline=$baseline n=$n p=$p nz=$nonzero prior=$prior mse=$(row.mse_mean)")
            catch err
                row = (
                    created_at = string(now()),
                    rep = rep,
                    seed = sim_seed,
                    baseline = string(baseline),
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    prior = prior,
                    sampler = string(sampler),
                    nuts_backend = string(nuts_backend),
                    samples = sw.samples,
                    warmup = sw.warmup,
                    chains = chains,
                    p0 = missing,
                    constraint = "soft_center",
                    use_car = actual_use_car,
                    car_distance = string(car_distance),
                    pseudocount = pseudocount,
                    tau_scale_mode = tau_mode,
                    event_rate = mean(sim.event .!= 0),
                    censor_rate = mean(sim.event .== 0),
                    ll_true_minus_zero = missing,
                    mse_zero = mean(sim.beta .^ 2),
                    mse_mean = missing,
                    mse_nonzero = missing,
                    mse_zero_coeffs = missing,
                    beta_hat_norm = missing,
                    sum_beta_hat = missing,
                    FP = missing,
                    FN = missing,
                    TP = missing,
                    TN = missing,
                    FDR = missing,
                    Cindex = missing,
                    Rhat_max = missing,
                    ESS_bulk_min = missing,
                    ESS_tail_min = missing,
                    divergences = missing,
                    git_commit_hash = BayesCox4Micro._git_commit_hash(),
                    error = sprint(showerror, err),
                )
                append_row!(outpath, row)
                @warn "Debug fit failed." baseline n p nonzero prior nuts_backend sw use_car_requested pseudocount tau_mode exception=(err, catch_backtrace())
            finally
                if old_tau === nothing
                    delete!(ENV, "BAYESCOX_TAU_CAUCHY_SCALE")
                else
                    ENV["BAYESCOX_TAU_CAUCHY_SCALE"] = old_tau
                end
            end
        end
    end
    println("Debug grid written to $(abspath(outpath))")
end

main()
