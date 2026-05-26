#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    Base.active_project() == project || Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using CSV
using DataFrames
using Dates
using Statistics

const ALL10_PRIORS = [
    "hs",
    "rhs",
    "new-rhs",
    "est-rhs",
    "cox-hs",
    "cox-rhs",
    "lasso",
    "sslasso",
    "prox-lasso",
    "prox-sslasso",
]

normalize_prior_name(prior::AbstractString) = lowercase(replace(prior, "_" => "-", " " => "-"))

function selected_hyperparams(prior::AbstractString, nonzero::Integer)
    key = normalize_prior_name(prior)
    p0x(mult::Real) = Float64(mult * nonzero)
    if key == "hs"
        return (p0 = p0x(1), hs_local_scale = 1.0, hs_tau_scale = 1.0)
    elseif key == "rhs"
        return (p0 = p0x(1), rhs_local_scale = 1.0, rhs_slab_df = 2.0, rhs_slab_s2 = 0.5)
    elseif key == "new-rhs"
        return (p0 = p0x(2), rhs_local_scale = 0.5, rhs_slab_df = 2.0, rhs_slab_s2 = 2.0)
    elseif key == "est-rhs"
        return (p0 = p0x(1), rhs_local_scale = 0.5, rhs_slab_df = 2.0, rhs_slab_s2 = 2.0)
    elseif key == "cox-hs"
        return (p0 = p0x(2), hs_local_scale = 1.0, hs_tau_scale = 1.0)
    elseif key == "cox-rhs"
        return (p0 = p0x(1), rhs_local_scale = 1.0, rhs_slab_df = 2.0, rhs_slab_s2 = 2.0)
    elseif key == "lasso"
        return (lasso_lambda2_shape = 0.5, lasso_lambda2_rate = 0.05)
    elseif key == "sslasso"
        return (sslasso_spike_scale = 0.04, sslasso_slab_scale = 1.0, sslasso_theta_a = 1.0, sslasso_theta_b = 1.0)
    elseif key == "prox-lasso"
        return (
            lasso_lambda2_shape = 0.5,
            lasso_lambda2_rate = 0.05,
            prox_l1_lambda = 0.001,
            alpha_prior_shape = 0.5,
            alpha_prior_rate = 0.05,
        )
    elseif key == "prox-sslasso"
        return (
            sslasso_spike_scale = 0.01,
            sslasso_slab_scale = 1.0,
            sslasso_theta_a = 1.0,
            sslasso_theta_b = 1.0,
            prox_l1_lambda = 0.001,
            alpha_prior_shape = 0.5,
            alpha_prior_rate = 0.05,
        )
    end
    throw(ArgumentError("unsupported prior: $prior"))
end

function add_hyperparam_columns!(row::DataFrame, params)
    for (key, value) in pairs(params)
        row[!, key] = fill(value, nrow(row))
    end
    return row
end

function error_row(; rep, seed, n, p, nonzero, baseline, prior, params, err)
    row = DataFrame(
        created_at = [string(now())],
        rep = [rep],
        seed = [seed],
        n = [n],
        p = [p],
        nonzero_count = [nonzero],
        baseline = [string(baseline)],
        prior = [prior],
        estimate = ["error"],
        MSE = [missing],
        Cindex = [missing],
        LPML = [missing],
        ELPD_LOO = [missing],
        ParetoK_max = [missing],
        event_rate = [missing],
        censor_rate = [missing],
        error = [sprint(showerror, err)],
    )
    add_hyperparam_columns!(row, params)
    return row
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "cox_all10_hyperparam_debug")))
    metrics_path = joinpath(outdir, "cox_all10_debug_metrics.csv")
    summary_path = joinpath(outdir, "cox_all10_debug_summary.csv")

    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200,400"))
    ps = split_ints(arg(args, "p", "50,150"))
    nonzeros = split_ints(arg(args, "nonzero", "6,12"))
    baselines = split_symbols(arg(args, "baseline", "weibull,exponential,gompertz"))
    priors = split_strings(arg(args, "priors", join(ALL10_PRIORS, ",")))
    sampler = Symbol(arg(args, "sampler", "nuts"))
    nuts_backend = Symbol(arg(args, "nuts-backend", "explicit"))
    n_samples = arg(args, "samples", 20)
    warmup = arg(args, "warmup", 10)
    chains = arg(args, "chains", 1)
    target_accept = arg(args, "target-accept", 0.8)
    seed0 = arg(args, "seed", 1)
    use_car_requested = arg(args, "use-car", true)
    car_distance = Symbol(arg(args, "car-distance", "bray"))
    pseudocount = arg(args, "pseudocount", 0.0)
    progress = arg(args, "progress", false)
    constraint_default = Symbol(arg(args, "constraint", "soft_center"))
    prox_constraint = Symbol(arg(args, "prox-constraint", "prox"))
    prox_sum0_lambda = arg(args, "prox-sum0-lambda", 1e-3)
    prox_root_tol = arg(args, "prox-root-tol", 1e-10)
    prox_root_maxiter = arg(args, "prox-root-maxiter", 100)

    rows = DataFrame()
    total = reps * length(ns) * length(ps) * length(nonzeros) * length(baselines) * length(priors)
    done = 0
    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        nonzero <= p || continue
        seed = seed0 + rep - 1
        sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
        event_rate = mean(sim.event .!= 0)
        censor_rate = mean(sim.event .== 0)
        for prior in priors
            done += 1
            key = normalize_prior_name(prior)
            params = selected_hyperparams(prior, nonzero)
            actual_use_car = use_car_requested && occursin("hs", key)
            constraint = startswith(key, "prox-") ? prox_constraint : constraint_default
            try
                fit = fit_and_evaluate(
                    sim.X,
                    sim.time,
                    sim.event;
                    prior_name = prior,
                    sampler = sampler,
                    nuts_backend = nuts_backend,
                    n_samples = n_samples,
                    warmup = warmup,
                    chains = chains,
                    target_accept = target_accept,
                    seed = seed,
                    use_car = actual_use_car,
                    car_distance = car_distance,
                    pseudocount = pseudocount,
                    constraint = constraint,
                    prox_sum0_lambda = prox_sum0_lambda,
                    prox_root_tol = prox_root_tol,
                    prox_root_maxiter = prox_root_maxiter,
                    progress = progress,
                    params...,
                )
                summary = ensure_simulation_summary(fit.summary, sim.beta)
                metrics = annotate_simulation_metrics(
                    fit.metrics,
                    sim,
                    summary;
                    rep = rep,
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    p0 = fit.p0,
                    baseline = baseline,
                    prior = prior,
                )
                metrics[!, :created_at] = fill(string(now()), nrow(metrics))
                metrics[!, :seed] = fill(seed, nrow(metrics))
                metrics[!, :event_rate] = fill(event_rate, nrow(metrics))
                metrics[!, :censor_rate] = fill(censor_rate, nrow(metrics))
                metrics[!, :constraint] = fill(string(constraint), nrow(metrics))
                metrics[!, :use_car] = fill(actual_use_car, nrow(metrics))
                add_hyperparam_columns!(metrics, params)
                append!(rows, metrics; cols = :union)
                CSV.write(metrics_path, rows)
                mean_row = metrics[metrics.estimate .== "mean", :]
                mse = nrow(mean_row) == 0 ? missing : mean_row.MSE[1]
                println("[$done/$total] ok baseline=$baseline n=$n p=$p nz=$nonzero prior=$prior mean_mse=$mse")
            catch err
                row = error_row(
                    rep = rep,
                    seed = seed,
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    baseline = baseline,
                    prior = prior,
                    params = params,
                    err = err,
                )
                row[!, :constraint] = [string(constraint)]
                row[!, :use_car] = [actual_use_car]
                row[!, :event_rate] = [event_rate]
                row[!, :censor_rate] = [censor_rate]
                append!(rows, row; cols = :union)
                CSV.write(metrics_path, rows)
                @warn "Cox all10 debug fit failed." baseline n p nonzero prior exception=(err, catch_backtrace())
            end
        end
    end

    ok = (:estimate in propertynames(rows)) ? rows[rows.estimate .!= "error", :] : DataFrame()
    if nrow(ok) > 0
        group_cols = [c for c in [:prior, :estimate, :n, :p, :nonzero_count, :baseline] if c in propertynames(ok)]
        summary = combine(
            groupby(ok, group_cols),
            :MSE => mean => :MSE,
            :Cindex => mean => :Cindex,
            :LPML => mean => :LPML,
            :ELPD_LOO => mean => :ELPD_LOO,
            :ParetoK_max => maximum => :ParetoK_max,
        )
        CSV.write(summary_path, summary)
        show(first(sort(summary, [:baseline, :n, :p, :nonzero_count, :prior, :estimate]), min(30, nrow(summary))); allrows = true, allcols = true)
        println()
    end
    println("Cox all10 hyperparameter debug metrics written to $(abspath(metrics_path))")
    println("Cox all10 hyperparameter debug summary written to $(abspath(summary_path))")
end

main()
