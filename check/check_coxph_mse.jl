#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using Survival

function _as_float_or_missing(x)
    x === missing && return missing
    return Float64(x)
end

function _maybe_property(x, name::Symbol)
    return hasproperty(x, name) ? getproperty(x, name) : missing
end

function fit_coxph_on_prepared_data(data::CoxPreparedData; tol::Real, l2_cost::Real)
    event_time = Survival.EventTime.(data.time, data.event .!= 0)
    fit = Survival.coxph(data.z, event_time; tol = tol, l2_cost = l2_cost)
    beta_hat = Float64.(Survival.coef(fit))
    if data.standardize
        beta_hat ./= data.col_scale
    end
    return fit, beta_hat
end

function coefficient_table(beta_hat::AbstractVector, beta_true::AbstractVector; context...)
    p = length(beta_true)
    df = DataFrame(
        variable = ["beta[$j]" for j in 1:p],
        beta_hat = beta_hat,
        beta_true = beta_true,
        nonzero = beta_true .!= 0,
        error = beta_hat .- beta_true,
        squared_error = (beta_hat .- beta_true) .^ 2,
    )
    for (key, value) in pairs(context)
        df[!, key] = fill(value, p)
    end
    select!(df, vcat(collect(keys(context)), [:variable, :beta_hat, :beta_true, :nonzero, :error, :squared_error]))
    return df
end

function metrics_row(beta_hat::AbstractVector, beta_true::AbstractVector; context...)
    nz = beta_true .!= 0
    zc = .!nz
    return DataFrame([merge(
        NamedTuple(context),
        (
        MSE = mean((beta_hat .- beta_true) .^ 2),
        MSE_nonzero = any(nz) ? mean((beta_hat[nz] .- beta_true[nz]) .^ 2) : missing,
        MSE_zero = any(zc) ? mean(beta_hat[zc] .^ 2) : missing,
        MAE = mean(abs.(beta_hat .- beta_true)),
        max_abs_error = maximum(abs.(beta_hat .- beta_true)),
        beta_hat_norm = norm(beta_hat),
        sum_beta_hat = sum(beta_hat),
        ),
    )])
end

function error_row(err; context...)
    return DataFrame([merge(
        NamedTuple(context),
        (
        MSE = missing,
        MSE_nonzero = missing,
        MSE_zero = missing,
        MAE = missing,
        max_abs_error = missing,
        beta_hat_norm = missing,
        sum_beta_hat = missing,
        error_type = string(typeof(err)),
        error_message = sprint(showerror, err),
        ),
    )])
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "checks", "coxph_mse")))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    seed0 = arg(args, "seed", 1)
    pseudocount = arg(args, "pseudocount", 0.0)
    standardize = arg(args, "standardize", false)
    closure = arg(args, "closure", true)
    tol = arg(args, "tol", 1e-4)
    l2_cost = arg(args, "l2-cost", 0.0)
    tie_method = parse_tie_method(args)

    metrics = DataFrame()
    coefs = DataFrame()

    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        seed = seed0 + rep - 1
        baseline_args = baseline_parameter_args(args, baseline)
        sim = generate_simulation_data(
            n,
            p,
            nonzero;
            baseline = baseline,
            seed = seed,
            baseline_args...,
        )
        context = (
            rep = rep,
            n = n,
            p = p,
            nonzero_count = nonzero,
            baseline = string(baseline),
            seed = seed,
            events = count(!=(0), sim.event),
            event_rate = mean(sim.event .!= 0),
            pseudocount = pseudocount,
            closure = closure,
            standardize = standardize,
            tie_method = string(tie_method),
            tol = tol,
            l2_cost = l2_cost,
        )
        try
            data = prepare_cox_data(
                sim.X,
                sim.time,
                sim.event;
                transform = :log,
                pseudocount = pseudocount,
                closure = closure,
                standardize = standardize,
                tie_method = tie_method,
            )
            fit, beta_hat = fit_coxph_on_prepared_data(data; tol = tol, l2_cost = l2_cost)
            row = metrics_row(beta_hat, sim.beta; context...)
            row.converged = [true]
            row.loglik = [_as_float_or_missing(_maybe_property(fit, :loglikelihood))]
            append!(metrics, row; cols = :union)
            append!(coefs, coefficient_table(beta_hat, sim.beta; context...); cols = :union)
            println("rep=$rep n=$n p=$p nonzero=$nonzero baseline=$baseline MSE=$(row.MSE[1])")
        catch err
            @warn "Cox PH fit failed." rep n p nonzero baseline exception=(err, catch_backtrace())
            row = error_row(err; context...)
            row.converged = [false]
            append!(metrics, row; cols = :union)
        end
    end

    CSV.write(joinpath(outdir, "coxph_mse_long.csv"), metrics)
    CSV.write(joinpath(outdir, "coxph_coefficients.csv"), coefs)

    if nrow(metrics) > 0 && :MSE in propertynames(metrics)
        ok = collect(skipmissing(metrics.MSE))
        if !isempty(ok)
            summary = combine(
                groupby(metrics, [:n, :p, :nonzero_count, :baseline, :pseudocount, :standardize, :l2_cost]),
                :MSE => mean => :MSE_mean,
                :MSE => std => :MSE_sd,
                :MSE_nonzero => mean => :MSE_nonzero_mean,
                :MSE_zero => mean => :MSE_zero_mean,
                :MAE => mean => :MAE_mean,
                :events => mean => :events_mean,
                :event_rate => mean => :event_rate_mean,
            )
            CSV.write(joinpath(outdir, "coxph_mse_summary.csv"), summary)
        end
    end

    println("Cox PH MSE check written to $(abspath(outdir))")
end

main()
