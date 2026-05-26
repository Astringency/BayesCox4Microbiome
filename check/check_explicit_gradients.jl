#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using BayesCox4Micro
using CSV
using DataFrames
using Dates
using ForwardDiff
using Random

const B = BayesCox4Micro

const GRADIENT_PRIORS = [
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

_env_int(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))
_env_symbol(name::AbstractString, default::Symbol) = Symbol(get(ENV, name, string(default)))

function _initial_for(prior, data)
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    return prior isa RegularizedHorseshoePrior ?
        initial_state(prior, p, n_events; n_obs = size(data.z, 1)) :
        initial_state(prior, p, n_events)
end

function _target_for(prior_name::AbstractString, prior, data)
    if prior_name == "prox-sslasso"
        p = size(data.z, 2)
        return CoxProxSSLassoTarget(
            data,
            prior,
            trues(p),
            Ref(0.5);
            constraint = :soft_center,
            prox_sum0_lambda = 1e-3,
            prox_l1_lambda = 1e-3,
            alpha_prior_shape = 1.0,
            alpha_prior_rate = 1.0,
            prox_root_tol = 1e-10,
            prox_root_maxiter = 100,
        )
    end
    return B._make_explicit_nuts_target(
        data,
        prior;
        constraint = :soft_center,
        prox_sum0_lambda = 1e-3,
        prox_l1_lambda = 1e-3,
        alpha_prior_shape = 1.0,
        alpha_prior_rate = 1.0,
        prox_root_tol = 1e-10,
        prox_root_maxiter = 100,
    )
end

function _initial_for_target(prior_name::AbstractString, prior, target, data, rng)
    if target isa Union{CoxProxLassoTarget,CoxProxSSLassoTarget}
        u = initial_state(target)
        u[end] = log(10.0)
    else
        u = _initial_for(prior, data)
    end
    u .+= 0.01 .* randn(rng, length(u))
    return u
end

function main()
    n = _env_int("BC4M_N", 200)
    p = _env_int("BC4M_P", 50)
    nonzero = _env_int("BC4M_NONZERO", 12)
    baseline = _env_symbol("BC4M_BASELINE", :weibull)
    seed = _env_int("BC4M_SEED", 20260428)
    outdir = get(ENV, "BC4M_OUTDIR", joinpath("results", "explicit_gradient_check_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
    mkpath(outdir)
    outfile = joinpath(outdir, "explicit_gradient_check.csv")

    rng = MersenneTwister(seed)
    sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
    data = prepare_cox_data(sim.X, sim.time, sim.event)
    rows = DataFrame()
    for prior_name in GRADIENT_PRIORS
        prior = cox_paper_prior(prior_name; X = sim.X, time = sim.time, event = sim.event, p0 = nonzero)
        row = try
            target = _target_for(prior_name, prior, data)
            u = _initial_for_target(prior_name, prior, target, data, rng)
            grad = zeros(Float64, length(u))
            t_manual = @timed gradient!(grad, target, u)
            t_ad = @timed ForwardDiff.gradient(x -> logdensity(target, x), u)
            grad_ad = t_ad.value
            max_abs = maximum(abs.(grad .- grad_ad))
            rel = max_abs / max(1.0, maximum(abs.(grad_ad)))
            lp_manual, grad2 = logdensity_and_gradient!(similar(grad), target, u)
            (
                prior = prior_name,
                status = "ok",
                dimension = length(u),
                max_abs_error = max_abs,
                relative_error = rel,
                logdensity = logdensity(target, u),
                logdensity_from_joint = lp_manual,
                manual_seconds = t_manual.time,
                manual_allocated_mb = t_manual.bytes / 2.0^20,
                forwarddiff_seconds = t_ad.time,
                forwarddiff_allocated_mb = t_ad.bytes / 2.0^20,
                error = "",
            )
        catch err
            (
                prior = prior_name,
                status = "error",
                dimension = missing,
                max_abs_error = missing,
                relative_error = missing,
                logdensity = missing,
                logdensity_from_joint = missing,
                manual_seconds = missing,
                manual_allocated_mb = missing,
                forwarddiff_seconds = missing,
                forwarddiff_allocated_mb = missing,
                error = sprint(showerror, err),
            )
        end
        push!(rows, row; cols = :union)
        CSV.write(outfile, rows)
        @info "Gradient check" prior = prior_name status = row.status max_abs_error = row.max_abs_error relative_error = row.relative_error
    end
    println(abspath(outfile))
    show(rows; allrows = true, allcols = true)
    println()
end

main()
