#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using BayesCox4Micro
using CSV
using DataFrames
using Dates
using MCMCChains
using Statistics

const REQUESTED_PRIORS = [
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

const REQUESTED_SAMPLERS = [:mh, :mala, :ula, :gibbs, :hmc, :nuts_explicit, :nuts_turing]

_csv_list(x::AbstractString) = strip.(split(x, ","; keepempty = false))
_env_int(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))
_env_float(name::AbstractString, default::Real) = parse(Float64, get(ENV, name, string(default)))
_env_symbol(name::AbstractString, default::Symbol) = Symbol(get(ENV, name, string(default)))

function _layout(result)
    if result.samples isa MHSamples
        return (chains = 1, draws = size(result.samples.draws, 1))
    elseif result.samples isa MultiMHSamples
        return (chains = length(result.samples.chains), draws = size(result.samples.chains[1].draws, 1))
    end
    chn = result_to_chains(result)
    return (chains = size(chn, 3), draws = size(chn, 1))
end

function _acceptance_rate(result)
    if result.samples isa MHSamples
        return result.samples.acceptance_rate
    elseif result.samples isa MultiMHSamples
        return mean(s.acceptance_rate for s in result.samples.chains)
    end
    return missing
end

function _ess_rhat(result)
    try
        stats = DataFrame(summarystats(result_to_chains(result)))
        ess_col =
            :ess_bulk in propertynames(stats) ? :ess_bulk :
            :ess in propertynames(stats) ? :ess : nothing
        rhat_col = :rhat in propertynames(stats) ? :rhat : nothing
        ess = ess_col === nothing ? missing : minimum(skipmissing(stats[!, ess_col]))
        rhat = rhat_col === nothing ? missing : maximum(skipmissing(stats[!, rhat_col]))
        return ess, rhat
    catch
        return missing, missing
    end
end

function _result_row(;
    prior_name,
    sampler,
    constraint,
    status,
    outer,
    result = nothing,
    err = nothing,
)
    if result === nothing
        return (
            prior = prior_name,
            sampler = string(sampler),
            constraint = string(constraint),
            status = status,
            total_seconds = ismissing(outer) ? missing : outer.time,
            allocated_mb = ismissing(outer) ? missing : outer.bytes / 2.0^20,
            gc_percent = ismissing(outer) || outer.time <= 0 ? missing : 100 * outer.gctime / outer.time,
            diagnostic_seconds = missing,
            diagnostic_allocated_mb = missing,
            diagnostic_gc_percent = missing,
            acceptance_rate = missing,
            chains = missing,
            draws_per_chain = missing,
            total_beta_draws = missing,
            p = missing,
            ess_bulk_min = missing,
            rhat_max = missing,
            error = err === nothing ? "" : sprint(showerror, err),
        )
    end
    diag = sampler_diagnostics(result)
    layout = _layout(result)
    draws = posterior_beta_draws(result)
    ess, rhat = _ess_rhat(result)
    return (
        prior = prior_name,
        sampler = string(sampler),
        constraint = string(constraint),
        status = status,
        total_seconds = outer.time,
        allocated_mb = outer.bytes / 2.0^20,
        gc_percent = outer.time <= 0 ? missing : 100 * outer.gctime / outer.time,
        diagnostic_seconds = diag.total_seconds,
        diagnostic_allocated_mb = diag.allocated_bytes / 2.0^20,
        diagnostic_gc_percent = diag.gc_percent,
        acceptance_rate = _acceptance_rate(result),
        chains = layout.chains,
        draws_per_chain = layout.draws,
        total_beta_draws = size(draws, 1),
        p = size(draws, 2),
        ess_bulk_min = ess,
        rhat_max = rhat,
        error = "",
    )
end

function _run_sampler_case(
    X,
    time,
    event;
    prior,
    sampler::Symbol,
    constraint::Symbol,
    n_samples::Integer,
    warmup::Integer,
    chains::Integer,
    target_accept::Real,
    case_seed::Integer,
    mala_step_size::Real,
)
    common = (
        prior = prior,
        sampler = sampler,
        n_samples = n_samples,
        warmup = warmup,
        chains = chains,
        multichains = true,
        target_accept = target_accept,
        seed = case_seed,
        use_car = false,
        constraint = constraint,
        prox_sum0_lambda = 1e-3,
        progress = false,
    )
    if sampler in (:mala, :ula)
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            prox_l1_lambda = 1e-3,
            mala_step_size = mala_step_size,
        )
    elseif sampler == :hmc
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            prox_l1_lambda = 1e-3,
            hmc_n_leapfrog = 2,
            hmc_step_size = 1e-3,
        )
    elseif sampler == :nuts_explicit
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            prox_l1_lambda = 1e-3,
            parallel_chains = true,
        )
    elseif sampler == :nuts_turing
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            prox_l1_lambda = 1e-3,
        )
    elseif sampler == :gibbs && prior isa SpikeSlabLassoPrior
        return sample_bayes_cox(
            X,
            time,
            event;
            common...,
            prox_l1_lambda = 1e-3,
        )
    else
        return sample_bayes_cox(X, time, event; common...)
    end
end

function main()
    n = _env_int("BC4M_N", 200)
    p = _env_int("BC4M_P", 50)
    nonzero = _env_int("BC4M_NONZERO", 12)
    baseline = _env_symbol("BC4M_BASELINE", :weibull)
    n_samples = _env_int("BC4M_SAMPLES", 500)
    warmup = _env_int("BC4M_WARMUP", 200)
    chains = _env_int("BC4M_CHAINS", 4)
    seed = _env_int("BC4M_SEED", 20260428)
    mala_step_size = _env_float("BC4M_MALA_STEP_SIZE", 1e-4)
    target_accept = _env_float("BC4M_TARGET_ACCEPT", 0.8)
    constraint = _env_symbol("BC4M_CONSTRAINT", :soft_center)
    outdir = get(ENV, "BC4M_OUTDIR", joinpath("results", "requested_sampler_matrix_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
    priors = haskey(ENV, "BC4M_PRIORS") ? _csv_list(ENV["BC4M_PRIORS"]) : REQUESTED_PRIORS
    samplers = haskey(ENV, "BC4M_SAMPLERS") ? Symbol.(_csv_list(ENV["BC4M_SAMPLERS"])) : REQUESTED_SAMPLERS

    mkpath(outdir)
    outfile = joinpath(outdir, "requested_sampler_matrix_summary.csv")
    sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
    rows = DataFrame()

    @info "Starting requested sampler matrix" n p nonzero baseline n_samples warmup chains constraint outdir threads = Threads.nthreads()
    for prior_name in priors, sampler in samplers
        prior = cox_paper_prior(prior_name; X = sim.X, time = sim.time, event = sim.event, p0 = nonzero)
        case_seed = seed + 1000 * findfirst(==(prior_name), priors) + findfirst(==(sampler), samplers)
        @info "Running case" prior = prior_name sampler constraint seed = case_seed
        row = try
            GC.gc()
            outer = @timed _run_sampler_case(
                sim.X,
                sim.time,
                sim.event;
                prior = prior,
                sampler = sampler,
                n_samples = n_samples,
                warmup = warmup,
                chains = chains,
                target_accept = target_accept,
                case_seed = case_seed,
                constraint = constraint,
                mala_step_size = mala_step_size,
            )
            _result_row(
                prior_name = prior_name,
                sampler = sampler,
                constraint = constraint,
                status = "ok",
                outer = outer,
                result = outer.value,
            )
        catch err
            _result_row(
                prior_name = prior_name,
                sampler = sampler,
                constraint = constraint,
                status = "error",
                outer = missing,
                err = err,
            )
        end
        push!(rows, row; cols = :union)
        CSV.write(outfile, rows)
        @info "Finished case" prior = prior_name sampler status = row.status seconds = row.total_seconds allocated_mb = row.allocated_mb
    end

    println(abspath(outfile))
    show(rows; allrows = true, allcols = true)
    println()
end

main()
