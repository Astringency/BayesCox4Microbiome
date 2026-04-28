include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using BayesCox4Micro
using CSV
using DataFrames
using Dates
using Statistics

const PRIORS = ["hs", "rhs", "new-rhs", "est-rhs", "cox-hs", "cox-rhs", "lasso", "sslasso"]
const SAMPLERS = [:mh, :mala, :ula, :gibbs, :nuts_explicit, :nuts_turing]

_env_int(name::AbstractString, default::Integer) = parse(Int, get(ENV, name, string(default)))
_env_symbol(name::AbstractString, default::Symbol) = Symbol(get(ENV, name, string(default)))
_env_list(name::AbstractString, default) =
    haskey(ENV, name) ? split(ENV[name], ","; keepempty = false) : default

function _result_row(; prior, sampler, status, seconds, result = nothing, err = nothing)
    if result === nothing
        return (
            prior = prior,
            sampler = string(sampler),
            status = status,
            seconds = seconds,
            allocated_mb = missing,
            gc_percent = missing,
            n_draws = missing,
            p = missing,
            accept_rate = missing,
            error = err === nothing ? "" : sprint(showerror, err),
        )
    end
    draws = posterior_beta_draws(result)
    diag = sampler_diagnostics(result)
    accept =
        result.samples isa MHSamples ? result.samples.acceptance_rate :
        result.samples isa MultiMHSamples ? mean(s.acceptance_rate for s in result.samples.chains) :
        missing
    return (
        prior = prior,
        sampler = string(sampler),
        status = status,
        seconds = seconds,
        allocated_mb = diag.allocated_bytes / 1024^2,
        gc_percent = diag.gc_percent,
        n_draws = size(draws, 1),
        p = size(draws, 2),
        accept_rate = accept,
        error = "",
    )
end

function main()
    outdir = joinpath("results", "sampler_matrix_$(Dates.format(now(), "yyyymmdd_HHMMSS"))")
    mkpath(outdir)
    outfile = joinpath(outdir, "sampler_matrix_summary.csv")

    n = _env_int("BC4M_N", 200)
    p = _env_int("BC4M_P", 50)
    nonzero = _env_int("BC4M_NONZERO", 6)
    baseline = _env_symbol("BC4M_BASELINE", :weibull)
    seed = _env_int("BC4M_SEED", 20260427)

    sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
    n_samples = _env_int("BC4M_SAMPLES", 500)
    warmup = _env_int("BC4M_WARMUP", 200)
    rows = DataFrame()

    priors = _env_list("BC4M_PRIORS", PRIORS)
    samplers = Symbol.(_env_list("BC4M_SAMPLERS", string.(SAMPLERS)))

    for prior_name in priors, sampler in samplers
        @info "Running sampler matrix case" prior = prior_name sampler = sampler
        prior = cox_paper_prior(prior_name; X = sim.X, time = sim.time, event = sim.event, p0 = nonzero)
        seconds = NaN
        row =
            try
                common_kwargs = (
                    prior = prior,
                    sampler = sampler,
                    n_samples = n_samples,
                    warmup = warmup,
                    chains = 1,
                    seed = seed,
                    progress = false,
                    use_car = false,
                    constraint = :soft_center,
                )
                timed = @timed if sampler in (:mala, :ula)
                    sample_bayes_cox(sim.X, sim.time, sim.event; common_kwargs..., mala_step_size = 1e-4)
                else
                    sample_bayes_cox(sim.X, sim.time, sim.event; common_kwargs...)
                end
                seconds = timed.time
                _result_row(
                    prior = prior_name,
                    sampler = sampler,
                    status = "ok",
                    seconds = seconds,
                    result = timed.value,
                )
            catch err
                _result_row(
                    prior = prior_name,
                    sampler = sampler,
                    status = "error",
                    seconds = seconds,
                    err = err,
                )
            end
        push!(rows, row; cols = :union)
        CSV.write(outfile, rows)
    end

    println(outfile)
    show(rows; allrows = true, allcols = true)
    println()
end

main()
