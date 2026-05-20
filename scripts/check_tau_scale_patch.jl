#!/usr/bin/env julia

if any(arg -> arg in ("--help", "-h"), ARGS)
    println("""
    Usage:
      julia --project=. scripts/check_tau_scale_patch.jl --output results/diagnosis/tau_scale_patch_check.md

    Options:
      --output PATH   Markdown file to write.
      --n INT         Simulated sample size for est-rhs construction. Default: 200.
      --p INT         Number of covariates. Default: 50.
      --nonzero INT   Number of nonzero coefficients. Default: 6.
      --baseline STR  Baseline for generated data. Default: weibull.
      --seed INT      Simulation seed. Default: 1.
    """)
    exit(0)
end

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "analysis_common.jl"))

using Dates

function resolved_scale(name::AbstractString, scale::Real, sim; n::Integer, p::Integer, nonzero::Integer)
    key = lowercase(name)
    kwargs =
        if key in ("hs", "cox-hs")
            (p0 = nonzero, hs_tau_scale = scale)
        else
            (p0 = nonzero, rhs_tau_scale = scale)
        end
    prior = cox_paper_prior(
        name;
        X = sim.X,
        time = sim.time,
        event = sim.event,
        pseudocount = 0.0,
        kwargs...,
    )
    n_events = count(!=(0), sim.event)
    if prior isa HorseshoePrior
        return BayesCox4Micro._resolved_tau_scale(prior, p, n_events)
    elseif prior isa RegularizedHorseshoePrior
        return BayesCox4Micro._resolved_tau_scale(prior, p, n_events; n_obs = n)
    end
    error("unsupported prior $name")
end

function main()
    args = parse_cli_args()
    outpath = abspath(arg(args, "output", joinpath("results", "diagnosis", "tau_scale_patch_check.md")))
    n = arg(args, "n", 200)
    p = arg(args, "p", 50)
    nonzero = arg(args, "nonzero", 6)
    baseline = Symbol(arg(args, "baseline", "weibull"))
    seed = arg(args, "seed", 1)
    mkpath(dirname(outpath))

    sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed)
    priors = ["hs", "rhs", "new-rhs", "est-rhs", "cox-hs", "cox-rhs"]
    scales = [1.0, 2.0, 5.0]
    rows = NamedTuple[]
    for prior in priors
        base = resolved_scale(prior, 1.0, sim; n = n, p = p, nonzero = nonzero)
        for scale in scales
            value = resolved_scale(prior, scale, sim; n = n, p = p, nonzero = nonzero)
            push!(rows, (
                prior = prior,
                tau_scale = scale,
                resolved_tau_scale = value,
                ratio_to_scale1 = value / base,
                expected_ratio = scale,
                pass = isapprox(value / base, scale; rtol = 1e-10, atol = 1e-10),
            ))
        end
    end

    open(outpath, "w") do io
        println(io, "# Tau-scale Patch Check")
        println(io)
        println(io, "Generated: $(now())")
        println(io)
        println(io, "Simulation context: n=$n, p=$p, nonzero=$nonzero, baseline=$baseline, seed=$seed, n_events=$(count(!=(0), sim.event)).")
        println(io)
        println(io, "| prior | tau_scale | resolved_tau_scale | ratio_to_scale1 | expected_ratio | pass |")
        println(io, "|---|---:|---:|---:|---:|---|")
        for row in rows
            println(
                io,
                "| $(row.prior) | $(row.tau_scale) | $(round(row.resolved_tau_scale; sigdigits=8)) | " *
                "$(round(row.ratio_to_scale1; sigdigits=8)) | $(row.expected_ratio) | $(row.pass) |",
            )
        end
        println(io)
        all_pass = all(row.pass for row in rows)
        println(io, all_pass ? "Result: all checked priors scale linearly with tau_scale." :
            "Result: at least one prior did not scale linearly with tau_scale.")
    end
    println("Tau-scale patch check written to $outpath")
end

main()
