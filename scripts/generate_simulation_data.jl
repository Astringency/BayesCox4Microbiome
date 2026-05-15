#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

function _include_script(path::AbstractString)
    if get(ENV, "USE_REVISE", "false") in ("1", "true", "yes", "y")
        try
            @eval import Revise
            return Base.invokelatest(Revise.includet, path)
        catch err
            @warn "USE_REVISE was requested but Revise.includet failed; falling back to include." exception=(err, catch_backtrace())
        end
    end
    return include(path)
end

_include_script(joinpath(@__DIR__, "analysis_common.jl"))

function simulation_label(; rep, n, p, nonzero, baseline, seed)
    return "rep$(rep)_$(baseline)_n$(n)_p$(p)_nz$(nonzero)_seed$(seed)"
end

function censoring_summary_row(sim; rep, n, p, nonzero, baseline, seed, tmax)
    n_obs = length(sim.event)
    events = count(!=(0), sim.event)
    censored = n_obs - events
    time = Float64.(sim.time)
    return (
        rep = rep,
        seed = seed,
        n = n,
        p = p,
        nonzero = nonzero,
        baseline = string(baseline),
        tmax = Float64(tmax),
        events = events,
        censored = censored,
        event_rate = events / n_obs,
        censoring_rate = censored / n_obs,
        min_time = minimum(time),
        q25_time = quantile(time, 0.25),
        median_time = median(time),
        q75_time = quantile(time, 0.75),
        max_time = maximum(time),
        mean_time = mean(time),
    )
end

function simulation_dataset_table(sim)
    p = size(sim.X, 2)
    taxa_cols = Symbol.("x" .* string.(1:p))
    df = DataFrame(sim.X, taxa_cols)
    df.time = sim.time
    df.event = sim.event
    return df
end

function beta_table(sim)
    p = length(sim.beta)
    return DataFrame(
        variable = "x" .* string.(1:p),
        beta = sim.beta,
        nonzero = sim.beta .!= 0,
    )
end

function main()
    args = parse_cli_args()
    outdir = arg(args, "outdir", joinpath("results", "simulation_data"))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    seed0 = arg(args, "seed", 1)
    tmax = arg(args, "tmax", 100.0)
    save_data = arg(args, "save-data", false)

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    datasets_dir = save_data ? ensure_dir(joinpath(outdir, "datasets")) : nothing
    beta_dir = save_data ? ensure_dir(joinpath(outdir, "beta")) : nothing
    rows = DataFrame()

    for rep in 1:reps, n in ns, p in ps, nonzero in nonzeros, baseline in baselines
        seed = seed0 + rep - 1
        sim = generate_simulation_data(n, p, nonzero; baseline = baseline, seed = seed, tmax = tmax)
        push!(
            rows,
            censoring_summary_row(
                sim;
                rep = rep,
                n = n,
                p = p,
                nonzero = nonzero,
                baseline = baseline,
                seed = seed,
                tmax = tmax,
            );
            cols = :union,
        )

        if save_data
            label = simulation_label(rep = rep, n = n, p = p, nonzero = nonzero, baseline = baseline, seed = seed)
            write_table(joinpath(datasets_dir, "$label.csv"), simulation_dataset_table(sim))
            write_table(joinpath(beta_dir, "beta_$label.csv"), beta_table(sim))
        end
    end

    summary_path = write_table(joinpath(tables_dir, "censoring_summary.csv"), rows)
    show(stdout, rows; allrows = true, allcols = true)
    println()
    println("Censoring summary written to $(abspath(summary_path))")
    if save_data
        println("Generated datasets written to $(abspath(datasets_dir))")
    end
end

main()
