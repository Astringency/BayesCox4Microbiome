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

function censoring_summary_row(sim; rep, n, p, nonzero, baseline, seed)
    n_obs = length(sim.event)
    events = count(!=(0), sim.event)
    censored = n_obs - events
    time = Float64.(sim.time)
    event_time = Float64.(sim.event_time)
    censor_time = Float64.(sim.censor_time)
    return (
        rep = rep,
        seed = seed,
        n = n,
        p = p,
        nonzero = nonzero,
        baseline = string(baseline),
        baseline_rate = Float64(sim.baseline_rate),
        baseline_scale = Float64(sim.baseline_scale),
        baseline_shape = sim.baseline_shape === nothing ? missing : Float64(sim.baseline_shape),
        censoring_distribution = string(sim.censoring_distribution),
        censoring_upper = Float64(sim.censoring_upper),
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
        min_event_time = minimum(event_time),
        median_event_time = median(event_time),
        max_event_time = maximum(event_time),
        min_censor_time = minimum(censor_time),
        median_censor_time = median(censor_time),
        max_censor_time = maximum(censor_time),
    )
end

function simulation_dataset_table(sim)
    p = size(sim.X, 2)
    taxa_cols = Symbol.("x" .* string.(1:p))
    df = DataFrame(sim.X, taxa_cols)
    df.event_time = sim.event_time
    df.censor_time = sim.censor_time
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
    require_known_args!(
        args,
        vcat(
            (
                "outdir",
                "reps",
                "n",
                "p",
                "nonzero",
                "baseline",
                "seed",
                "save-data",
            ),
            BASELINE_PARAMETER_CLI_KEYS,
            SIMULATION_SCENARIO_CLI_KEYS,
        ),
    )
    outdir = arg(args, "outdir", joinpath("results", "simulation_data"))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    seed0 = arg(args, "seed", 1)
    save_data = arg(args, "save-data", false)
    scenario_kwargs = simulation_scenario_parameters(args)

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    datasets_dir = save_data ? ensure_dir(joinpath(outdir, "datasets")) : nothing
    beta_dir = save_data ? ensure_dir(joinpath(outdir, "beta")) : nothing
    rows = DataFrame()

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
            scenario_kwargs...,
        )
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
            );
            cols = :union,
        )

        if save_data
            if sim.scenario == :phylo_bm
                save_planted_phylogeny(joinpath(outdir, "phylogeny", "p$(p)_nz$(nonzero)"),
                    scenario_phylogeny(p, sim.nonzero_index, sim.scenario))
            end
            label = simulation_label(rep = rep, n = n, p = p, nonzero = nonzero, baseline = baseline, seed = seed)
            write_table(joinpath(datasets_dir, "$label.csv"), simulation_dataset_table(sim))
            write_table(joinpath(beta_dir, "beta_$label.csv"), beta_table(sim))
            test_sim = generate_simulation_test_data(n, p, nonzero;
                baseline, seed, baseline_args..., scenario_kwargs...)
            if sim.scenario != :baseline
                save_scenario_observations(sim, datasets_dir, label)
                save_scenario_observations(test_sim, joinpath(outdir, "test_data"), label * "_measurement")
            end
            write_table(joinpath(outdir, "test_data", "$label.csv"), simulation_dataset_table(test_sim))
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

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
