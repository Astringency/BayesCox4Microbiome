#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "..", "..", "Project.toml"))
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

_include_script(joinpath(@__DIR__, "..", "analysis_common.jl"))

function split_floats(s::AbstractString)
    return parse.(Float64, strip.(split(s, ",")))
end

function _finite_std(x::AbstractVector{<:Real})
    length(x) <= 1 && return NaN
    return std(x)
end

function validate_censoring_rate(; n, p, nonzero, baseline, upper, seed0, reps)
    rates = Float64[]
    for rep in 1:reps
        sim = generate_simulation_data(
            n,
            p,
            nonzero;
            baseline = baseline,
            seed = seed0 + rep - 1,
            censoring_distribution = :uniform,
            censoring_upper = upper,
        )
        push!(rates, mean(sim.event .== 0))
    end
    return (
        mean = mean(rates),
        sd = _finite_std(rates),
        min = minimum(rates),
        max = maximum(rates),
    )
end

function main()
    args = parse_cli_args()
    outdir = arg(args, "outdir", joinpath("results", "censoring_calibration"))
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50,150"))
    nonzeros = split_ints(arg(args, "nonzero", "6,12"))
    baselines = split_symbols(arg(args, "baseline", "weibull,exponential,gompertz"))
    targets = split_floats(arg(args, "targets", "0.2,0.3,0.4"))
    seed0 = arg(args, "seed", 20260515)
    calibration_size = arg(args, "calibration-size", 20000)
    validation_reps = arg(args, "validation-reps", 100)
    tol = arg(args, "tol", 1e-4)

    tables_dir = ensure_dir(joinpath(outdir, "tables"))
    rows = DataFrame()

    case_id = 0
    for p in ps, nonzero in nonzeros, baseline in baselines
        case_id += 1
        calibration_seed = seed0 + 100000 * case_id
        rng = MersenneTwister(calibration_seed)
        process = simulation_event_process(rng, calibration_size, p, nonzero; baseline = baseline)
        for target in targets
            upper = calibrate_uniform_censoring_upper(process.event_time, target; tol = tol)
            expected = uniform_censoring_rate(process.event_time, upper)
            for n in ns
                validation_seed = seed0 + 1000000 * case_id + 10000 * round(Int, 100 * target) + n
                validation = validate_censoring_rate(
                    n = n,
                    p = p,
                    nonzero = nonzero,
                    baseline = baseline,
                    upper = upper,
                    seed0 = validation_seed,
                    reps = validation_reps,
                )
                push!(
                    rows,
                    (
                        n = n,
                        p = p,
                        nonzero = nonzero,
                        baseline = string(baseline),
                        target_censoring_rate = target,
                        censoring_distribution = "uniform",
                        censoring_upper = upper,
                        calibration_size = calibration_size,
                        calibration_seed = calibration_seed,
                        calibration_expected_censoring_rate = expected,
                        validation_reps = validation_reps,
                        validation_mean_censoring_rate = validation.mean,
                        validation_sd_censoring_rate = validation.sd,
                        validation_min_censoring_rate = validation.min,
                        validation_max_censoring_rate = validation.max,
                    );
                    cols = :union,
                )
            end
        end
    end

    path = write_table(joinpath(tables_dir, "censoring_calibration.csv"), rows)
    show(stdout, rows; allrows = true, allcols = true)
    println()
    println("Censoring calibration written to $(abspath(path))")
end

if normpath(abspath(PROGRAM_FILE)) == normpath(abspath(@__FILE__))
    main()
end
