#!/usr/bin/env julia
# Recompute held-out C-index, coverage and empirical FDP from saved summaries.
# No posterior fitting; original outputs remain read-only.
include(joinpath(@__DIR__, "analysis_common.jl"))

function reevaluate_saved_metrics(source)
    meta = JSON.parse(read(joinpath(source, "metadata.json"), String))
    get(meta, "scenario", "baseline") == "baseline" || error("this historical replay supports baseline DGP only")
    n, p, nz, seed = Int(meta["n"]), Int(meta["p"]), Int(meta["nonzero_count"]), Int(meta["seed"])
    baseline = Symbol(meta["baseline"])
    dgp = (baseline_rate = get(meta, "baseline_rate", nothing),
        baseline_shape = baseline == :exponential ? nothing : get(meta, "baseline_shape", nothing))
    train = generate_simulation_data(n, p, nz; seed, baseline, dgp...)
    test = generate_simulation_test_data(n, p, nz; seed, baseline, dgp...)
    get(meta, "censoring_upper", train.censoring_upper) == train.censoring_upper ||
        error("historical censoring upper differs from replay DGP")
    raw_summary = CSV.read(joinpath(source, "posterior_summary.csv"), DataFrame)
    has_map = :map in propertynames(raw_summary)
    summary = ensure_simulation_summary(raw_summary, train.beta)
    # Legacy helper fills a missing MAP with the median; never report it as MAP here.
    !has_map && (summary.map .= NaN)
    if get(meta, "standardize", false) && get(meta, "posterior_beta_scale", "model") == "model"
        error("standardized legacy summary requires explicit scale recovery")
    end
    pc = Float64(get(meta, "pseudocount", 0.0))
    train_data = prepare_cox_data(train.X, train.time, train.event; pseudocount = pc,
        standardize = false, tie_method = Symbol(get(meta, "tie_method", "efron")))
    ztest = _validation_design_matrix(train.X, test.X, train_data; requested_pseudocount = pc)
    rows = DataFrame(estimate = ["mean", "median", "MAP"],
        Cindex_train = Union{Missing,Float64}[missing, missing, missing],
        Cindex_test = Union{Missing,Float64}[missing, missing, missing],
        MSE = Union{Missing,Float64}[missing, missing, missing])
    for (i, col) in enumerate((:mean, :median, :map))
        col in propertynames(summary) || continue
        beta = summary[!, col]
        all(x -> x isa Real && isfinite(x), beta) || continue
        rows.Cindex_train[i] = concordance_index(train.time, train.event, train_data.z * beta)
        rows.Cindex_test[i] = concordance_index(test.time, test.event, ztest * beta)
        rows.MSE[i] = mean((beta - train.beta).^2)
    end
    add_simulation_coverage!(rows, summary; beta = train.beta)
    intervals_valid = all(eachindex(summary.lower)) do j
        lo, hi = summary.lower[j], summary.upper[j]
        lo isa Real && hi isa Real && isfinite(lo) && isfinite(hi) && lo <= hi
    end
    fd = intervals_valid ? false_discovery(train.beta, summary) : nothing
    rows.empirical_FDP = fill(fd === nothing ? missing : fd.FDR, 3)
    for k in (:FP, :TP, :FN, :TN)
        rows[!, k] = fill(fd === nothing ? missing : fd[k], 3)
    end
    for (k,v) in pairs((source = abspath(source), method = get(meta, "method_name", get(meta, "prior_name", get(meta, "prior", "unknown"))),
        n = n, p = p, nonzero = nz, baseline = string(baseline), seed = seed,
        replicate = get(meta, "replicate_id", seed), constraint = get(meta, "constraint", "unknown"),
        p0 = get(meta, "p0", missing), n_test = 1000, test_seed = test.test_seed,
        interval_rule = "marginal equal-tail 95%; selected iff zero excluded; FDP=FP/max(TP+FP,1)"))
        rows[!, k] = fill(v, 3)
    end
    return rows
end

function reevaluate_metrics_main()
    args = parse_cli_args()
    inventory = CSV.read(arg(args, "inventory", ""), DataFrame)
    outdir = ensure_dir(abspath(arg(args, "outdir", "results/revision_20261011/historical_metrics")))
    maxruns = arg(args, "max-runs", typemax(Int))
    all_rows, failures = DataFrame[], NamedTuple[]
    for (i, row) in enumerate(eachrow(inventory))
        i > maxruns && break
        row.status == "complete" || continue
        source = String(row.source)
        sourceid = bytes2hex(sha256(source))[1:20]
        dest = ensure_dir(joinpath(outdir, "runs", sourceid))
        mp, sp = joinpath(source, "metadata.json"), joinpath(source, "posterior_summary.csv")
        fingerprint = Dict("metric_schema" => 1, "source" => source,
            "metadata_sha256" => bytes2hex(sha256(read(mp))), "summary_sha256" => bytes2hex(sha256(read(sp))))
        manifest, output = joinpath(dest, "manifest.json"), joinpath(dest, "metrics.csv")
        try
            reusable = isfile(manifest) && isfile(output) && JSON.parse(read(manifest, String)) == fingerprint
            table = if reusable
                CSV.read(output, DataFrame)
            else
                r = reevaluate_saved_metrics(source)
                write_table(output, r)
                _loo_atomic_json(manifest, fingerprint)
                r
            end
            push!(all_rows, table)
        catch err
            push!(failures, (source = source, error = sprint(showerror, err)))
        end
        i % 50 == 0 && (println("Reevaluated $i saved summaries; failures=$(length(failures))"); flush(stdout))
    end
    !isempty(all_rows) && write_table(joinpath(outdir, "metrics_by_run.csv"), vcat(all_rows...; cols = :union))
    write_table(joinpath(outdir, "failures.csv"), isempty(failures) ?
        DataFrame(source = String[], error = String[]) : DataFrame(failures))
    isempty(failures) || error("Some saved metrics could not be recomputed; inspect failures.csv")
    println("Finished $(length(all_rows)) runs without posterior resampling.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && reevaluate_metrics_main()
