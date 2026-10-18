#!/usr/bin/env julia
# Backfill nested evaluation for existing fits. Historical fixed-config CV is
# deliberately not reused as an outer-test estimate.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")))
isdefined(@__MODULE__, :parse_cli_args) || include(joinpath(@__DIR__, "..", "analysis_common.jl"))

include(joinpath(@__DIR__, "replay_common.jl"))

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const DEFAULT_BAYES_SUMMARY = joinpath(ROOT, "results", "summary", "realdata", "bayes_all_methods", "bayes_all_methods.csv")
const DEFAULT_OUTDIR = joinpath(ROOT, "results", "summary", "realdata", "bayes_cv_fallback")

function nested_replay_kwargs(meta; samples, warmup, chains, seed)
    kwargs = metadata_kwargs(meta; samples, warmup, chains, seed)
    # Recompute automatic screening from each training subset.
    if lowercase(string(get(meta, "p0_requested", "auto"))) == "auto"
        kwargs[:p0] = nothing
    end
    return kwargs
end

function realdata_cv_replay_main()
    opts = parse_args(ARGS)
    outdir = normpath(opt(opts, "outdir", DEFAULT_OUTDIR))
    samples = parse(Int, opt(opts, "samples", "400"))
    warmup = parse(Int, opt(opts, "warmup", "400"))
    chains = parse(Int, opt(opts, "chains", "2"))
    inner_folds = parse(Int, opt(opts, "cv-folds", "5"))
    outer_folds = parse(Int, opt(opts, "outer-cv-folds", "5"))
    seed = parse(Int, opt(opts, "seed", "20260618"))
    limit = parse(Int, opt(opts, "limit", "0"))
    force = lowercase(opt(opts, "force", "false")) in ("1", "true", "yes", "y")
    priors = split(opt(opts, "priors", ""), ',')
    mkpath(outdir)
    summary_path = joinpath(outdir, "cv_fallback_summary.csv")
    existing = isfile(summary_path) ? CSV.read(summary_path, DataFrame) : DataFrame()
    if !all(c -> c in propertynames(existing), (:evaluation, :status))
        existing = DataFrame()
    else
        existing = existing[string.(existing.evaluation) .== "nested_cv", :]
    end
    bayes = CSV.read(normpath(opt(opts, "bayes-summary", DEFAULT_BAYES_SUMMARY)), DataFrame)
    targets = if force || !(:cv_evaluation in propertynames(bayes))
        bayes
    else
        bayes[(string.(bayes.cv_evaluation) .!= "nested_cv") .| ismissing.(bayes.cv_cindex), :]
    end
    isempty(first(priors)) || (targets = targets[in.(string.(targets.prior), Ref(Set(priors))), :])
    limit > 0 && (targets = first(targets, min(limit, nrow(targets))))
    for row in eachrow(targets)
        run_dir = string(row.run_dir)
        meta = read_json(joinpath(run_dir, "metadata.json"))
        X, time, event = realdata_arrays(meta)
        kwargs = nested_replay_kwargs(meta; samples, warmup, chains, seed)
        prior = string(row.prior)
        # Inner candidate grids override all previously searched prior parameters.
        # Fixed sampler/model choices come from the original run.
        destination = joinpath(outdir, string(row.constraint), string(row.dataset_id))
        force && (destination = mktempdir(mkpath(destination); prefix = "rerun_", cleanup = false))
        result = evaluate_nested_cindex(X, time, event;
            priors = [prior], base_sampler_kwargs = kwargs,
            base_prior_kwargs = Dict(), base_car_kwargs = Dict(),
            base_model_kwargs = Dict(:p0 => kwargs[:p0], :constraint => kwargs[:constraint]),
            outdir = destination, dataset_id = string(row.dataset_id),
            outer_folds, inner_folds, seed)
        summary = copy(result.summary)
        summary.dataset_name = fill(row.dataset_name, nrow(summary))
        summary.dataset_group = fill(row.dataset_group, nrow(summary))
        summary.run_dir = fill(run_dir, nrow(summary))
        summary.source = fill("nested_cv", nrow(summary))
        summary.cv_samples = fill(samples, nrow(summary))
        summary.cv_warmup = fill(warmup, nrow(summary))
        summary.cv_chains = fill(chains, nrow(summary))
        if nrow(existing) > 0
            replace_rows = (string.(existing.dataset_id) .== string(row.dataset_id)) .&
                (string.(existing.constraint) .== string(row.constraint)) .&
                (string.(existing.prior) .== prior)
            existing = existing[.!replace_rows, :]
        end
        existing = vcat(existing, summary; cols = :union)
        write_table(summary_path, existing)
    end
    println(summary_path)
end

if normpath(abspath(PROGRAM_FILE)) == normpath(abspath(@__FILE__))
    realdata_cv_replay_main()
end
