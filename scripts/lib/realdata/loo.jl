#!/usr/bin/env julia

using Pkg
const T1D_LOO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Base.active_project() == joinpath(T1D_LOO_ROOT, "Project.toml") || Pkg.activate(T1D_LOO_ROOT)
isdefined(@__MODULE__, :parse_cli_args) || include(joinpath(@__DIR__, "..", "..", "analysis_common.jl"))
include(joinpath(@__DIR__, "replay_common.jl"))

function t1d_loo_main(args = ARGS)
    opts = parse_args(args)
    allowed = Set(["help", "loo-mode", "outdir", "methods", "cox-rhs-constraint",
        "prox-sslasso-constraint", "samples", "warmup", "chains", "seed", "parallel",
        "folds", "resume", "p0-policy", "source-root"])
    unknown = setdiff(Set(keys(opts)), allowed)
    isempty(unknown) || throw(ArgumentError("unknown options: $(join(sort!(collect(unknown)), ", "))"))
    if haskey(opts, "help")
        println("""
        T1D Cox-RHS / Prox-SSLASSO LOO (Julia +1.11.7)
          --loo-mode fixed_risk|subject_refit|both    default: fixed_risk
          --methods cox-rhs,prox-sslasso             default: both methods
          --cox-rhs-constraint soft_center|prox      default: soft_center
          --prox-sslasso-constraint prox|soft_center default: prox
          --parallel true|false                     default: false (parallel folds; julia -t N)
          --folds all|1,2,3                         subject_refit only; default: all subjects
          --samples N --warmup N --chains N --seed N overrides saved settings
          --p0-policy training|fixed                default: training
          --resume true|false                       default: true
          --source-root DIR                         default: results/realdata
          --outdir DIR                              default: results/loo/T1D
        Output is separate from historical results. Missing aggregate scores mean incomplete LOO.
        fixed_risk uses event-factor PSIS with the full-data risk sets; no subject refits.
        subject_refit scores partial-likelihood ratios, not survival-time densities.
        WAIC remains a fixed-risk full-fit diagnostic in fixed_risk_metrics.csv.
        """)
        return nothing
    end
    mode_text = get(opts, "loo-mode", "fixed_risk")
    mode_text in ("subject_refit", "fixed_risk", "both") ||
        throw(ArgumentError("invalid --loo-mode"))
    methods = split(get(opts, "methods", "cox-rhs,prox-sslasso"), ',')
    !isempty(methods) && allunique(methods) && all(in(("cox-rhs", "prox-sslasso")), methods) ||
        throw(ArgumentError("--methods must select cox-rhs and/or prox-sslasso"))
    p0_policy = get(opts, "p0-policy", "training")
    p0_policy in ("training", "fixed") || throw(ArgumentError("invalid --p0-policy"))
    parallel = parse(Bool, get(opts, "parallel", "false")) && mode_text != "fixed_risk"
    resume = parse(Bool, get(opts, "resume", "true"))
    parallel && Threads.nthreads() <= 1 && throw(ArgumentError("use julia -t N for parallel folds"))
    BLAS.set_num_threads(1)
    source_root = abspath(get(opts, "source-root", joinpath(T1D_LOO_ROOT, "results", "realdata")))
    outdir = abspath(get(opts, "outdir", joinpath(T1D_LOO_ROOT, "results", "loo", "T1D")))
    # Never overwrite the source analyses, including when --outdir is set manually.
    relative_out = relpath(outdir, source_root)
    (relative_out == "." || (first(splitpath(relative_out)) != ".." && !isabspath(relative_out))) &&
        throw(ArgumentError("--outdir must be outside --source-root"))
    summaries = DataFrame[]
    any_failed = false
    for method in methods
        default_constraint = method == "cox-rhs" ? "soft_center" : "prox"
        constraint = get(opts, "$method-constraint", default_constraint)
        constraint in ("soft_center", "prox") || throw(ArgumentError("invalid constraint"))
        metadata_path = joinpath(source_root, constraint, "T1D", "runs", "realdata_$method", "metadata.json")
        meta = read_json(metadata_path)
        string(meta["prior_name"]) == method || throw(ArgumentError("source prior does not match $method"))
        string(meta["constraint"]) == constraint || throw(ArgumentError("source constraint mismatch"))
        X, time, event = realdata_arrays(meta)
        kwargs = metadata_kwargs(meta;
            samples = parse(Int, get(opts, "samples", string(meta["n_samples"]))),
            warmup = parse(Int, get(opts, "warmup", string(meta["warmup"]))),
            chains = parse(Int, get(opts, "chains", string(meta["chains"]))),
            seed = parse(Int, get(opts, "seed", string(meta["seed"]))),
        )
        # Repeat auto sparsity screening ONLY on training outcomes for each refit.
        if p0_policy == "training" && get(meta, "p0_requested", "auto") == "auto"
            kwargs[:p0] = nothing
        end
        parallel && (kwargs[:multichains] = false)
        selected_text = get(opts, "folds", "all")
        folds = selected_text == "all" ? collect(eachindex(time)) : parse.(Int, split(selected_text, ','))
        model_dir = joinpath(outdir, "$(constraint)_$(method)")
        checkpoint_dir = joinpath(model_dir, "folds")
        if mode_text != "fixed_risk"
            # Fail on stale checkpoints BEFORE overwriting any model-level output.
            _loo_checkpoint_manifest(X, time, event,
                merge(kwargs, Dict(:loo_diagnostics => true)), checkpoint_dir, resume)
        end
        mkpath(model_dir)
        config = Dict("source_metadata" => metadata_path, "method" => method,
            "constraint" => constraint, "loo_mode" => mode_text, "p0_policy" => p0_policy,
            "fit_kwargs" => Dict(string(k) => v for (k, v) in kwargs),
            "julia_version" => string(VERSION), "n" => length(time), "events" => sum(event),
            "hyperparameter_policy" => "saved tuning settings; no nested retuning",
            "score_target" => (mode_text == "fixed_risk" ? "event_attribution_fixed_risk" :
                "subject_deletion_partial_likelihood_ratio for refits; fixed-risk diagnostics separately"))
        _loo_atomic_json(joinpath(model_dir, "settings.json"), config)
        @info "Fitting full T1D data for fixed-risk reference diagnostics" method constraint
        fit = fit_and_evaluate(X, time, event; kwargs...)
        fixed_metrics = copy(fit.metrics)
        fixed_metrics.NLPML = fixed_metrics.LPML
        fixed_metrics.loo_mode .= "fixed_risk"
        fixed_metrics.score_target .= "event_attribution_fixed_risk"
        CSV.write(joinpath(model_dir, "fixed_risk_metrics.csv"), fixed_metrics)
        CSV.write(joinpath(model_dir, "full_fit_posterior_summary.csv"), fit.summary)
        modes = mode_text == "both" ? (:fixed_risk, :subject_refit) : (Symbol(mode_text),)
        for mode in modes
            evaluated = evaluate_cox_loo(X, time, event; mode, fit_kwargs = kwargs,
                full_fit = fit, folds, checkpoint_dir, resume = true, parallel)
            s = evaluated.summary
            s.method .= method
            s.constraint .= constraint
            CSV.write(joinpath(model_dir, "$(mode)_summary.csv"), s)
            if mode == :subject_refit
                CSV.write(joinpath(model_dir, "subject_refit_pointwise.csv"), evaluated.pointwise)
                any_failed |= any(==("failed"), evaluated.pointwise.status)
            end
            push!(summaries, s)
            CSV.write(joinpath(outdir, "loo_summary.csv"), reduce((a, b) -> vcat(a, b; cols = :union), summaries))
            println(select(s, :mode, :method, :constraint, :n_completed, :complete, :LOOIC, :NLPML))
        end
    end
    any_failed && error("Some LOO folds failed; inspect pointwise files and rerun to retry them")
    println("LOO outputs: $outdir")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    t1d_loo_main()
end
