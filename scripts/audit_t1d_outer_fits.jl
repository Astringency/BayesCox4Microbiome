#!/usr/bin/env julia
# Only for already completed outer folds whose posterior draws were not retained.
# Reuse the selected configuration and split; never rerun inner tuning. Write a
# separate audit and require exact reproduction of all three cached C-indices.
const AUDIT_CODE_ROOT = get(ENV, "BCX_FROZEN_CODE", dirname(@__DIR__))
include(joinpath(AUDIT_CODE_ROOT, "scripts", "run_t1d_clinical_revision.jl"))

function audit_t1d_outer_main()
    args = parse_cli_args()
    source = abspath(arg(args, "source-arm", ""))
    outdir = ensure_dir(abspath(arg(args, "outdir", "results/check/t1d_outer_audit")))
    input = abspath(arg(args, "input", joinpath(AUDIT_CODE_ROOT, "data", "T1D.csv")))
    spec = JSON.parsefile(joinpath(source, "analysis_specification.json"))
    method, adjusted = spec["method_name"], spec["adjustment"] == "adjusted"
    df = CSV.read(input, DataFrame)
    C, cnames = t1d_clinical_design(df)
    cols = spec["component_cols"]
    X, time, event = Matrix{Float64}(df[:, cols]), Float64.(df.time), Int.(df.status)
    @assert size(X) == (spec["n"], spec["p"])
    sampler = Dict(:sampler => :hmc, :n_samples => spec["n_samples"], :warmup => spec["warmup"],
        :chains => spec["chains"], :multichains => true, :target_accept => 0.9,
        :hmc_n_leapfrog => spec["hmc_n_leapfrog"], :noncentered => true, :progress => false)
    model = Dict{Symbol,Any}(:constraint => :prox, :prox_sum0_lambda => spec["prox_sum0_lambda"],
        :pseudocount => spec["pseudocount"], :standardize => false, :p0 => nothing)
    adjusted && merge!(model, Dict(:clinical_names => cnames, :clinical_prior_df => 3., :clinical_prior_scale => 2.5))
    prior_kw = Dict(:alpha_prior_kind => :invgamma, :alpha_prior_shape => 1., :alpha_prior_scale => size(X, 2))
    car = Dict(:use_car => false, :sample_car_strength => false)
    n_complete = 0
    for fold in 1:spec["outer_folds"]
        label = "outer_$(lpad(fold, 2, '0'))"
        checkpoint = joinpath(source, "nested_cv", method, label * ".json")
        isfile(checkpoint) || continue
        cached = JSON.parsefile(checkpoint)
        cached["status"] == "success" || continue
        local_dir = joinpath(source, "nested_cv", method, label)
        if isfile(joinpath(local_dir, "fit", "samples_wide.csv"))
            println("$label already retains its posterior; no refit")
            continue
        end
        dest = ensure_dir(joinpath(outdir, label))
        manifest = Dict("checkpoint_sha256" => bytes2hex(sha256(read(checkpoint))),
            "input_sha256" => bytes2hex(sha256(read(input))), "source" => source,
            "purpose" => "Recover unavailable outer-fit diagnostics, reusing all inner tuning",
            "code_root" => abspath(AUDIT_CODE_ROOT),
            "source_manifest_sha256" => isfile(joinpath(AUDIT_CODE_ROOT, "source_manifest.json")) ?
                bytes2hex(sha256(read(joinpath(AUDIT_CODE_ROOT, "source_manifest.json")))) : "local-checkout")
        mp, vp = joinpath(dest, "manifest.json"), joinpath(dest, "validation.json")
        if isfile(mp) && isfile(vp) && JSON.parsefile(mp) == manifest
            @assert JSON.parsefile(vp)["reproduces_cached_scores"]
            n_complete += 1
            continue
        end
        split = JSON.parsefile(joinpath(local_dir, "split_indices.json"))
        train, test = Int.(split["outer_train"]), Int.(split["outer_test"])
        @assert isempty(intersect(train, test))
        config = Dict(Symbol(k) => v for (k,v) in JSON.parsefile(joinpath(local_dir, "selected_config.json")))
        kwargs = _cindex_fit_kwargs(method, sampler, prior_kw, car, model, config, size(X,2),
            spec["seed"] + 1_000_000 * fold + 999_999)
        adjusted && (kwargs[:clinical_X] = C[train, :])
        println("Recovering saved-score outer fit: $method $(spec["adjustment"]) $label"); flush(stdout)
        fit = fit_and_evaluate(X[train, :], time[train], event[train]; kwargs...)
        clinical = adjusted ? (clinical_X_val = C[test, :],) : NamedTuple()
        scores = _validation_cindices(fit, X[train, :], X[test, :], time[test], event[test];
            requested_pseudocount = spec["pseudocount"], clinical...)
        errors = Dict(e => abs(scores[e] - cached["test_Cindex_" * e]) for e in ("mean", "median", "MAP"))
        verified = all(v -> isfinite(v) && v < 1e-12, values(errors))
        save_run_outputs(fit.result; output_dir = joinpath(dest, "fit"),
            metadata = Dict("method_name" => method, "dataset_name" => "T1D", "adjustment" => spec["adjustment"],
                "outer_fold" => fold, "selected_config" => config, "n" => length(train), "p" => size(X,2),
                "evaluation" => "outer_diagnostic_replay", "reproduces_cached_scores" => verified),
            posterior_summary = fit.summary, save_jld2 = false, save_wide_csv = true)
        _loo_atomic_json(mp, manifest)
        _loo_atomic_json(vp, Dict("reproduces_cached_scores" => verified, "absolute_cindex_errors" => errors,
            "scores" => scores, "diagnostics" => _fit_diagnostics(fit.result)))
        verified || error("outer replay differs from cached predictions; original scores were not changed")
        n_complete += 1
    end
    println("$n_complete cached outer folds reproduced; no inner tuning repeated")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && audit_t1d_outer_main()
