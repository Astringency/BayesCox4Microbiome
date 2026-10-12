#!/usr/bin/env julia
# Joint adjustment, paired nested-CV splits, and reuse of existing full-data fits.
include(joinpath(@__DIR__, "analysis_common.jl"))
include(joinpath(@__DIR__, "tune_hyperparams_cindex.jl"))

function t1d_clinical_design(df)
    all(x -> x in ("F", "M"), df.Sex) || error("Sex must be F/M without missing values")
    all(x -> x in ("Control", "Tylosin"), df.Antibiotics) || error("Antibiotics must be Control/Tylosin without missing values")
    return Float64.(hcat(df.Sex .== "M", df.Antibiotics .== "Tylosin")),
        ["clinical[Sex_M]", "clinical[Antibiotics_Tylosin]"]
end

function clinical_revision_main()
    args = parse_cli_args()
    input = arg(args, "input", joinpath("data", "T1D.csv"))
    root = arg(args, "outdir", joinpath("results", "revision_20261011", "T1D_clinical"))
    method = arg(args, "prior", "cox-rhs")
    method in ("cox-rhs", "prox-sslasso") || error("unsupported prior")
    adjusted = arg(args, "adjusted", true)
    label = adjusted ? "adjusted" : "unadjusted"
    outdir = ensure_dir(joinpath(root, label, method))
    # Real-data sampling retains the existing T1D scale; simulation scale is separate.
    samples, warmup, chains = arg(args, "samples", 800), arg(args, "warmup", 800), arg(args, "chains", 4)
    seed = arg(args, "seed", 1)
    df = CSV.read(input, DataFrame)
    C, cnames = t1d_clinical_design(df)
    cols = filter(c -> !(c in ("Sex", "Antibiotics", "time", "status")), names(df))
    X, time, event = Matrix{Float64}(df[:, cols]), Float64.(df.time), Int.(df.status)
    clinical = adjusted ? C : nothing
    sampler = Dict(:sampler => :hmc, :n_samples => samples, :warmup => warmup, :chains => chains,
        :multichains => true, :target_accept => 0.9, :hmc_n_leapfrog => arg(args, "leapfrog", 100),
        :noncentered => true, :progress => false)
    model = Dict{Symbol,Any}(:constraint => :prox, :prox_sum0_lambda => 1e-6,
        :pseudocount => 1e-6, :standardize => false, :p0 => nothing)
    adjusted && merge!(model, Dict(:clinical_names => cnames,
        :clinical_prior_df => 3.0, :clinical_prior_scale => 2.5))
    prior_kwargs = Dict{Symbol,Any}(:alpha_prior_kind => :invgamma,
        :alpha_prior_shape => 1.0, :alpha_prior_scale => size(X, 2))
    car = Dict(:use_car => false, :sample_car_strength => false)
    grids = deepcopy(PRIOR_PARAM_GRIDS)
    # Use the established grid; do not pick candidates from outer-fold results.
    arg(args, "smoke", false) && (grids[method] = [Dict{String,Any}(
        (method == "cox-rhs" ? "rhs_local_scale" : "sslasso_spike_scale") => 0.5)])
    outer, inner = arg(args, "outer-folds", 5), arg(args, "inner-folds", 5)
    metadata = Dict("dataset_name" => "T1D", "method_name" => method, "prior_name" => method,
        "adjustment" => label, "clinical_coding" => Dict("Sex_M" => "M=1,F=0", "Antibiotics_Tylosin" => "Tylosin=1,Control=0"),
        "clinical_prior" => "independent Student-t(df=3, location=0, scale=2.5); Cox adaptation",
        "clinical_prior_reference" => "https://doi.org/10.1214/08-AOAS191",
        "n" => length(time), "p" => size(X, 2), "n_samples" => samples, "warmup" => warmup,
        "chains" => chains, "constraint" => "prox", "prox_sum0_lambda" => 1e-6,
        "pseudocount" => 1e-6, "standardize" => false, "component_cols" => cols,
        "outer_folds" => outer, "inner_folds" => inner, "seed" => seed,
        "sampler" => "hmc", "hmc_n_leapfrog" => sampler[:hmc_n_leapfrog],
        "note" => "Full-data tuning scores are not test performance. Clinical covariates are outside the taxon constraint.")
    _write_tuning_json(joinpath(outdir, "analysis_specification.json"), metadata)

    # Full-data historical posterior is a descriptive comparator, never a CV fit.
    historical = arg(args, "historical-root", joinpath("results", "realdata", "prox", "T1D"))
    source = joinpath(historical, "runs", "realdata_" * method)
    if !adjusted && method == "cox-rhs" && isfile(joinpath(source, "samples_wide.csv"))
        old = JSON.parse(read(joinpath(source, "metadata.json"), String))
        get(old, "n", 0) == length(time) && get(old, "p", 0) == size(X, 2) || error("historical data dimensions differ")
        old_cols = get(old, "component_cols", [])
        old_cols isa AbstractString && (old_cols = split(old_cols, ','))
        old_cols == cols || error("historical taxon order differs")
        wide = CSV.read(joinpath(source, "samples_wide.csv"), DataFrame)
        beta_cols = ["beta[$j]" for j in axes(X, 2)]
        draws = Matrix{Float64}(wide[:, beta_cols])
        get(old, "standardize", false) == false || error("historical standardized coefficients require replay preprocessing")
        summary = posterior_summary(draws; names = beta_cols)
        summary.variable = cols
        write_table(joinpath(outdir, "coefficients.csv"), summary)
        old_data = prepare_cox_data(X, time, event; pseudocount = Float64(get(old, "pseudocount", 1e-6)))
        write_table(joinpath(outdir, "ph_input.csv"), DataFrame(time = time, status = event,
            Sex_M = C[:, 1], Antibiotics_Tylosin = C[:, 2],
            microbiome_score = old_data.z * vec(mean(draws; dims = 1))))
        BayesCox4Micro.save_revision_diagnostics(wide, ensure_dir(joinpath(outdir, "historical_diagnostics")))
        _write_tuning_json(joinpath(outdir, "reuse_manifest.json"), Dict("source" => abspath(source),
            "purpose" => "unadjusted full-data posterior comparison only; not used as held-out performance",
            "metadata" => old, "resampled" => false,
            "data_provenance" => "Same project T1D input, n/p and exact taxon order matched; historical metadata contains no original file digest"))
    end

    # The same seed gives paired outer/inner subject splits in both adjustment arms.
    cv = evaluate_nested_cindex(X, time, event; priors = [method], base_sampler_kwargs = sampler,
        base_prior_kwargs = prior_kwargs, base_car_kwargs = car, base_model_kwargs = model,
        outdir, dataset_id = "T1D", outer_folds = outer, inner_folds = inner, seed,
        prior_grids = grids, clinical_X = clinical)
    all(==("success"), cv.summary.status) || error("nested CV incomplete; inspect fold errors and resume")
    # Existing full-data comparator needs no refit. Adjusted fits require joint sampling.
    isfile(joinpath(outdir, "reuse_manifest.json")) && return
    configs = _cindex_candidate_configs(method, sampler; tune_priorparams = true,
        tune_algparams = false, prior_grids = grids)
    tuning_dir = ensure_dir(joinpath(outdir, "full_data_tuning"))
    choice_file = joinpath(tuning_dir, "selected_config.json")
    selected_config = if isfile(choice_file)
        Dict(Symbol(k) => v for (k,v) in JSON.parse(read(choice_file, String)))
    else
        selected = _select_inner_cindex(X, time, event; prior = method, configs,
            base_sampler_kwargs = sampler, base_prior_kwargs = prior_kwargs, base_car_kwargs = car,
            base_model_kwargs = model, cv_folds = inner, seed, fit_function = fit_and_evaluate,
            score_function = _validation_cindices, clinical_X = clinical,
            checkpoint_dir = joinpath(tuning_dir, "inner_checkpoints"))
        write_table(joinpath(tuning_dir, "inner_cv_metrics.csv"), selected.metrics)
        selected.config === nothing && error("full-data tuning failed")
        _write_tuning_json(choice_file, selected.config)
        selected.config
    end
    run = joinpath(outdir, "runs", "realdata_" * method)
    isfile(joinpath(run, "metadata.json")) && (!adjusted || isfile(joinpath(outdir, "clinical_coefficients.csv"))) && return
    kwargs = _cindex_fit_kwargs(method, sampler, prior_kwargs, car, model, selected_config, size(X, 2), seed)
    kwargs[:compute_metrics] = true
    clinical !== nothing && (kwargs[:clinical_X] = clinical)
    fit = fit_and_evaluate(X, time, event; kwargs...)
    fit.summary.variable = cols
    write_table(joinpath(outdir, "coefficients.csv"), fit.summary)
    write_table(joinpath(outdir, "training_metrics.csv"), fit.metrics)
    adjusted && write_table(joinpath(outdir, "clinical_coefficients.csv"),
        posterior_summary(posterior_clinical_draws(fit.result); names = cnames))
    save_run_outputs(fit.result; output_dir = run, run_id = "realdata_" * method,
        metadata = merge(metadata, Dict("selected_config" => selected_config, "p0" => fit.p0)),
        posterior_summary = fit.summary, save_jld2 = false, save_wide_csv = true)
    prediction = DataFrame(time = time, status = event, Sex_M = C[:, 1], Antibiotics_Tylosin = C[:, 2],
        microbiome_score = fit.result.data.z * vec(mean(posterior_beta_draws(fit.result); dims = 1)))
    write_table(joinpath(outdir, "ph_input.csv"), prediction)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && clinical_revision_main()
