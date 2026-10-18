# Loaded by tune_hyperparams_cindex.jl. All selection is confined to outer training data.

function _cindex_candidate_configs(prior, sampler_kwargs;
    tune_priorparams, tune_algparams, prior_grids)
    prior_grid = tune_priorparams ? get(prior_grids, String(prior), nothing) : [Dict{Symbol,Any}()]
    prior_grid === nothing && throw(ArgumentError("no tuning grid for $prior"))
    algorithms = tune_algparams ? _algorithm_grid(sampler_kwargs) : [Dict{Symbol,Any}()]
    configs = [_merge_tuning_kwargs(p, a) for p in prior_grid for a in algorithms]
    isempty(configs) && throw(ArgumentError("empty tuning grid for $prior"))
    return configs
end

function _cindex_fit_kwargs(prior, sampler, prior_kwargs, car, model, config, p, seed;
    clinical_columns = 0)
    kwargs = _actual_fit_kwargs(prior, sampler, prior_kwargs, car, model, config;
        p, realdata = true, clinical_columns)
    kwargs[:seed] = seed
    kwargs[:compute_metrics] = false
    # Explicit p0 remains fixed; auto p0 is recalculated by each training fit.
    fixed_p0 = _kwget(model, :p0, nothing)
    fixed_p0 !== nothing && get(config, :p0, nothing) === nothing && (kwargs[:p0] = fixed_p0)
    return kwargs
end

function _select_inner_cindex(X, time, event; prior, configs, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, cv_folds, seed,
    fit_function, score_function, clinical_X = nothing, checkpoint_dir = nothing,
    config_indices = eachindex(configs))
    folds = _kfold_indices(event; k = cv_folds, seed)
    checkpoint_dir === nothing || _loo_checkpoint_manifest(
        clinical_X === nothing ? X : hcat(X, clinical_X), time, event,
        Dict(:prior => prior, :configs => configs, :sampler => base_sampler_kwargs,
            :prior_kwargs => base_prior_kwargs, :car => base_car_kwargs, :model => base_model_kwargs,
            :folds => cv_folds, :seed => seed), checkpoint_dir, true; mode = "inner_cv")
    rows = Dict{Symbol,Any}[]
    scores = fill(-Inf, length(configs))
    for (config_index, config) in enumerate(configs)
        config_index in config_indices || continue
        values = Float64[]
        for (fold, val_idx) in enumerate(folds)
            train_idx = setdiff(eachindex(time), val_idx)
            row = Dict{Symbol,Any}(:config_index => config_index, :fold => fold,
                :prior => prior, :n_train => length(train_idx), :n_validation => length(val_idx),
                :status => "failed", :validation_Cindex => missing, :error_message => "")
            _add_config_fields!(row, config)
            checkpoint = checkpoint_dir === nothing ? nothing :
                joinpath(checkpoint_dir, "config_$(lpad(config_index, 4, '0'))_fold_$(fold).json")
            if checkpoint !== nothing && isfile(checkpoint)
                saved = JSON.parse(read(checkpoint, String))
                if get(saved, "status", "") == "success"
                    saved_score = _finite_number(get(saved, "validation_Cindex", nothing))
                    if saved_score !== nothing
                        push!(rows, Dict{Symbol,Any}(Symbol(k) => v === nothing ? missing : v for (k,v) in saved))
                        push!(values, saved_score)
                        continue
                    end
                end
            end
            try
                kwargs = _cindex_fit_kwargs(prior, base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, config, size(X, 2),
                    seed + 10_000 * config_index + fold;
                    clinical_columns = clinical_X === nothing ? 0 : size(clinical_X, 2))
                clinical_X !== nothing && (kwargs[:clinical_X] = clinical_X[train_idx, :])
                fit = fit_function(X[train_idx, :], time[train_idx], event[train_idx]; kwargs...)
                if hasproperty(fit, :result) && fit.result isa CoxSamplerResult
                    for (key, value) in pairs(_fit_diagnostics(fit.result))
                        row[Symbol("fit_", key)] = value
                    end
                end
                clinical_score_kwargs = clinical_X === nothing ? NamedTuple() : (clinical_X_val = clinical_X[val_idx, :],)
                vc = score_function(fit, X[train_idx, :], X[val_idx, :], time[val_idx], event[val_idx];
                    clinical_score_kwargs...)
                c = _finite_number(get(vc, "mean", missing))
                c === nothing && error("inner validation fold has no finite C-index")
                row[:status] = "success"
                row[:validation_Cindex] = c
                row[:validation_Cindex_median] = get(vc, "median", missing)
                row[:validation_Cindex_MAP] = get(vc, "MAP", missing)
                push!(values, c)
            catch err
                err isa InterruptException && rethrow()
                row[:error_message] = sprint(showerror, err)
            end
            push!(rows, row)
            checkpoint === nothing || _loo_atomic_json(checkpoint, _json_ready(row))
        end
        # Never select a configuration on a convenient subset of successful folds.
        length(values) == cv_folds && (scores[config_index] = mean(values))
    end
    best = argmax(scores) # deterministic grid order breaks ties
    return (config = isfinite(scores[best]) ? configs[best] : nothing,
        config_index = best, score = scores[best], metrics = _dict_rows_dataframe(rows), folds = folds)
end

# The final full-data model uses the same selection rule as an outer training fit.
# These scores are saved for tuning diagnostics only, never as test performance.
function _tune_realdata_full_cindex(X, time, event; priors, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, outdir, dataset_id,
    cv_folds, seed, tune_priorparams, tune_algparams, simulation_metadata, nested)
    tuning_dir = ensure_dir(joinpath(outdir, "tuning"))
    best_priors = Dict{String,Any}()
    best_algorithms = Dict{String,Any}()
    summaries = Dict{Symbol,Any}[]
    metrics = DataFrame()
    algorithm_keys = tune_algparams ? Set(k for cfg in _algorithm_grid(base_sampler_kwargs) for k in keys(cfg)) : Set{Symbol}()
    if tune_priorparams || tune_algparams
        for prior in String.(priors)
            configs = _cindex_candidate_configs(prior, base_sampler_kwargs;
                tune_priorparams, tune_algparams, prior_grids = PRIOR_PARAM_GRIDS)
            selected = _select_inner_cindex(X, time, event; prior, configs,
                base_sampler_kwargs, base_prior_kwargs, base_car_kwargs, base_model_kwargs,
                cv_folds, seed, fit_function = fit_and_evaluate, score_function = _validation_cindices)
            append!(metrics, selected.metrics; cols = :union)
            for (index, cfg) in enumerate(configs)
                sub = selected.metrics[selected.metrics.config_index .== index, :]
                ok = nrow(sub) == cv_folds && all(==("success"), sub.status)
                row = Dict{Symbol,Any}(:prior => prior, :config_index => index,
                    :prior_config_index => index, :dataset_id => dataset_id,
                    :mode => "realdata", :stage => "prior", :evaluation => "inner_tuning",
                    :status => ok ? "success" : "failed", :n_folds => cv_folds,
                    :n_successful_folds => count(==("success"), sub.status))
                merge!(row, _metadata_columns(simulation_metadata))
                for (col, mean_key, se_key) in ((:validation_Cindex, :CV_Cindex_mean, :CV_Cindex_se),
                    (:validation_Cindex_median, :CV_Cindex_median_mean, :CV_Cindex_median_se),
                    (:validation_Cindex_MAP, :CV_Cindex_MAP_mean, :CV_Cindex_MAP_se))
                    vals = col in propertynames(sub) ? filter(isfinite, Float64.(collect(skipmissing(sub[!, col])))) : Float64[]
                    valid = ok && length(vals) == cv_folds
                    row[mean_key] = valid ? mean(vals) : missing
                    row[se_key] = valid ? std(vals) / sqrt(cv_folds) : missing
                    col == :validation_Cindex && (row[:CV_Cindex_sd] = valid ? std(vals) : missing)
                end
                _add_config_fields!(row, cfg)
                push!(summaries, row)
            end
            write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_metrics), metrics)
            write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.prior_summary), _dict_rows_dataframe(summaries))
            selected.config === nothing && error("no complete full-data tuning configuration for $prior")
            best_priors[prior] = Dict(String(k) => v for (k, v) in selected.config if k != :p0 && !(k in algorithm_keys))
            best_algorithms[prior] = Dict("alg_" * String(k) => v for (k, v) in selected.config if k in algorithm_keys)
        end
    end
    best = Dict{String,Any}("dataset_id" => dataset_id, "mode" => "realdata",
        "cv_folds" => cv_folds, "seed" => seed, "evaluation" => "inner_tuning",
        "tune_priorparams" => tune_priorparams, "tune_algparams" => tune_algparams,
        "best_algorithm_configs" => best_algorithms, "best_algorithm_config" => nothing)
    _write_tuning_json(joinpath(tuning_dir, "best_prior_configs.json"), best_priors)
    _write_tuning_json(joinpath(tuning_dir, TUNING_OUTPUT_FILES.best_config), best)
    summary = _dict_rows_dataframe(summaries)
    nrow(summary) > 0 && write_table(joinpath(tuning_dir, TUNING_OUTPUT_FILES.all_results), summary)
    return (tuning_dir = tuning_dir, prior = (metrics = metrics, summary = summary),
        algorithm = nothing, best = best, best_prior_configs = best_priors, nested = nested)
end

function _nested_cindex_summary(rows, priors; dataset_id, constraint, outer_folds, inner_folds)
    summaries = Dict{Symbol,Any}[]
    for prior in priors
        local_rows = [r for r in rows if r[:prior] == prior]
        n_success = count(r -> r[:status] == "success", local_rows)
        complete = n_success == outer_folds
        row = Dict{Symbol,Any}(:dataset_id => dataset_id, :constraint => string(constraint),
            :prior => prior, :evaluation => "nested_cv", :n_folds => outer_folds,
            :inner_folds => inner_folds, :n_successful_folds => n_success,
            :status => complete ? "success" : "incomplete")
        for (estimate, mean_key, sd_key, se_key) in (
            ("mean", :CV_Cindex_mean, :CV_Cindex_sd, :CV_Cindex_se),
            ("median", :CV_Cindex_median_mean, :CV_Cindex_median_sd, :CV_Cindex_median_se),
            ("MAP", :CV_Cindex_MAP_mean, :CV_Cindex_MAP_sd, :CV_Cindex_MAP_se))
            vals = _finite_values(local_rows, Symbol("test_Cindex_", estimate))
            valid = complete && length(vals) == outer_folds
            row[mean_key] = valid ? mean(vals) : missing
            row[sd_key] = valid ? std(vals) : missing
            row[se_key] = valid ? std(vals) / sqrt(outer_folds) : missing
        end
        push!(summaries, row)
    end
    return _dict_rows_dataframe(summaries)
end

function _write_nested_table(path, table, priors)
    # Match the full-fit tables: rerunning one method keeps other methods' rows.
    if isfile(path)
        old = CSV.read(path, DataFrame)
        if :prior in propertynames(old)
            keep = [!(string(p) in priors) for p in old.prior]
            table = vcat(old[keep, :], table; cols = :union)
        end
    end
    return write_table(path, table)
end

"""
Nested C-index evaluation: select hyperparameters on inner folds, refit on the
entire outer training fold, then score its untouched outer test fold. Report an
unweighted mean of within-test-fold concordances, never pooled cross-model pairs.
Completed outer folds are resumable; incomplete evaluations have missing scores.
Input columns and candidate grids must be chosen without using outer outcomes.
"""
function evaluate_nested_cindex(X, time, event; priors, base_sampler_kwargs,
    base_prior_kwargs, base_car_kwargs, base_model_kwargs, outdir,
    dataset_id::AbstractString, outer_folds::Integer = 5, inner_folds::Integer = 5,
    seed::Integer = 1, tune_priorparams::Bool = true, tune_algparams::Bool = false,
    prior_grids = PRIOR_PARAM_GRIDS, fit_function = fit_and_evaluate,
    score_function = _validation_cindices, clinical_X = nothing)
    n = length(time)
    size(X, 1) == n == length(event) || throw(DimensionMismatch("inconsistent data lengths"))
    clinical_X !== nothing && size(clinical_X, 1) != n && throw(DimensionMismatch("clinical row count"))
    2 <= outer_folds <= n || throw(ArgumentError("outer folds must be between 2 and n"))
    outer = _kfold_indices(event; k = outer_folds, seed)
    all(idx -> length(idx) >= 2 && n - length(idx) >= inner_folds >= 2, outer) ||
        throw(ArgumentError("not enough subjects for the requested inner/outer folds"))
    priors = String.(priors)
    configs = Dict(prior => _cindex_candidate_configs(prior, base_sampler_kwargs;
        tune_priorparams, tune_algparams, prior_grids) for prior in priors)
    settings = Dict(:sampler => _tuning_dict(base_sampler_kwargs),
        :prior => _tuning_dict(base_prior_kwargs), :car => _tuning_dict(base_car_kwargs),
        :model => _tuning_dict(base_model_kwargs), :configs => configs,
        :outer_folds => outer_folds, :inner_folds => inner_folds, :seed => seed,
        :dataset_id => dataset_id, :evaluation => "nested_cv_v1")
    clinical_X !== nothing && (settings[:clinical_columns] = size(clinical_X, 2))
    # Reuse the data/settings/code fingerprint machinery, with separate checkpoints
    # for each prior so adding another method does not invalidate completed methods.
    root = ensure_dir(joinpath(outdir, "nested_cv"))
    rows = Dict{Symbol,Any}[]
    constraint = _kwget(base_model_kwargs, :constraint, :soft_center)
    for prior in priors
        prior_dir = joinpath(root, prior)
        prior_settings = merge(settings, Dict(:configs => configs[prior], :prior_name => prior))
        fingerprint_X = clinical_X === nothing ? X : hcat(X, clinical_X)
        fingerprint = _loo_checkpoint_manifest(fingerprint_X, time, event, prior_settings, prior_dir, true; mode = "nested_cv")
        for (fold, test_idx) in enumerate(outer)
            train_idx = setdiff(eachindex(time), test_idx)
            checkpoint = joinpath(prior_dir, "outer_$(lpad(fold, 2, '0')).json")
            if isfile(checkpoint)
                saved = JSON.parsefile(checkpoint; use_mmap = false)
                if get(saved, "fingerprint", "") == fingerprint && get(saved, "status", "") == "success"
                    push!(rows, Dict{Symbol,Any}(Symbol(k) => v === nothing ? missing : v for (k, v) in saved))
                    continue
                end
            end
            @info "Nested C-index outer fold" dataset_id prior fold outer_folds
            row = Dict{Symbol,Any}(:dataset_id => dataset_id, :constraint => string(constraint),
                :prior => prior, :outer_fold => fold, :n_train => length(train_idx),
                :n_test => length(test_idx), :evaluation => "nested_cv", :status => "failed",
                :fingerprint => fingerprint, :error_message => "",
                :test_Cindex_mean => missing, :test_Cindex_median => missing, :test_Cindex_MAP => missing)
            fold_dir = ensure_dir(joinpath(prior_dir, "outer_$(lpad(fold, 2, '0'))"))
            try
                inner = _select_inner_cindex(X[train_idx, :], time[train_idx], event[train_idx];
                    prior, configs = configs[prior], base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, cv_folds = inner_folds,
                    seed = seed + 1_000_000 * fold, fit_function, score_function,
                    clinical_X = clinical_X === nothing ? nothing : clinical_X[train_idx, :],
                    checkpoint_dir = joinpath(fold_dir, "inner_checkpoints"))
                write_table(joinpath(fold_dir, "inner_cv_metrics.csv"), inner.metrics)
                _write_tuning_json(joinpath(fold_dir, "split_indices.json"), Dict(
                    "outer_train" => train_idx, "outer_test" => test_idx,
                    "inner_validation" => [train_idx[idx] for idx in inner.folds]))
                inner.config === nothing && error("no configuration succeeded in every inner fold")
                row[:inner_Cindex_mean] = inner.score
                row[:selected_config_index] = inner.config_index
                row[:selected_config] = JSON.json(_json_ready(inner.config))
                _write_tuning_json(joinpath(fold_dir, "selected_config.json"), inner.config)
                kwargs = _cindex_fit_kwargs(prior, base_sampler_kwargs, base_prior_kwargs,
                    base_car_kwargs, base_model_kwargs, inner.config, size(X, 2),
                    seed + 1_000_000 * fold + 999_999;
                    clinical_columns = clinical_X === nothing ? 0 : size(clinical_X, 2))
                clinical_X !== nothing && (kwargs[:clinical_X] = clinical_X[train_idx, :])
                fit = fit_function(X[train_idx, :], time[train_idx], event[train_idx]; kwargs...)
                if hasproperty(fit, :result) && fit.result isa CoxSamplerResult
                    for (key, value) in pairs(_fit_diagnostics(fit.result))
                        row[Symbol("fit_", key)] = value
                    end
                    # Keep the outer posterior so later metrics/diagnostics do not
                    # require fitting it again. Outer outcomes never enter the fit.
                    save_run_outputs(fit.result; output_dir = joinpath(fold_dir, "fit"),
                        run_id = "outer_$(lpad(fold, 2, '0'))_$(prior)",
                        metadata = Dict("dataset_name" => dataset_id, "method_name" => prior,
                            "evaluation" => "nested_cv_outer_training", "outer_fold" => fold,
                            "n" => length(train_idx), "p" => size(X, 2), "fingerprint" => fingerprint,
                            "selected_config" => _json_ready(inner.config)),
                        posterior_summary = fit.summary, save_jld2 = false, save_wide_csv = true)
                end
                clinical_score_kwargs = clinical_X === nothing ? NamedTuple() : (clinical_X_val = clinical_X[test_idx, :],)
                vc = score_function(fit, X[train_idx, :], X[test_idx, :], time[test_idx], event[test_idx];
                    clinical_score_kwargs...)
                for estimate in ("mean", "median", "MAP")
                    c = _finite_number(get(vc, estimate, missing))
                    c === nothing && error("outer test fold has no finite $estimate C-index")
                    row[Symbol("test_Cindex_", estimate)] = c
                end
                row[:status] = "success"
            catch err
                err isa InterruptException && rethrow()
                row[:error_message] = sprint(showerror, err)
                @warn "Nested C-index fold failed; resume to retry" prior fold error=row[:error_message]
            end
            _loo_atomic_json(checkpoint, _json_ready(row))
            push!(rows, row)
            _write_nested_table(joinpath(root, "nested_cv_folds.csv"), _dict_rows_dataframe(rows), priors)
            _write_nested_table(joinpath(root, "nested_cv_summary.csv"), _nested_cindex_summary(rows, priors;
                dataset_id, constraint, outer_folds, inner_folds), priors)
        end
    end
    summary = _nested_cindex_summary(rows, priors; dataset_id, constraint, outer_folds, inner_folds)
    folds = _dict_rows_dataframe(rows)
    _write_nested_table(joinpath(root, "nested_cv_folds.csv"), folds, priors)
    _write_nested_table(joinpath(root, "nested_cv_summary.csv"), summary, priors)
    return (summary = summary, folds = folds)
end
