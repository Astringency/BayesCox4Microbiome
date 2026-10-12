# Included by analysis_common.jl. No fitting or file writes occur on include.
using SHA
using Serialization

function check_loo_mode(mode::Symbol)
    mode in (:fixed_risk, :subject_refit) ||
        throw(ArgumentError("loo mode must be :fixed_risk or :subject_refit"))
    return mode
end

"""
    subject_deletion_logratios(train_data, augmented_data, draws)

Evaluate log L_PL(D; beta) - log L_PL(D_-i; beta) at TRAINING posterior
draws. Both data objects must use the same training-fitted transformation.
This is a cross-validated partial-likelihood ratio score, not a normalized
survival-time predictive density. Censored subjects affect this score too.
Using the total Efron likelihood avoids an arbitrary allocation of tied events.
"""
function subject_deletion_logratios(train_data::CoxPreparedData,
                                  augmented_data::CoxPreparedData,
                                  draws::AbstractMatrix)
    length(augmented_data.time) == length(train_data.time) + 1 ||
        throw(DimensionMismatch("augmented data must add exactly one subject"))
    train_data.tie_method == augmented_data.tie_method ||
        throw(ArgumentError("training and augmented tie methods must match"))
    return [cox_logpartiallikelihood(augmented_data, view(draws, s, :)) -
            cox_logpartiallikelihood(train_data, view(draws, s, :))
            for s in axes(draws, 1)]
end

function _loo_augment(train_data, X_train, X_test, time_test, event_test; requested_pseudocount)
    z_test = _validation_design_matrix(X_train, X_test, train_data; requested_pseudocount)
    time = vcat(train_data.time, Float64(time_test))
    data = CoxPreparedData(
        vcat(train_data.z, z_test), time, vcat(train_data.event, Int(event_test)),
        BayesCox4Micro._build_riskset(time), train_data.tie_method,
        train_data.transform, train_data.pseudocount, train_data.closure,
        train_data.standardize, train_data.col_mean, train_data.col_scale,
    )
    return data
end

function _loo_atomic_json(path, value)
    mkpath(dirname(path))
    tmp = tempname(dirname(path))
    try
        open(tmp, "w") do io
            JSON.print(io, value, 2)
            write(io, '\n')
        end
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp)
    end
    return path
end

function _loo_code_digest()
    root = normpath(joinpath(@__DIR__, ".."))
    files = [joinpath(root, "main.jl"), joinpath(root, "Project.toml")]
    isfile(joinpath(root, "Manifest.toml")) && push!(files, joinpath(root, "Manifest.toml"))
    for subdir in ("mainfuncs", "priors", "scripts")
        for (dir, _, names) in walkdir(joinpath(root, subdir)), name in names
            endswith(name, ".jl") && push!(files, joinpath(dir, name))
        end
    end
    io = IOBuffer()
    for file in sort(files)
        write(io, relpath(file, root), '\0', read(file), '\0')
    end
    return bytes2hex(sha256(take!(io)))
end

function _loo_checkpoint_manifest(X, time, event, fit_kwargs, checkpoint_dir, resume; mode = "subject_refit")
    io = IOBuffer()
    settings = sort!(collect(pairs(fit_kwargs)); by = x -> string(first(x)))
    serialize(io, (Matrix{Float64}(X), Float64.(time), Int.(event), settings,
                   string(VERSION), _loo_code_digest()))
    fingerprint = bytes2hex(sha256(take!(io)))
    checkpoint_dir === nothing && return fingerprint
    path = joinpath(checkpoint_dir, "manifest.json")
    if isfile(path)
        resume || throw(ArgumentError("checkpoint exists; resume it or choose a new directory"))
        get(JSON.parsefile(path; use_mmap = false), "fingerprint", "") == fingerprint ||
            throw(ArgumentError("$mode checkpoint data, settings, Julia version or code changed; use a new directory"))
    else
        if isdir(checkpoint_dir) && !isempty(readdir(checkpoint_dir))
            throw(ArgumentError("nonempty checkpoint directory has no manifest"))
        end
        _loo_atomic_json(path, Dict("fingerprint" => fingerprint, "n" => length(time),
                                   "mode" => mode, "schema_version" => 1))
    end
    return fingerprint
end

function _loo_draw_diagnostics(result, draws)
    layout = posterior_draw_layout(result, size(draws, 1))
    layout.draws >= 4 || return (nothing, nothing)
    arr = _loo_loglik_array(permutedims(draws), layout.chains, layout.draws)
    stats = DataFrame(PosteriorStats.summarize(arr; kind = :diagnostics))
    rhat = Float64.(stats.rhat)
    ess = Float64.(stats.ess_bulk)
    return (all(isfinite, rhat) ? maximum(rhat) : nothing,
            all(isfinite, ess) ? minimum(ess) : nothing)
end

function _loo_refit_fold(X, time, event, i, fit_kwargs, fit_function, diagnostics)
    train = [j for j in eachindex(time) if j != i]
    kwargs = Dict{Symbol,Any}(pairs(fit_kwargs))
    kwargs[:seed] = Int(get(kwargs, :seed, 1)) + i
    kwargs[:compute_metrics] = false
    started = time_ns()
    fit = fit_function(X[train, :], time[train], event[train]; kwargs...)
    data = fit.result.data
    data.time == Float64.(time[train]) && data.event == Int.(event[train]) ||
        throw(ArgumentError("refit must use exactly the training subjects, in input order"))
    draws = posterior_beta_draws(fit.result)
    size(draws, 1) > 0 && all(isfinite, draws) || error("nonfinite or empty posterior draws")
    requested_pc = get(kwargs, :pseudocount, nothing)
    # Match fit_and_evaluate's default, distinguishing additive PC from zero replacement.
    requested_pc === nothing && (requested_pc = get(kwargs, :realdata, false) ? 1e-6 : 0.0)
    augmented = _loo_augment(data, X[train, :], X[i:i, :], time[i], event[i];
                             requested_pseudocount = requested_pc)
    logratios = subject_deletion_logratios(data, augmented, draws)
    all(isfinite, logratios) || error("nonfinite subject-deletion log partial-likelihood ratios")
    max_rhat, min_ess = diagnostics ? _loo_draw_diagnostics(fit.result, draws) : (nothing, nothing)
    return Dict{String,Any}(
        "subject" => i, "event" => Int(event[i]), "n_train" => length(train),
        "events_train" => sum(event[train]), "status" => "success",
        "log_score" => logmeanexp(logratios),
        "mean_log_ratio" => mean(logratios), "posterior_draws" => size(draws, 1),
        "seed" => kwargs[:seed], "p0" => (ismissing(fit.p0) ? nothing : fit.p0),
        "beta_rhat_max" => max_rhat, "beta_ess_bulk_min" => min_ess,
        "elapsed_seconds" => (time_ns() - started) / 1e9,
    )
end

"""
    evaluate_cox_loo(X, time, event; mode=:fixed_risk, fit_kwargs=(;), ...)

`:fixed_risk`: existing event-factor PSIS-LOO, harmonic-mean NLPML, and WAIC.
`:subject_refit`: independently fit every D_-i (including censored subjects),
rebuild training risk sets, and score log mean_s exp(log L(D;beta_s)-log L(D_-i;beta_s)).
The latter reports LOO_LOG_SCORE, LOOIC=-2*score and NLPML=-score on a
partial-likelihood-ratio scale. ELPD_LOO, WAIC and Pareto-k are inapplicable
to this target and are missing, not relabeled fixed-risk estimates.

`folds` may select a pilot subset. Aggregate criteria remain missing unless all
n subjects succeed. `checkpoint_dir` stores one atomic JSON file per fold;
resume checks a fingerprint of data, fit settings, source and Julia version.
Supplied hyperparameters are held fixed; training-dependent prior scales and
preprocessing are recomputed inside each fit. No nested hyperparameter tuning.
`fit_function` is an optional fitting adapter with the fit_and_evaluate contract.
"""
function evaluate_cox_loo(X::AbstractMatrix, time::AbstractVector, event::AbstractVector;
    mode::Symbol = :fixed_risk, fit_kwargs = (;), full_fit = nothing,
    folds = eachindex(time), checkpoint_dir = nothing, resume::Bool = true,
    parallel::Bool = false, diagnostics::Bool = true,
    fit_function = fit_and_evaluate,
)
    check_loo_mode(mode)
    n = length(time)
    n == length(event) == size(X, 1) || throw(DimensionMismatch("X, time, event sizes differ"))
    n >= 2 || throw(ArgumentError("LOO requires at least two subjects"))
    all(x -> x in (0, 1), event) || throw(ArgumentError("event must be binary"))
    all(isfinite, time) && all(>=(0), time) || throw(ArgumentError("invalid survival times"))
    all(isfinite, X) && all(>=(0), X) || throw(ArgumentError("invalid covariates"))
    kwargs = Dict{Symbol,Any}(pairs(fit_kwargs))
    if mode == :fixed_risk
        kwargs[:compute_metrics] = true
        fit = full_fit === nothing ? fit_function(X, time, event; kwargs...) : full_fit
        fit.metrics === nothing && throw(ArgumentError("full_fit needs computed metrics"))
        m = first(eachrow(fit.metrics))
        summary = DataFrame(mode = ["fixed_risk"], target = ["event_attribution_fixed_risk"],
            n_subjects = [n], n_units = [sum(event)], n_completed = [sum(event)], complete = [true],
            LOO_LOG_SCORE = [m.ELPD_LOO], ELPD_LOO = [m.ELPD_LOO], LOOIC = [m.LOOIC],
            NLPML = [m.LPML], WAIC_pw1 = [m.WAIC_pw1], WAIC_pw2 = [m.WAIC_pw2],
            ParetoK_max = [m.ParetoK_max], ParetoK_bad = [m.ParetoK_bad],
            estimator = ["psis; NLPML uses unsmoothed harmonic mean"])
        return (summary = summary, pointwise = DataFrame(), full_fit = fit)
    end
    sum(event) >= 2 || throw(ArgumentError("subject_refit requires an event in every training fold"))
    get(kwargs, :approx, false) && throw(ArgumentError("subject_refit requires approx=false"))
    selected = collect(Int, folds)
    !isempty(selected) && all(i -> 1 <= i <= n, selected) && allunique(selected) ||
        throw(ArgumentError("folds must be nonempty, unique subject indices in 1:n"))
    if parallel
        Threads.nthreads() > 1 || throw(ArgumentError("parallel LOO requires julia -t N with N>1"))
        kwargs[:multichains] = false # parallelize folds, not nested chains
    end
    # Diagnostics affect saved records, so include the setting in checkpoint identity.
    fingerprint = _loo_checkpoint_manifest(X, time, event,
        merge(kwargs, Dict(:loo_diagnostics => diagnostics)), checkpoint_dir, resume)
    records = Vector{Dict{String,Any}}(undef, length(selected))
    function run_fold(k)
        i = selected[k]
        path = checkpoint_dir === nothing ? nothing : joinpath(checkpoint_dir, "fold_$(lpad(i, 5, '0')).json")
        if resume && path !== nothing && isfile(path)
            saved = JSON.parsefile(path; use_mmap = false)
            get(saved, "fingerprint", "") == fingerprint && get(saved, "subject", 0) == i ||
                error("invalid checkpoint for subject $i")
            if get(saved, "status", "") == "success"
                isfinite(saved["log_score"]) || error("invalid saved score for subject $i")
                records[k] = saved
                return
            end
        end
        record = try
            _loo_refit_fold(X, time, event, i, kwargs, fit_function, diagnostics)
        catch err
            err isa InterruptException && rethrow()
            @error "LOO refit failed" subject = i exception = (err, catch_backtrace())
            Dict{String,Any}("subject" => i, "event" => Int(event[i]), "status" => "failed",
                             "log_score" => nothing, "error" => sprint(showerror, err))
        end
        record["fingerprint"] = fingerprint
        path === nothing || _loo_atomic_json(path, record)
        records[k] = record
        @info "LOO fold finished" subject = i n status = record["status"]
    end
    if parallel
        Threads.@threads for k in eachindex(selected)
            run_fold(k)
        end
    else
        foreach(run_fold, eachindex(selected))
    end
    good = filter(r -> r["status"] == "success", records)
    complete = length(good) == n
    total = complete ? sum(r["log_score"] for r in good) : missing
    summary = DataFrame(mode = ["subject_refit"], target = ["subject_deletion_partial_likelihood_ratio"],
        n_subjects = [n], n_units = [n], n_completed = [length(good)], complete = [complete],
        LOO_LOG_SCORE = [total], ELPD_LOO = [missing], LOOIC = [-2 * total], NLPML = [-total],
        WAIC_pw1 = [missing], WAIC_pw2 = [missing], ParetoK_max = [missing], ParetoK_bad = [missing],
        estimator = ["independent MCMC refits; log posterior mean partial-likelihood ratio"])
    # Keep missing fields (e.g. failed fits and unavailable R-hat) as CSV missing values.
    keys_all = sort!(collect(union((Set(keys(r)) for r in records)...)))
    pointwise = DataFrame([Symbol(key) => [something(get(r, key, nothing), missing) for r in records]
                           for key in keys_all])
    sort!(pointwise, :subject)
    return (summary = summary, pointwise = pointwise, full_fit = full_fit)
end
