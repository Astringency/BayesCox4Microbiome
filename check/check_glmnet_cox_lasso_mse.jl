#!/usr/bin/env julia

using Pkg

function _activate_repo_project()
    project = abspath(joinpath(@__DIR__, "..", "Project.toml"))
    active = Base.active_project()
    active !== project && Pkg.activate(dirname(project))
end

_activate_repo_project()

include(joinpath(@__DIR__, "..", "scripts", "analysis_common.jl"))

using GLMNet

function _optional_float_arg(args, key::AbstractString)
    haskey(args, key) || return nothing
    return parse(Float64, args[key])
end

function fit_glmnet_cox_lasso_on_prepared_data(
    data::CoxPreparedData;
    lambda,
    nfolds::Integer,
    seed::Integer,
    alpha::Real,
    nlambda::Integer,
    lambda_min_ratio::Real,
    tol::Real,
    maxit::Integer,
    glmnet_standardize::Bool,
)
    time = Float64.(data.time)
    status = Float64.(data.event .!= 0)
    x = Matrix{Float64}(data.z)

    if lambda === nothing
        fit = GLMNet.glmnetcv(
            x,
            time,
            status;
            alpha = alpha,
            nfolds = nfolds,
            rng = MersenneTwister(seed),
            nlambda = nlambda,
            lambda_min_ratio = lambda_min_ratio,
            tol = tol,
            maxit = maxit,
            standardize = glmnet_standardize,
        )
        beta_hat = Float64.(GLMNet.coef(fit))
        lambda_used = Float64(GLMNet.lambdamin(fit))
        dev_ratio = fit.path.dev_ratio[argmin(fit.meanloss)]
        npasses = fit.path.npasses
        fit_kind = "glmnetcv"
    else
        fit = GLMNet.glmnet(
            x,
            time,
            status;
            alpha = alpha,
            lambda = [Float64(lambda)],
            tol = tol,
            maxit = maxit,
            standardize = glmnet_standardize,
        )
        beta_hat = Float64.(fit.betas[:, 1])
        lambda_used = Float64(fit.lambda[1])
        dev_ratio = fit.dev_ratio[1]
        npasses = fit.npasses
        fit_kind = "glmnet"
    end

    if data.standardize
        beta_hat ./= data.col_scale
    end

    return (
        fit = fit,
        beta_hat = beta_hat,
        lambda = lambda_used,
        dev_ratio = dev_ratio,
        npasses = npasses,
        fit_kind = fit_kind,
        active = count(!iszero, beta_hat),
    )
end

function coefficient_table(beta_hat::AbstractVector, beta_true::AbstractVector; context...)
    p = length(beta_true)
    df = DataFrame(
        variable = ["beta[$j]" for j in 1:p],
        beta_hat = beta_hat,
        beta_true = beta_true,
        nonzero = beta_true .!= 0,
        selected = beta_hat .!= 0,
        error = beta_hat .- beta_true,
        squared_error = (beta_hat .- beta_true) .^ 2,
    )
    for (key, value) in pairs(context)
        df[!, key] = fill(value, p)
    end
    select!(
        df,
        vcat(
            collect(keys(context)),
            [:variable, :beta_hat, :beta_true, :nonzero, :selected, :error, :squared_error],
        ),
    )
    return df
end

function metrics_row(beta_hat::AbstractVector, beta_true::AbstractVector; context...)
    nz = beta_true .!= 0
    zc = .!nz
    selected = beta_hat .!= 0
    tp = count(selected .& nz)
    fp = count(selected .& zc)
    fn = count((.!selected) .& nz)
    tn = count((.!selected) .& zc)
    return DataFrame([merge(
        NamedTuple(context),
        (
            MSE = mean((beta_hat .- beta_true) .^ 2),
            MSE_nonzero = any(nz) ? mean((beta_hat[nz] .- beta_true[nz]) .^ 2) : missing,
            MSE_zero = any(zc) ? mean(beta_hat[zc] .^ 2) : missing,
            MAE = mean(abs.(beta_hat .- beta_true)),
            max_abs_error = maximum(abs.(beta_hat .- beta_true)),
            beta_hat_norm = norm(beta_hat),
            active = count(selected),
            true_positive = tp,
            false_positive = fp,
            false_negative = fn,
            true_negative = tn,
            TPR = count(nz) > 0 ? tp / count(nz) : missing,
            FPR = count(zc) > 0 ? fp / count(zc) : missing,
        ),
    )])
end

function error_row(err; context...)
    return DataFrame([merge(
        NamedTuple(context),
        (
            MSE = missing,
            MSE_nonzero = missing,
            MSE_zero = missing,
            MAE = missing,
            max_abs_error = missing,
            beta_hat_norm = missing,
            active = missing,
            true_positive = missing,
            false_positive = missing,
            false_negative = missing,
            true_negative = missing,
            TPR = missing,
            FPR = missing,
            error_type = string(typeof(err)),
            error_message = sprint(showerror, err),
        ),
    )])
end

function main()
    args = parse_cli_args()
    outdir = ensure_dir(arg(args, "outdir", joinpath("results", "checks", "glmnet_cox_lasso_mse")))
    reps = arg(args, "reps", 1)
    ns = split_ints(arg(args, "n", "200"))
    ps = split_ints(arg(args, "p", "50"))
    nonzeros = split_ints(arg(args, "nonzero", "6"))
    baselines = split_symbols(arg(args, "baseline", "weibull"))
    seed0 = arg(args, "seed", 1)
    pseudocount = arg(args, "pseudocount", 0.0)
    standardize = arg(args, "standardize", false)
    closure = arg(args, "closure", true)
    tie_method = parse_tie_method(args)

    lambda = _optional_float_arg(args, "lambda")
    nfolds = arg(args, "nfolds", 5)
    alpha = arg(args, "alpha", 1.0)
    nlambda = arg(args, "nlambda", 100)
    lambda_min_ratio = arg(args, "lambda-min-ratio", 1e-4)
    tol = arg(args, "tol", 1e-7)
    maxit = arg(args, "maxit", 1_000_000)
    glmnet_standardize = arg(args, "glmnet-standardize", true)

    metrics = DataFrame()
    coefs = DataFrame()

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
        )
        context = (
            rep = rep,
            n = n,
            p = p,
            nonzero_count = nonzero,
            baseline = string(baseline),
            seed = seed,
            events = count(!=(0), sim.event),
            event_rate = mean(sim.event .!= 0),
            pseudocount = pseudocount,
            closure = closure,
            standardize = standardize,
            tie_method = string(tie_method),
            lambda_requested = lambda === nothing ? missing : Float64(lambda),
            nfolds = lambda === nothing ? nfolds : missing,
            alpha = alpha,
            nlambda = lambda === nothing ? nlambda : missing,
            lambda_min_ratio = lambda === nothing ? lambda_min_ratio : missing,
            tol = tol,
            maxit = maxit,
            glmnet_standardize = glmnet_standardize,
        )
        try
            data = prepare_cox_data(
                sim.X,
                sim.time,
                sim.event;
                transform = :log,
                pseudocount = pseudocount,
                closure = closure,
                standardize = standardize,
                tie_method = tie_method,
            )
            fitinfo = fit_glmnet_cox_lasso_on_prepared_data(
                data;
                lambda = lambda,
                nfolds = nfolds,
                seed = seed,
                alpha = alpha,
                nlambda = nlambda,
                lambda_min_ratio = lambda_min_ratio,
                tol = tol,
                maxit = maxit,
                glmnet_standardize = glmnet_standardize,
            )
            row = metrics_row(fitinfo.beta_hat, sim.beta; context...)
            row.converged = [true]
            row.fit_kind = [fitinfo.fit_kind]
            row.lambda = [fitinfo.lambda]
            row.dev_ratio = [fitinfo.dev_ratio]
            row.npasses = [fitinfo.npasses]
            append!(metrics, row; cols = :union)
            coef_context = merge(
                context,
                (
                    fit_kind = fitinfo.fit_kind,
                    lambda = fitinfo.lambda,
                    dev_ratio = fitinfo.dev_ratio,
                    npasses = fitinfo.npasses,
                ),
            )
            append!(coefs, coefficient_table(fitinfo.beta_hat, sim.beta; coef_context...); cols = :union)
            println(
                "rep=$rep n=$n p=$p nonzero=$nonzero baseline=$baseline " *
                "lambda=$(fitinfo.lambda) active=$(fitinfo.active) MSE=$(row.MSE[1])",
            )
        catch err
            @warn "GLMNet Cox LASSO fit failed." rep n p nonzero baseline exception=(err, catch_backtrace())
            row = error_row(err; context...)
            row.converged = [false]
            append!(metrics, row; cols = :union)
        end
    end

    CSV.write(joinpath(outdir, "glmnet_cox_lasso_mse_long.csv"), metrics)
    CSV.write(joinpath(outdir, "glmnet_cox_lasso_coefficients.csv"), coefs)

    if nrow(metrics) > 0 && :MSE in propertynames(metrics)
        ok = collect(skipmissing(metrics.MSE))
        if !isempty(ok)
            summary = combine(
                groupby(
                    metrics,
                    [
                        :n,
                        :p,
                        :nonzero_count,
                        :baseline,
                        :pseudocount,
                        :standardize,
                        :fit_kind,
                        :alpha,
                        :glmnet_standardize,
                    ],
                ),
                :MSE => mean => :MSE_mean,
                :MSE => std => :MSE_sd,
                :MSE_nonzero => mean => :MSE_nonzero_mean,
                :MSE_zero => mean => :MSE_zero_mean,
                :MAE => mean => :MAE_mean,
                :active => mean => :active_mean,
                :TPR => mean => :TPR_mean,
                :FPR => mean => :FPR_mean,
                :lambda => mean => :lambda_mean,
                :events => mean => :events_mean,
                :event_rate => mean => :event_rate_mean,
            )
            CSV.write(joinpath(outdir, "glmnet_cox_lasso_mse_summary.csv"), summary)
        end
    end

    println("GLMNet Cox LASSO MSE check written to $(abspath(outdir))")
end

main()
