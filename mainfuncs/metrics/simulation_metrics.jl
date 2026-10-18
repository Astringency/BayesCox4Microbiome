"""Mean squared coefficient error, on the same scale as the truth."""
function coefficient_mse(truth::AbstractVector, estimate::AbstractVector)
    length(truth) == length(estimate) || throw(DimensionMismatch("truth and estimate lengths differ"))
    isempty(truth) && return missing
    return mean(abs2, estimate .- truth)
end

using DataFrames
using Statistics

const SIMULATION_COVERAGE_COLUMNS = (
    :coverage_95_all, :coverage_95_nonzero, :coverage_95_zero,
)

_missing_simulation_coverage() = NamedTuple{SIMULATION_COVERAGE_COLUMNS}((missing, missing, missing))
_finite_coverage_value(x) = x isa Real && isfinite(x)

"""
    simulation_coverage(beta, lower, upper)

Fraction of true coefficients covered by their marginal 95% credible intervals,
on the original coefficient scale. Endpoints are inclusive. An empty group or
any invalid interval in that group gives `missing`, without changing its denominator.
"""
function simulation_coverage(beta::AbstractVector, lower::AbstractVector, upper::AbstractVector)
    length(beta) == length(lower) == length(upper) ||
        throw(DimensionMismatch("truth and credible interval lengths must match."))
    all(_finite_coverage_value, beta) || return _missing_simulation_coverage()
    valid = [_finite_coverage_value(lo) && _finite_coverage_value(hi) && lo <= hi
        for (lo, hi) in zip(lower, upper)]
    covered = [ok && lo <= truth <= hi for (ok, truth, lo, hi) in zip(valid, beta, lower, upper)]
    function group_coverage(indices)
        isempty(indices) && return missing
        all(valid[indices]) || return missing
        return count(covered[indices]) / length(indices)
    end
    return (
        coverage_95_all = group_coverage(collect(eachindex(beta))),
        coverage_95_nonzero = group_coverage(findall(!iszero, beta)),
        coverage_95_zero = group_coverage(findall(iszero, beta)),
    )
end

"""Attach interval coverage to each point-estimate row; it is the same for mean, median and MAP."""
function add_simulation_coverage!(metrics::DataFrame, summary::AbstractDataFrame;
    beta = :true_beta in propertynames(summary) ? summary.true_beta : nothing)
    lower_col = :lower in propertynames(summary) ? :lower : :q025
    upper_col = :upper in propertynames(summary) ? :upper : :q975
    coverage = if beta === nothing || !(lower_col in propertynames(summary)) || !(upper_col in propertynames(summary))
        _missing_simulation_coverage()
    else
        simulation_coverage(beta, summary[!, lower_col], summary[!, upper_col])
    end
    for col in SIMULATION_COVERAGE_COLUMNS
        metrics[!, col] = fill(coverage[col], nrow(metrics))
    end
    return metrics
end

_coverage_values(values) = Float64[x for x in values if _finite_coverage_value(x)]
_coverage_mean(values) = (v = _coverage_values(values); isempty(v) ? missing : mean(v))
_coverage_sd(values) = (v = _coverage_values(values); length(v) < 2 ? missing : std(v))
_coverage_n(values) = count(_finite_coverage_value, values)

"""Aggregate per-repetition coverage, retaining the number of valid repetitions for each group."""
function simulation_coverage_summary_specs(metrics::AbstractDataFrame)
    specs = Any[]
    for col in SIMULATION_COVERAGE_COLUMNS
        col in propertynames(metrics) || continue
        push!(specs, col => _coverage_mean => Symbol(col, "_mean"))
        push!(specs, col => _coverage_sd => Symbol(col, "_sd"))
        push!(specs, col => _coverage_n => Symbol(col, "_n"))
    end
    return specs
end

function false_discovery(beta::AbstractVector, summary::DataFrame)
    selected = (summary.lower .* summary.upper) .> 0
    truth = beta .!= 0
    fp = sum((.!truth) .& selected)
    tp = sum(truth .& selected)
    fn = sum(truth .& (.!selected))
    tn = sum((.!truth) .& (.!selected))
    fdr = fp + tp == 0 ? 0.0 : fp / (fp + tp)
    return (FP = fp, FN = fn, TP = tp, TN = tn, FDR = fdr)
end

function add_simulation_test_metrics!(metrics, summary, train, test;
    tie_method = :efron)
    # Saved summaries are already on the unstandardized coefficient scale.
    # Removing training centering only adds a constant to all risk scores.
    data = prepare_cox_data(train.X, train.time, train.event;
        transform = :log, standardize = false, tie_method)
    z_test = _validation_design_matrix(train.X, test.X, data)
    metrics.Cindex_train = copy(metrics.Cindex)
    metrics.Cindex_test = [concordance_index(test.time, test.event,
        z_test * Float64.(summary[!, est == "MAP" ? :map : Symbol(est)]))
        for est in string.(metrics.estimate)]
    metrics.n_test = fill(length(test.time), nrow(metrics))
    metrics.test_seed = fill(test.test_seed, nrow(metrics))
    return metrics
end

function annotate_simulation_metrics(metrics::DataFrame, sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    out = copy(metrics)
    mse = Dict(
        "mean" => coefficient_mse(sim.beta, Float64.(summary[!, :mean])),
        "median" => coefficient_mse(sim.beta, Float64.(summary[!, :median])),
        "MAP" => coefficient_mse(sim.beta, Float64.(summary[!, :map])),
    )
    out[!, :prior] = fill(prior, nrow(out))
    out[!, :rep] = fill(rep, nrow(out))
    out[!, :n] = fill(n, nrow(out))
    out[!, :p] = fill(p, nrow(out))
    out[!, :nonzero_count] = fill(nonzero, nrow(out))
    out[!, :p0] = fill(p0, nrow(out))
    out[!, :baseline] = fill(string(baseline), nrow(out))
    out[!, :scenario] = fill(string(get(sim, :scenario, :baseline)), nrow(out))
    out[!, :MSE] = [get(mse, string(e), NaN) for e in out.estimate]
    add_simulation_coverage!(out, summary; beta = sim.beta)
    return out
end

function simulation_false_discovery_table(sim, summary::DataFrame; rep, n, p, nonzero, p0, baseline, prior)
    fd = false_discovery(sim.beta, summary)
    return DataFrame(
        rep = [rep],
        n = [n],
        p = [p],
        nonzero_count = [nonzero],
        p0 = [p0],
        baseline = [string(baseline)],
        scenario = [string(get(sim, :scenario, :baseline))],
        prior = [prior],
        FP = [fd.FP],
        FN = [fd.FN],
        TP = [fd.TP],
        TN = [fd.TN],
        FDR = [fd.FDR],
    )
end

function _simulation_accuracy(summary::DataFrame, beta_true, estimate::AbstractString)
    if beta_true === nothing
        return Dict{Symbol,Any}(
            :MSE => missing,
            :FDR => missing,
            :FP => missing,
            :FN => missing,
            :TP => missing,
            :TN => missing,
        )
    end
    col = estimate == "MAP" ? :map : Symbol(estimate)
    beta_hat = Float64.(summary[!, col])
    fd = false_discovery(beta_true, summary)
    return Dict{Symbol,Any}(
        :MSE => coefficient_mse(beta_true, beta_hat),
        :FDR => fd.FDR,
        :FP => fd.FP,
        :FN => fd.FN,
        :TP => fd.TP,
        :TN => fd.TN,
    )
end

