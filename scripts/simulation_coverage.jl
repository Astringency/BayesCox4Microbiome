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
