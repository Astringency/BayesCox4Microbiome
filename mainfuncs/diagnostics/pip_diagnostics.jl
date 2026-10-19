"""Autocorrelation-aware MCSE of each binary indicator mean within one chain.
Constant chains and chains with fewer than ten retained draws are not estimable.
"""
function _indicator_chain_mcse(raw::AbstractMatrix, parameters)
    result = Union{Missing,Float64}[missing for _ in parameters]
    size(raw, 1) >= 10 || return result
    varying = findall(j -> minimum(view(raw, :, j)) != maximum(view(raw, :, j)), axes(raw, 2))
    isempty(varying) && return result
    MCMCChains = _require_pkg(:MCMCChains)
    chains = _call_latest(getproperty(MCMCChains, :Chains),
        reshape(raw[:, varying], size(raw, 1), length(varying), 1), Symbol.(parameters[varying]))
    estimates = DataFrame(_call_latest(getproperty(MCMCChains, :mcse), chains; kind = mean))
    for (j, estimate) in zip(varying, estimates.mcse)
        !ismissing(estimate) && isfinite(estimate) && estimate > 0 && (result[j] = estimate)
    end
    return result
end

"""
    pip_consistency_diagnostics(wide; mixing = selection_mixing_diagnostics(wide))

Compare each taxon's inclusion probabilities across independent retained chains.
`pip_range` is max(PIP) - min(PIP). `pip_mcse_standardized_max` is the largest
pairwise absolute PIP difference divided by sqrt(MCSE₁² + MCSE₂²), using
autocorrelation-aware, within-chain MCSEs of the binary means. It is missing
unless every pair is estimable. The available-pair maximum and its denominator
count are also retained for auditing; neither is a convergence test or a
multiple-comparison-adjusted statistic. Constant chains never receive zero MCSE.
"""
function pip_consistency_diagnostics(wide::AbstractDataFrame;
    mixing = selection_mixing_diagnostics(wide))
    isempty(mixing) && return DataFrame()
    rows = NamedTuple[]
    for g in groupby(mixing, :parameter)
        n = nrow(g)
        length(unique(g.chain)) == n || throw(ArgumentError("duplicate taxon-by-chain summaries"))
        pips = Float64.(g.inclusion_probability)
        estimates = Float64[]
        for c in 1:n, d in c+1:n
            a, b = g.pip_mcse[c], g.pip_mcse[d]
            if !ismissing(a) && !ismissing(b) && isfinite(a) && isfinite(b) && a > 0 && b > 0
                push!(estimates, abs(pips[c] - pips[d]) / hypot(a, b))
            end
        end
        pairs = n * (n - 1) ÷ 2
        available_max = isempty(estimates) ? missing : maximum(estimates)
        constants = pips[g.constant_chain]
        push!(rows, (parameter = first(g.parameter), n_chains = n,
            pip_pooled = sum(pips .* g.n_draws) / sum(g.n_draws),
            pip_min = minimum(pips), pip_max = maximum(pips),
            pip_range = n >= 2 ? maximum(pips) - minimum(pips) : missing,
            pip_mcse_standardized_max = pairs > 0 && length(estimates) == pairs ? available_max : missing,
            pip_mcse_standardized_available_max = available_max,
            n_chain_pairs = pairs, n_estimable_chain_pairs = length(estimates),
            n_constant_chains = length(constants),
            conflicting_constant_chains = length(unique(constants)) > 1))
    end
    return DataFrame(rows)
end
