"""
    _build_riskset(time::AbstractVector{<:Real}) -> CoxRiskSet

Construct a compact risk-set representation for Cox proportional hazards models.

The observations are sorted in descending order of their observed times, so that
for any unique event/censoring time group `g`, the corresponding risk set can be
represented by the prefix `order[1:group_stop[g]]`, which contains all observations
with `time >= t_g`.

Observations with identical time values are grouped together. The returned
`CoxRiskSet` contains:

- `order`: indices of observations sorted by decreasing time;
- `group_start`: starting position of each unique time group in `order`;
- `group_stop`: ending position of each unique time group in `order`;
- `obs_to_group`: mapping from each original observation index to its time group.
"""
function _build_riskset(time::AbstractVector{<:Real})
    n = length(time)
    order = sortperm(collect(Float64, time); rev=true)
    group_start = Int[]
    group_stop = Int[]
    obs_to_group = Vector{Int}(undef, n)

    pos = 1
    group = 0
    while pos <= n
        group += 1
        start_pos = pos
        t = time[order[pos]]
        while pos <= n && time[order[pos]] == t
            obs_to_group[order[pos]] = group
            pos += 1
        end
        push!(group_start, start_pos)
        push!(group_stop, pos - 1)
    end

    return CoxRiskSet(order, group_start, group_stop, obs_to_group)
end

"""
    _standardize_columns(z::Matrix{Float64}) -> Tuple{Matrix{Float64}, Vector{Float64}, Vector{Float64}}

Standardize each column of the input matrix to have mean zero and unit standard
deviation.

Returns a tuple `(out, mu, sigma)`, where:

- `out` is the standardized matrix;
- `mu` contains the column means of the original matrix;
- `sigma` contains the column standard deviations used for scaling.
data, such as validation or test matrices.
"""
function _standardize_columns(z::Matrix{Float64})
    p = size(z, 2)
    mu = vec(mean(z; dims=1))
    sigma = Vector{Float64}(undef, p)
    out = similar(z)
    for j in 1:p
        s = std(view(z, :, j))
        sigma[j] = s > 0 ? s : 1.0
        @views out[:, j] .= (z[:, j] .- mu[j]) ./ sigma[j]
    end
    return out, mu, sigma
end

function _check_cox_tie_method(tie_method::Symbol)
    tie_method in (:efron, :breslow) ||
        throw(ArgumentError("tie_method must be :efron or :breslow."))
    return tie_method
end

"""
    prepare_cox_data(X, time, event; transform=:log,
                     closure=true, standardize=false, tie_method=:efron)

Prepare microbiome covariates for the Cox partial-likelihood model.
For `transform=:log`, replace only zeros with half the minimum positive abundance
in `X`, then optionally apply closure and take elementwise logarithms.
The returned `zero_replacement` field records the applied replacement
(zero when no replacement was needed).
"""
function prepare_cox_data(
    X::AbstractMatrix{<:Real},
    time::AbstractVector{<:Real},
    event::AbstractVector;
    transform::Symbol = :log,
    closure::Bool = true,
    standardize::Bool = false,
    tie_method::Symbol = :efron,
)
    n, p = size(X)
    length(time) == n || throw(DimensionMismatch("length(time) must match size(X, 1)."))
    length(event) == n || throw(DimensionMismatch("length(event) must match size(X, 1)."))
    p > 0 || throw(ArgumentError("X must contain at least one covariate column."))

    t = collect(Float64, time)
    all(isfinite, t) || throw(ArgumentError("time must be finite."))
    all(>=(0), t) || throw(ArgumentError("time must be nonnegative."))

    e = [Int(v) for v in event]
    all(v -> v == 0 || v == 1, e) ||
        throw(ArgumentError("event must contain only 0/1 indicators."))
    count(!=(0), e) > 0 || throw(ArgumentError("at least one uncensored event is required."))
    tie = _check_cox_tie_method(tie_method)

    x = Matrix{Float64}(X)
    all(isfinite, x) || throw(ArgumentError("X must be finite."))
    all(>=(0), x) || throw(ArgumentError("X must be nonnegative."))

    zero_replacement = 0.0

    z =
        if transform == :log
            x2 = copy(x)

            if any(iszero, x2)
                positive_vals = x2[x2 .> 0]
                isempty(positive_vals) &&
                    throw(ArgumentError("X contains no positive values; cannot perform log transform."))
                zero_replacement = minimum(positive_vals) / 2
                x2[x2 .== 0.0] .= zero_replacement
            end

            if closure
                rowsums = vec(sum(x2; dims=2))
                all(>(0), rowsums) ||
                    throw(ArgumentError("each row sum after zero handling must be positive."))
                x2 ./= rowsums
            end

            log.(x2)
        elseif transform == :none
            x
        else
            throw(ArgumentError("transform must be :log or :none."))
        end

    if standardize
        z, mu, sigma = _standardize_columns(z)
    else
        mu = zeros(p)
        sigma = ones(p)
    end

    return CoxPreparedData(
        z,
        t,
        e,
        _build_riskset(t),
        tie,
        transform,
        zero_replacement,
        closure,
        standardize,
        mu,
        sigma,
    )
end
