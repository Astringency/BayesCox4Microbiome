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

"""
    prepare_cox_data(X, time, event; transform=:log, pseudocount=1e-6,
                     closure=true, standardize=false)

Prepare microbiome covariates for the Cox partial-likelihood model.
`transform=:log` implements the paper's elementwise log transform.
"""
function prepare_cox_data(
    X::AbstractMatrix{<:Real},
    time::AbstractVector{<:Real},
    event::AbstractVector;
    transform::Symbol = :log,
    pseudocount::Real = 0.0,
    closure::Bool = true,
    standardize::Bool = false,
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

    pc = _check_nonnegative(pseudocount, "pseudocount")
    x = Matrix{Float64}(X)
    all(isfinite, x) || throw(ArgumentError("X must be finite."))
    all(>=(0), x) || throw(ArgumentError("X must be nonnegative."))

    applied_pc = Float64(pc)

    z =
        if transform == :log
            x2 = copy(x)

            if pc == 0
                # replaced zero abundances with half of the minimum observed abundance
                positive_vals = x2[x2 .> 0]

                isempty(positive_vals) &&
                    throw(ArgumentError("X contains no positive values; cannot perform log transform."))

                min_positive = minimum(positive_vals)
                half_min = min_positive / 2

                x2[x2 .== 0.0] .= half_min
                applied_pc = half_min
            else
                # plus pseudocount
                x2 .+= pc
                applied_pc = Float64(pc)
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
        transform,
        applied_pc,
        closure,
        standardize,
        mu,
        sigma,
    )
end
