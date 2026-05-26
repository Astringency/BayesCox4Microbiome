"""
    Bayesian LASSO:

    βⱼ∣ τⱼ∼ N(0, σ²τⱼ²),
    τⱼ² ∼ Gamma(1, λ²/2), j=1,2,…,p,
    λ² ∼ Gamma(a₀, b₀) or a fixed value.
"""
Base.@kwdef struct BayesianLassoPrior <: AbstractCoxPrior
    lambda::Union{Nothing,Float64} = nothing
    lambda2_shape::Float64 = 1.0
    lambda2_rate::Float64 = 0.1
    proximal::Bool = false
end
