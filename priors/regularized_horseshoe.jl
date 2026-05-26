"""
    Regularized Horseshoe Prior:

    βⱼ∣ τ,λⱼ,c ∼ N(0, τ²λ̃ⱼ²), j=1,2,…,p,
    λ̃ⱼ² = c²λⱼ² / (c² + τ²λⱼ²), 
    λⱼ∼ C⁺(0,1), τ ∼ C⁺(0,σ̃τ̃₀),
    c² ∼ Inv-Gamma(ν/2, νs²/2),
    σ̃² ∼ Hald-t(3, 0, 10) for linear model, or σ̃ = 1 for cox model,
    τ̃₀ = p₀σ̃ / [(p-p₀) √n] for linear model, or τ̃₀ = p₀σ̃ / [(p-p₀) √n_δ] for cox model.
"""

Base.@kwdef struct RegularizedHorseshoePrior <: AbstractCoxPrior
    tau_scale::Float64 = 1.0
    use_cox_tau0::Bool = true
    p0::Float64 = 1.0
    local_scale::Float64 = 1.0
    slab_df::Float64 = 4.0
    slab_s2::Float64 = 2.0
    tau_formula::Symbol = :auto
    pseudo_sigma::Union{Nothing,Float64} = nothing
    sample_pseudo_sigma::Bool = false
    # Used when sample_pseudo_sigma=true: pseudo_sigma^2 ~ half-t(df, 0, scale).
    pseudo_sigma_df::Float64 = 3.0
    pseudo_sigma_scale::Float64 = 10.0
end
