"""
    Horseshoe Prior:
    
    βⱼ∣ τ,λⱼ∼ N(0, τ²λⱼ²), j=1,2,…,p,
    λⱼ∼ C⁺(0,1),
    τ ∼ C⁺(0,τ̃₀).
    τ̃₀ = p₀ / [(p-p₀) √n] for linear model,
    τ̃₀ = p₀ / [(p-p₀) √n_δ] for cox model.
"""

Base.@kwdef struct HorseshoePrior <: AbstractCoxPrior
    tau_scale::Float64 = 1.0
    use_cox_tau0::Bool = false
    p0::Float64 = 1.0
    local_scale::Float64 = 1.0
end
