"""
    Spike-and-Slab LASSO:

    βⱼ∣ γⱼ, sⱼ∼ Laplace(0, sⱼ), j=1,2,…,p,
    sⱼ= (1-γⱼ)s₀ + γⱼs₁,
    γⱼ∣ θ ∼ Bernoulli(θ),
    θ ∼ Beta(1,1).
"""
Base.@kwdef struct SpikeSlabLassoPrior <: AbstractCoxPrior
    spike_scale::Float64 = 0.04
    slab_scale::Float64 = 0.5
    theta_a::Float64 = 1.0
    theta_b::Float64 = 1.0
    proximal::Bool = false
end
