Base.@kwdef struct SpikeSlabLassoPrior <: AbstractCoxPrior
    spike_scale::Float64 = 0.04
    slab_scale::Float64 = 0.5
    theta_a::Float64 = 1.0
    theta_b::Float64 = 1.0
    proximal::Bool = false
end
