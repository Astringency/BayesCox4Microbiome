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
    pseudo_sigma_df::Float64 = 3.0
    pseudo_sigma_scale::Float64 = 10.0
end
