Base.@kwdef struct RegularizedHorseshoePrior <: AbstractCoxPrior
    tau_scale::Float64 = 1.0
    use_cox_tau0::Bool = true
    p0::Float64 = 1.0
    local_scale::Float64 = 1.0
    slab_df::Float64 = 4.0
    slab_s2::Float64 = 2.0
end
