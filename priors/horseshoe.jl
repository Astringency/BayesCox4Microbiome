Base.@kwdef struct HorseshoePrior <: AbstractCoxPrior
    tau_scale::Float64 = 1.0
    use_cox_tau0::Bool = false
    p0::Float64 = 1.0
    local_scale::Float64 = 1.0
end
