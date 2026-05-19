Base.@kwdef struct BayesianLassoPrior <: AbstractCoxPrior
    lambda::Union{Nothing,Float64} = nothing
    lambda2_shape::Float64 = 1.0
    lambda2_rate::Float64 = 0.1
    tau2_shape::Float64 = 1.0
    proximal::Bool = false
end
