"""Diagonal Cox information at beta=0, evaluated only on the training design.

Efron ties use the same fractional removal of deaths as the fitted likelihood.
No true coefficients, test observations, or fitted coefficients enter this calculation.
"""
function null_cox_information(X, time, event; tie_method=:efron, standardize=false)
    data = prepare_cox_data(X, time, event; standardize, tie_method)
    z = data.z
    information = zeros(size(z, 2))
    for t in sort(unique(time[event .!= 0]))
        risk = findall(>=(t), time)
        deaths = findall((time .== t) .& (event .!= 0))
        d = length(deaths)
        s1, s2 = vec(sum(z[risk,:]; dims=1)), vec(sum(abs2, z[risk,:]; dims=1))
        e1, e2 = vec(sum(z[deaths,:]; dims=1)), vec(sum(abs2, z[deaths,:]; dims=1))
        for l in 0:d-1
            f = tie_method == :efron ? l/d : 0.0
            denom = length(risk) - f*d
            information .+= max.(0.0, (s2 .- f.*e2)./denom .- ((s1 .- f.*e1)./denom).^2)
        end
    end
    return information
end

function calibrate_rhs_scale(X, time, event; p0, tie_method=:efron, standardize=false)
    p, nevents = size(X,2), count(!iszero, event)
    0 < p0 < p || throw(ArgumentError("calibration requires 0 < p0 < p"))
    nevents > 0 || throw(ArgumentError("calibration requires observed events"))
    information = null_cox_information(X,time,event; tie_method,standardize)
    count(>(0),information) > p0 || throw(ArgumentError("insufficient informative columns for p0"))
    roots = sqrt.(information)
    expected(tau) = sum(tau .* roots ./ (1 .+ tau .* roots))
    lo, hi = 0.0, 1.0
    while expected(hi) < p0
        hi *= 2
        isfinite(hi) || error("failed to bracket information-calibrated scale")
    end
    for _ in 1:90
        mid = (lo+hi)/2
        if expected(mid) < p0; lo=mid; else; hi=mid; end
    end
    tau = (lo+hi)/2
    default = cox_tau0(p,p0,nevents)
    return Dict("n_events"=>nevents,"p0"=>p0,"tau_default"=>default,
        "tau_information"=>tau,"multiplier"=>tau/default,
        "expected_signal_count"=>expected(tau),"information_diagonal"=>information,
        "source"=>"training design and survival outcomes; information evaluated at beta=0",
        "approximation"=>"diagonal information effective-sparsity approximation")
end
