"""Cached event-time risk sets for centered/scaled log-time Cox effects."""
struct TimeCoxDesign
    X::Matrix{Float64}
    risk::Vector{Vector{Int}}
    deaths::Vector{Vector{Int}}
    basis::Vector{Float64}
    center::Float64
    scale::Float64
    tie_method::Symbol
end
function TimeCoxDesign(X,time,event; tie_method=:efron,
    center=mean(log.(time[event .!= 0])),scale=std(log.(time[event .!= 0])))
    size(X,1)==length(time)==length(event) || throw(DimensionMismatch("time Cox observations"))
    scale>0 && isfinite(scale) && all(>(0),time) || throw(ArgumentError("PH time basis needs positive, distinct event times"))
    tie_method in (:efron,:breslow) || throw(ArgumentError("unsupported ties"))
    times=sort(unique(time[event .!= 0]))
    return TimeCoxDesign(Matrix{Float64}(X),[findall(>=(t),time) for t in times],
        [findall((time.==t).&(event.!=0)) for t in times],
        (log.(times).-center)./scale,Float64(center),Float64(scale),tie_method)
end

"""Evaluate every member of each risk set at the same event-time basis value."""
function time_cox_loglikelihood_and_score(design::TimeCoxDesign,beta,delta)
    X=design.X;n,p=size(X)
    length(beta)==length(delta)==p || throw(DimensionMismatch("time Cox coefficients"))
    static=X*beta;slope=X*delta
    sb=zeros(n);ss=zeros(n);lp=0.0
    for k in eachindex(design.risk)
        R,D,g=design.risk[k],design.deaths[k],design.basis[k]
        eta=static .+ g.*slope
        shift=maximum(view(eta,R))
        isfinite(shift) || return (-Inf,zeros(p),zeros(p))
        wr=exp.(eta[R].-shift);wd=exp.(eta[D].-shift)
        s0,e0,d=sum(wr),sum(wd),length(D)
        lp+=sum(eta[D].-shift)
        invden,correction=0.0,0.0
        for l in 0:d-1
            fraction=design.tie_method==:efron ? l/d : 0.0
            denom=s0-fraction*e0
            lp-=log(denom);invden+=inv(denom);correction+=fraction/denom
        end
        for (i,w) in zip(R,wr)
            score=-w*invden;sb[i]+=score;ss[i]+=g*score
        end
        for (i,w) in zip(D,wd)
            score=1+w*correction;sb[i]+=score;ss[i]+=g*score
        end
    end
    return lp,X'*sb,X'*ss
end
function time_cox_loglikelihood(X,time,event,beta,delta;kwargs...)
    return first(time_cox_loglikelihood_and_score(TimeCoxDesign(X,time,event;kwargs...),beta,delta))
end

function ph_time_design(p,c,scope)
    scope in ("static","compositional","noncompositional","both") || error("unknown time scope")
    scope in ("noncompositional","both") && c==0 && error("time scope requires clinical covariates")
    B=nullspace(ones(1,p))
    micro=scope in ("compositional","both") ? vcat(B,zeros(c,p-1)) : zeros(p+c,0)
    clinical=scope in ("noncompositional","both") ? vcat(zeros(p,c),Matrix{Float64}(I,c,c)) : zeros(p+c,0)
    return hcat(micro,clinical)
end
