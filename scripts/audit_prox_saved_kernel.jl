# Read-only checks at retained states, using the same frozen package as the fits.
using BayesCox4Micro, CSV, DataFrames, JSON, Statistics, Test

function audit_saved(source, out)
    meta=JSON.parsefile(joinpath(source,"metadata.json")); p=Int(meta["p"])
    df=CSV.read(replace(meta["test_data"],"/test_data/"=>"/training_data/"),DataFrame)
    data=prepare_cox_data(Matrix{Float64}(df[:,["x$j" for j in 1:p]]),df.time,df.event)
    prior=SpikeSlabLassoPrior(spike_scale=meta["sslasso_spike_scale"],
        slab_scale=meta["sslasso_slab_scale"],proximal=true)
    wide=CSV.read(joinpath(source,"samples_wide.csv"),DataFrame)
    gamma=trues(p);theta=Ref(0.5)
    target=CoxProxSSLassoTarget(data,prior,gamma,theta;constraint=:prox,
        prox_sum0_lambda=meta["prox_sum0_lambda"],prox_l1_lambda=meta["prox_l1_lambda"],
        alpha_prior_shape=meta["alpha_prior_shape"],alpha_prior_scale=meta["alpha_prior_scale"],
        prox_root_tol=1e-10,prox_root_maxiter=100)
    @test target.gamma === gamma
    errors=Dict[]
    for ch in groupby(wide,:chain), k in [100,2500,4900]
        r=ch[k,:]
        u=vcat(Float64.(collect(r[["beta[$j]" for j in 1:p]])),[r.eta])
        gamma .= Bool.(collect(r[["gamma[$j]" for j in 1:p]]));theta[]=r.theta
        actual=logdensity(target,u);lp_error=abs(actual-r.logposterior)
        g=gradient!(zeros(p+1),target,u); fd=zeros(p+1)
        for j in eachindex(u)
            h=1e-6*max(1.,abs(u[j]));up=copy(u);um=copy(u);up[j]+=h;um[j]-=h
            fd[j]=(logdensity(target,up)-logdensity(target,um))/(2h)
        end
        err=maximum(abs.(g-fd)./(1 .+ abs.(fd)))
        @test lp_error < 1e-8
        @test err < 1e-3
        push!(errors,Dict("chain"=>r.chain,"iteration"=>k,"density_abs_error"=>lp_error,
                          "gradient_max_scaled_error"=>err))
    end
    open(out,"w") do io;JSON.print(io,Dict("source"=>source,"gamma_alias_pass"=>true,
        "n_saved_states"=>length(errors),"checks"=>errors),2);end
    println("SAVED KERNEL PASS ",maximum(r["gradient_max_scaled_error"] for r in errors))
end
audit_saved(ARGS...)
