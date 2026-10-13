module T1DPartitionTests
using Test
include(joinpath(@__DIR__,"..","scripts","lib","realdata","t1d_review.jl"))

@testset "T1D partitioned folds match serial nested CV" begin
    input=joinpath(@__DIR__,"..","data","T1D.csv")
    df=CSV.read(input,DataFrame)
    cols=filter(c -> !(c in ("Sex","Antibiotics","time","status")),names(df))
    X,time,event=Matrix{Float64}(df[:,cols]),Float64.(df.time),Int.(df.status)
    C,cnames=t1d_clinical_design(df)
    for method in ("cox-rhs","prox-sslasso"), adjusted in (false,true)
        mktempdir() do temp
            serial=joinpath(temp,"serial")
            partition=joinpath(temp,"partition")
            spec=Dict("method_name"=>method,"adjustment"=>adjusted ? "adjusted" : "unadjusted",
                "n_samples"=>12,"warmup"=>8,"chains"=>2,"hmc_n_leapfrog"=>3,
                "outer_folds"=>2,"inner_folds"=>2,"seed"=>1,"component_cols"=>cols)
            sampler=Dict(:sampler=>:hmc,:n_samples=>12,:warmup=>8,:chains=>2,:multichains=>true,
                :target_accept=>0.9,:hmc_n_leapfrog=>3,:noncentered=>true,:progress=>false)
            prior=Dict{Symbol,Any}(:alpha_prior_kind=>:invgamma,:alpha_prior_shape=>1.,:alpha_prior_scale=>size(X,2))
            car=Dict(:use_car=>false,:sample_car_strength=>false)
            model=Dict{Symbol,Any}(:constraint=>:prox,:prox_sum0_lambda=>1e-6,:pseudocount=>1e-6,:standardize=>false,:p0=>nothing)
            adjusted && merge!(model,Dict(:clinical_names=>cnames,:clinical_prior_df=>3.,:clinical_prior_scale=>2.5))
            grids=deepcopy(PRIOR_PARAM_GRIDS)
            grids[method]=[Dict{String,Any}((method=="cox-rhs" ? "rhs_local_scale" : "sslasso_spike_scale")=>0.5)]
            reference=evaluate_nested_cindex(X,time,event;priors=[method],base_sampler_kwargs=sampler,
                base_prior_kwargs=prior,base_car_kwargs=car,base_model_kwargs=model,outdir=serial,
                dataset_id="T1D",outer_folds=2,inner_folds=2,seed=1,prior_grids=grids,
                clinical_X=adjusted ? C : nothing)
            @test reference.summary.status[1]=="success"
            mkpath(joinpath(partition,"nested_cv",method))
            _write_tuning_json(joinpath(partition,"analysis_specification.json"),spec)
            cp(joinpath(serial,"nested_cv",method,"manifest.json"),joinpath(partition,"nested_cv",method,"manifest.json"))
            setup=t1d_partition_setup(partition;input,smoke=true)
            for fold in 1:2
                run_t1d_partition(setup,string(fold))
                label="outer_$(lpad(fold,2,'0'))"
                original=JSON.parse(read(joinpath(serial,"nested_cv",method,label*".json"),String))
                output=joinpath(partition,"nested_cv",method,label*".json")
                actual=JSON.parse(read(output,String))
                for key in ("fingerprint","selected_config_index","selected_config","inner_Cindex_mean",
                            "test_Cindex_mean","test_Cindex_median","test_Cindex_MAP")
                    @test actual[key]==original[key]
                end
                old_samples=CSV.read(joinpath(serial,"nested_cv",method,label,"fit","samples_wide.csv"),DataFrame)
                new_samples=CSV.read(joinpath(partition,"nested_cv",method,label,"fit","samples_wide.csv"),DataFrame)
                @test isequal(old_samples,new_samples)
                timestamp=stat(output).mtime
                run_t1d_partition(setup,string(fold))
                @test stat(output).mtime==timestamp
            end
            run_t1d_partition(setup,"full_tuning")
            @test isfile(joinpath(partition,"full_data_tuning","selected_config.json"))
            bad=copy(spec);bad["warmup"]=9
            _write_tuning_json(joinpath(partition,"analysis_specification.json"),bad)
            @test_throws ErrorException t1d_partition_setup(partition;input,smoke=true)
        end
    end
end

end # module
