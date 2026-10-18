function _posterior_gradient_dataset(; tie_method::Symbol = :efron)
    rng = MersenneTwister(5501)
    n, p = 30, 5
    X = rand(rng, n, p) .+ 0.2
    time = Float64.(rand(rng, 1:8, n))
    event = Int.(rand(rng, n) .< 0.65)
    event[1] = 1
    data = prepare_cox_data(X, time, event; transform = :none, tie_method = tie_method)
    return data, X, time, event
end

function _posterior_gradient_prior_cases(X, time, event)
    return [
        ("hs", cox_paper_prior("hs"; p0 = 1.0)),
        ("rhs", cox_paper_prior("rhs"; p0 = 1.0)),
        ("new-rhs", cox_paper_prior("new-rhs"; p0 = 1.0)),
        ("est-rhs", cox_paper_prior("est-rhs"; p0 = 1.0, X = X, time = time, event = event, transform = :none)),
        ("cox-hs", cox_paper_prior("cox-hs"; p0 = 1.0)),
        ("cox-rhs", cox_paper_prior("cox-rhs"; p0 = 1.0)),
        ("lasso", cox_paper_prior("lasso")),
        ("sslasso", cox_paper_prior("sslasso")),
        ("prox-lasso", cox_paper_prior("prox-lasso")),
        ("prox-sslasso", cox_paper_prior("prox-sslasso")),
    ]
end

function _posterior_gradient_initial(prior, data; rng = MersenneTwister(99))
    p = size(data.z, 2)
    n_events = count(!=(0), data.event)
    if B._is_proximal_l1_prior(prior)
        base = [0.8, -0.6, 0.4, -0.1, 0.05]
        return vcat(base[1:p], [log(0.3)])
    elseif prior isa RegularizedHorseshoePrior
        u = initial_state(prior, p, n_events; n_obs = size(data.z, 1))
    else
        u = initial_state(prior, p, n_events)
    end
    u .+= 0.03 .* randn(rng, length(u))
    return u
end

function _fd_gradient(f, u; h = 1e-6)
    g = zeros(Float64, length(u))
    for j in eachindex(u)
        e = zeros(Float64, length(u))
        e[j] = h
        g[j] = (f(u .+ e) - f(u .- e)) / (2h)
    end
    return g
end

@testset "Posterior gradient finite differences for paper priors" begin
    data, X, time, event = _posterior_gradient_dataset()
    rng = MersenneTwister(101)
    for (name, prior) in _posterior_gradient_prior_cases(X, time, event)
        u = _posterior_gradient_initial(prior, data; rng = rng)
        if prior isa BayesianLassoPrior && prior.proximal
            work = B.ProxGradientWorkspace(data)
            f = x -> B._lasso_prox_logposterior(
                x,
                data,
                prior;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_scale = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
            grad = zeros(Float64, length(u))
            B._lasso_prox_gradient!(
                grad,
                u,
                data,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_scale = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
        elseif prior isa SpikeSlabLassoPrior && prior.proximal
            gamma = [true, false, true, false, true]
            theta = 0.45
            work = B.ProxGradientWorkspace(data)
            f = x -> B._ssl_prox_logposterior(
                x,
                data,
                prior,
                gamma,
                theta;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_scale = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
            grad = zeros(Float64, length(u))
            B._ssl_prox_gradient!(
                grad,
                u,
                data,
                prior,
                gamma,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
                prox_l1_lambda = 0.9,
                alpha_prior_shape = 1.0,
                alpha_prior_scale = 0.4,
                prox_root_tol = 1e-12,
                prox_root_maxiter = 200,
            )
        else
            work = CoxGradientWorkspace(data)
            f = x -> logposterior(x, data, prior; constraint = :prox, prox_sum0_lambda = 0.7)
            grad = zeros(Float64, length(u))
            B.logposterior_and_gradient!(
                grad,
                u,
                data,
                prior,
                work;
                constraint = :prox,
                prox_sum0_lambda = 0.7,
            )
        end
        fd = _fd_gradient(f, u)
        @test maximum(abs.(grad .- fd)) < 2e-4
    end
end
