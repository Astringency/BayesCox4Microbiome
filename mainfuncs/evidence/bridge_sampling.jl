# Optimal bridge sampling (Meng & Wong, 1996; Gronau et al., 2017).
# https://doi.org/10.1016/j.jmp.2017.09.005
_bridge_logistic(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
function _bridge_logmeanexp(x)
    m = maximum(x)
    isfinite(m) || throw(ArgumentError("no finite bridge contributions"))
    return m + log(mean(exp.(x .- m)))
end

"""Optimal bridge ratio for equal-sized samples from target and proposal.
Arguments contain log(target kernel / normalized proposal) at both samples.
"""
function bridge_logratio(at_target, at_proposal; tolerance=1e-10, maxiter=200)
    length(at_target) == length(at_proposal) > 1 || throw(ArgumentError("bridge samples must have equal sizes > 1"))
    all(isfinite, at_target) || throw(ArgumentError("nonfinite density at target draws"))
    all(x -> isfinite(x) || x == -Inf, at_proposal) || throw(ArgumentError("invalid proposal density ratios"))
    finite_proposal = filter(isfinite, at_proposal)
    isempty(finite_proposal) && throw(ArgumentError("proposal has no overlap with target"))
    lo = min(minimum(at_target), minimum(finite_proposal)) - 40
    hi = max(maximum(at_target), maximum(finite_proposal)) + 40
    for iteration in 1:maxiter
        z = (lo + hi) / 2
        f = mean(_bridge_logistic(x-z) for x in at_proposal) -
            mean(_bridge_logistic(z-x) for x in at_target)
        f > 0 ? (lo = z) : (hi = z)
        hi-lo <= tolerance && return (log_normalizer=(lo+hi)/2, iterations=iteration)
    end
    error("bridge ratio did not converge in $maxiter iterations")
end

# A normalized joint Gaussian/Bernoulli proposal also supports discrete gamma.
# Smoothing keeps support on unobserved indicator patterns.
function _bridge_proposal(x, bits)
    size(x,1) > size(x,2)+1 || throw(ArgumentError("too few training draws for bridge covariance"))
    C = Matrix(cov(x))
    ridge = max.(diag(C).*1e-6, 1e-12)
    q = MvNormal(vec(mean(x; dims=1)), Symmetric(C+Diagonal(ridge)))
    probabilities = (vec(sum(bits; dims=1)) .+ 0.5) ./ (size(bits,1)+1)
    return (; continuous=q, probabilities)
end
function _bridge_proposal_logpdf(q, x, bits)
    lp = logpdf(q.continuous, x)
    for j in eachindex(bits)
        lp += bits[j] ? log(q.probabilities[j]) : log1p(-q.probabilities[j])
    end
    return lp
end

# Nonoverlapping batch means retain within-chain dependence. A constant bridge
# weight has zero empirical variance; this is unrelated to constant gamma MCSE.
function _bridge_mean_variance(values, labels)
    variance = 0.0
    for label in unique(labels)
        x = values[labels .== label]
        length(x) >= 16 || throw(ArgumentError("bridge MCSE needs at least 16 evaluation draws per chain"))
        batch_length = max(2, floor(Int, sqrt(length(x))))
        batches = length(x) ÷ batch_length
        means = [mean(view(x, (i-1)*batch_length+1:i*batch_length)) for i in 1:batches]
        variance += (length(x)/length(values))^2 * var(means)/batches
    end
    return variance
end

"""
    bridge_sampling(logtarget, draws; indicators, chain_labels, seed)

Estimate the integral of an unnormalized continuous/discrete target density.
Within each chain, the first half trains a normalized Gaussian/Bernoulli proposal
and the second half estimates the bridge ratio; proposal fitting and evaluation
use disjoint draws. Input rows must be in temporal order within each chain.
The reported MCSE conditions on the fitted proposal and uses batch means for
MCMC contributions. Inspect overlap and independent repeated estimates too.
"""
function bridge_sampling(logtarget, draws::AbstractMatrix;
    indicators=falses(size(draws,1),0), chain_labels=ones(Int,size(draws,1)), seed=1)
    n,d = size(draws)
    size(indicators,1)==length(chain_labels)==n || throw(DimensionMismatch("bridge draw labels"))
    all(isfinite,draws) || throw(ArgumentError("nonfinite bridge draws"))
    all(x -> x == 0 || x == 1, indicators) || throw(ArgumentError("indicators must be binary"))
    bits = Bool.(indicators)
    training, evaluation = Int[], Int[]
    for label in unique(chain_labels)
        ids = findall(==(label), chain_labels); middle = length(ids) ÷ 2
        append!(training, ids[1:middle]); append!(evaluation, ids[middle+1:end])
    end
    q = _bridge_proposal(draws[training,:],bits[training,:])
    rng = MersenneTwister(seed)
    m = length(evaluation)
    proposal = permutedims(rand(rng, q.continuous, m))
    proposal_bits = rand(rng,m,size(bits,2)) .< permutedims(q.probabilities)
    at_target = [logtarget(view(draws,i,:),view(bits,i,:)) -
        _bridge_proposal_logpdf(q,view(draws,i,:),view(bits,i,:)) for i in evaluation]
    at_proposal = [logtarget(view(proposal,i,:),view(proposal_bits,i,:)) -
        _bridge_proposal_logpdf(q,view(proposal,i,:),view(proposal_bits,i,:)) for i in 1:m]
    solution = bridge_logratio(at_target,at_proposal)
    z = solution.log_normalizer
    wt = _bridge_logistic.(z .- at_target)
    wq = _bridge_logistic.(at_proposal .- z)
    derivative = mean(wt.*(1 .- wt)) + mean(wq.*(1 .- wq))
    derivative > 0 || error("no numerical overlap in bridge estimate")
    mcse = sqrt(_bridge_mean_variance(wt,chain_labels[evaluation]) + var(wq)/m) / derivative
    overlap = mean(wt)+mean(wq)
    ess_target = sum(wt)^2/sum(abs2,wt)
    ess_proposal = sum(wq)^2/sum(abs2,wq)
    return (; solution..., log_normalizer_mcse=mcse, overlap,
        contribution_ess_target=ess_target, contribution_ess_proposal=ess_proposal,
        n_training=length(training), n_evaluation=m,
        numerical_screen=overlap>=0.01 && min(ess_target,ess_proposal)>=100 && mcse<=0.1)
end

"""Subtract the actual prior's log normalizer from the posterior-kernel integral.
This makes evidence invariant to arbitrary multiplicative constants in a prior
kernel, including method-specific zero-sum and Moreau constants.
"""
function bridge_evidence(logprior, loglikelihood, posterior, prior; seed=1)
    p = bridge_sampling((u,g)->logprior(u,g)+loglikelihood(u,g),posterior.draws;
        indicators=posterior.indicators,chain_labels=posterior.chain_labels,seed)
    q = bridge_sampling(logprior,prior.draws;
        indicators=prior.indicators,chain_labels=prior.chain_labels,seed=seed+1)
    return (logZ=p.log_normalizer-q.log_normalizer,
        logZ_mcse=hypot(p.log_normalizer_mcse,q.log_normalizer_mcse),
        prior_logZ=q.log_normalizer,posterior_bridge=p,prior_bridge=q,
        numerical_screen=p.numerical_screen && q.numerical_screen)
end
