#model
#{
#
##  PRIORS
#	alpha[1] <- 0;       # zero contrast for baseline food
#	for (k in 2 : K) {
#		alpha[k] ~ dnorm(0, 0.00001) # vague priors
#	}
## Loop around lakes:
#	for (k in 1 : K){
#		beta[1, k] <- 0
#	} # corner-point contrast with first lake
#	for (i in 2 : I) {
#		beta[i, 1] <- 0 ;  # zero contrast for baseline food
#		for (k in 2 : K){
#			beta[i, k] ~ dnorm(0, 0.00001) # vague priors
#		}
#	}
## Loop around sizes:
#	for (k in 1 : K){
#		gamma[1, k] <- 0 # corner-point contrast with first size
#	}
#	for (j in 2 : J) {
#		gamma[j, 1] <- 0 ;  # zero contrast for baseline food
#		for ( k in 2 : K){
#			gamma[j, k] ~ dnorm(0, 0.00001) # vague priors
#		}
#	}
#
## LIKELIHOOD
#	for (i in 1 : I) {     # loop around lakes
#		for (j in 1 : J) {     # loop around sizes
#
## Multinomial response
##     X[i,j,1 : K] ~ dmulti( p[i, j, 1 : K] , n[i, j]  )
##     n[i, j] <- sum(X[i, j, ])
##     for (k in 1 : K) {     # loop around foods
##        p[i, j, k]        <- phi[i, j, k] / sum(phi[i, j, ])
##        log(phi[i ,j, k]) <- alpha[k] + beta[i, k]  + gamma[j, k]
##       }
#
## Fit standard Poisson regressions relative to baseline
#			lambda[i, j] ~ dnorm(0, 0.00001) # vague priors
#			for (k in 1 : K) {     # loop around foods
#				X[i, j, k] ~ dpois(mu[i, j, k])
#				log(mu[i, j, k]) <- lambda[i, j] + alpha[k] + beta[i, k]  + gamma[j, k]
#			}
#		}
#	}
#
## TRANSFORM OUTPUT TO ENABLE COMPARISON
##  WITH AGRESTI'S RESULTS
#
#	for (k in 1 : K) {     # loop around foods
#		for (i in 1 : I) {     # loop around lakes
#			b[i, k] <- beta[i, k] - mean(beta[, k]);   # sum to zero constraint
#		}
#		for (j in 1 : J) {     # loop around sizes
#			g[j, k] <- gamma[j, k] - mean(gamma[, k]); # sum to zero constraint
#		}
#	}
#}

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dpois_llf, dmulti_llf

#  data: I, J, K, X[I, J, K]
vars = "I,J,K,X".split(',')
_, data = load('../Aligatorsdata.txt')
I, J, K, X = (data[p] for p in vars)
# init: alpha[K], beta[I, K], gamma[J, K], lambda[I, J]
pars = "alpha,beta,gamma,lambda".split(',')
#_, init = load('../Aligatorsinits.txt')
_, init = load('../Aligatorsinits1.txt')
p0, labels = define_pars(init, pars)
active_index = ~np.isnan(p0)
labels = [label for label, active in zip(labels, active_index) if active]

def aligators(active_pars):
    pars = p0.copy()
    pars[active_index] = active_pars
    alpha = pars[0:K]
    beta = pars[K:K+I*K].reshape(I, K)
    gamma = pars[K+I*K:K+I*K+J*K].reshape(J, K)
    Lambda = pars[K+I*K+J*K:K+I*K+J*K+I*J].reshape(I, J)

    cost = 0
    alpha[0] = 0 # zero contrast for baseline food
    cost += np.sum(dnorm_llf(alpha[1:], 0, 0.00001)) # vague priors

    # Loop around lakes:
    beta[0, :] = 0 # corner-point contrast with first lake
    beta[:, 0] = 0 # zero contrast for baseline food
    cost += np.sum(dnorm_llf(beta[1:, 1:], 0, 0.00001)) # vague priors

    # Loop around sizes:
    gamma[0, :] = 0 # corner-point contrast with first size
    gamma[:, 0] = 0 # zero contrast for baseline food
    cost += np.sum(dnorm_llf(gamma[1:, 1:], 0, 0.00001)) # vague priors

    if 0:
        # Multinomial response
        n = np.sum(X, axis=2)
        phi = np.exp(alpha[None, None, :] + beta[:, None, :] + gamma[None, :, :])
        p = phi/np.sum(phi, axis=2)
        cost += np.sum(dmulti_llf(X, p, n))
    else:
        # Fit standard Poisson regressions relative to baseline
        cost += np.sum(dnorm_llf(Lambda, 0, 0.00001)) # vague priors
        mu = np.exp(Lambda[:, :, None] + alpha[None, None, :]
                    + beta[:, None, :] + gamma[None, :, :])
        cost += np.sum(dpois_llf(X, mu))

    return -cost

def post(active_pars):
    samples = active_pars.shape[1]
    # TRANSFORM OUTPUT TO ENABLE COMPARISON
    #  WITH AGRESTI'S RESULTS
    pars = np.tile(p0, (samples, 1)).T
    pars[active_index] = active_pars
    beta = pars[K:K+I*K, :].reshape(I, K, samples)
    gamma = pars[K+I*K:K+I*K+J*K, :].reshape(J, K, samples)
    b = beta - np.mean(beta, axis=0) # sum to zero constraint
    g = gamma - np.mean(gamma, axis=0) # sum to zero constraint
    b = b.reshape(I*K, samples)
    g = g.reshape(J*K, samples)
    newvars = [bk for bk in b]+[gk for gk in g]
    return newvars
bvars = ["b[%d, %d]"%(i+1, k+1) for i in range(I) for k in range(K)]
gvars = ["g[%d, %d]"%(j+1, k+1) for j in range(J) for k in range(K)]
post_vars = bvars + gvars

num_pars = np.sum(active_index)
dof = I*J*K-num_pars
problem = DirectProblem(aligators, p0[active_index], labels=labels, dof=dof)
problem.setp(p0[active_index])
problem.derive_vars = post, post_vars
bvars = ["b[%d, %d]"%(i+1, k+2) for i in range(4) for k in range(4)]
gvars = ["g[%d, %d]"%(i+1, k+2) for i in range(2) for k in range(4)]
problem.visible_vars = bvars + gvars
#problem.visible_vars = ["b[1, 2]", "g[1, 2]"]

#	mean	sd	median	2.5pc	97.5pc
#b[1,2]	-1.83	0.4312	-1.804	-2.732	-1.056
#b[1,3]	-0.3825	0.6152	-0.3781	-1.64	 0.8085
#b[1,4]	 0.5577	0.5673	 0.5483	-0.5421	 1.719
#b[1,5]	 0.2742	0.3523	 0.2684	-0.4085	 0.9755
#b[2,2]	 0.8812	0.3331	 0.8787	 0.2306	 1.541
#b[2,3]	 0.9578	0.5207	 0.9476	-0.0313	 2.007
#b[2,4]	-1.263	1.02	-1.124	-3.685	 0.3479
#b[2,5]	-0.6589	0.5433	-0.6336	-1.811	 0.3244
#b[3,2]	 1.054	0.3447	 1.05	 0.405	 1.747
#b[3,3]	 1.445	0.517	 1.423	 0.5002	 2.538
#b[3,4]	 0.9338	0.601	 0.923	-0.2385	 2.134
#b[3,5]	 0.9858	0.3955	 0.9842	 0.2053	 1.769
#b[4,2]	-0.1053	0.2952	-0.1062	-0.6902	 0.4788
#b[4,3]	-2.02	0.9853	-1.877	-4.369	-0.5193
#b[4,4]	-0.2281	0.6148	-0.2198	-1.476	 0.9799
#b[4,5]	-0.6012	0.4096	-0.5953	-1.446	 0.1813
#g[1,2]	 0.759	0.2064	 0.7557	 0.3658	 1.192
#g[1,3]	-0.1946	0.3016	-0.1849	-0.8038	 0.3788
#g[1,4]	-0.3324	0.3422	-0.3288	-1.037	 0.3353
#g[1,5]	 0.1775	0.2322	 0.1759	-0.273	 0.6328
#g[2,2]	-0.759	0.2064	-0.7557	-1.191	-0.3654
#g[2,3]	 0.1946	0.3016	 0.1851	-0.3787	 0.8039
#g[2,4]	 0.3324	0.3422	 0.3288	-0.3337	 1.037
#g[2,5]	-0.1775	0.2322	-0.1759	-0.6319	 0.2731
