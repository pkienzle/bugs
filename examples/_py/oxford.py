"""
Oxford: smooth fit to log-odds ratios

::

    model
    {
        for (i in 1 : K) {
            r0[i]  ~ dbin(p0[i], n0[i])
            r1[i] ~ dbin(p1[i], n1[i])
            logit(p0[i]) <- mu[i]
            logit(p1[i]) <- mu[i] + logPsi[i]
            logPsi[i]    <- alpha + beta1 * year[i] + beta2 * (year[i] * year[i] - 22) + b[i]
            b[i] ~ dnorm(0, tau)
            mu[i]  ~ dnorm(0.0, 1.0E-6)
        }j
        alpha  ~ dnorm(0.0, 1.0E-6)
        beta1  ~ dnorm(0.0, 1.0E-6)
        beta2  ~ dnorm(0.0, 1.0E-6)
        tau    ~ dgamma(1.0E-3, 1.0E-3)
        sigma <- 1 / sqrt(tau)
    }
"""

#raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dbin_llf, ilogit

#  data: K=120, r1[K], n1[K], r0[K], n0[K], year[K]
vars = "r1,n1,r0,n0,year,K".split(',')
_, data = load('../Oxforddata.txt')
r1, n1, r0, n0, year, K = (data[p] for p in vars)
# init: alpha, beta1, beta2, tau, mu[K], b[K]
pars = "alpha,beta1,beta2,tau,mu,b".split(',')
_, init = load('../Oxfordinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):
    alpha, beta1, beta2, tau = p[:4]
    mu, b = p[4:4+K], p[4+K:]

    # median values from fit
    #alpha, beta1, beta2, tau = 0.5793, -0.0457, 0.007004, 1/0.08059**2

    logPsi = alpha + beta1*year + beta2*(year*year - 22) + b
    p0 = ilogit(mu)
    p1 = ilogit(mu + logPsi)

    cost = 0
    cost += np.sum(dbin_llf(r0, p0, n0))
    cost += np.sum(dbin_llf(r1, p1, n1))
    cost += np.sum(dnorm_llf(b, 0, tau))
    cost += np.sum(dnorm_llf(mu, 0, 1e-6))
    cost += dnorm_llf(alpha, 0, 1e-6)
    cost += dnorm_llf(beta1, 0, 1e-6)
    cost += dnorm_llf(beta2, 0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

def post(p):
    alpha, beta1, beta2, tau = p[:4]
    sigma = 1 / sqrt(tau)
    #sigma = 0.08059*np.ones_like(tau)
    return [sigma]
post_vars = ["sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

#problem._bounds[0, 3] = 0  # tau bounded below by 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha", "beta1", "beta2", "sigma"]

openbugs_result = """
       mean     sd       MC_error  2.5pc    median    97.5pc    start  sample
alpha  0.5817   0.06228  0.001469   0.459    0.5813    0.7053   2001   10000
beta1 -0.04654  0.01526  4.205E-4  -0.07656 -0.04668  -0.01708  2001   10000
beta2  0.007115 0.003034 7.765E-5   0.0013   0.007114  0.0131   2001   10000
sigma  0.1078   0.06774  0.005011   0.02571  0.08953   0.2693   2001   10000
"""
