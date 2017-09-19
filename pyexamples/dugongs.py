"""
Dugongs: nonlinear growth curve

::
    model
    {
        for( i in 1 : N ) {
            Y[i] ~ dnorm(mu[i], tau)
            mu[i] <- alpha - beta * pow(gamma,x[i])
        }
        alpha ~ dflat()T(0,)
        beta ~ dflat()T(0,)
        gamma ~ dunif(0.5, 1.0)
        tau ~ dgamma(0.001, 0.001)
        sigma <- 1 / sqrt(tau)
        U3 <- logit(gamma)
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, logit

#  data: N=27, x[N], Y[N]
_, data = load('../examples/Dugongsdata.txt')
globals().update(data)
# init: alpha, beta, tau, gamma
pars = "alpha,beta,tau,gamma".split(',')
_, init = load('../examples/Dugongsinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):
    alpha, beta, tau, gamma = p

    mu = alpha - beta * gamma**x

    cost = 0
    cost += np.sum(dnorm_llf(Y, mu, tau))
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

def post(p):
    alpha, beta, tau, gamma = p
    U3 = logit(gamma)
    sigma = 1 / sqrt(tau)
    return U3, sigma
post_vars = ["U3", "sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0, 0] = 0 #alpha ~ dflat()T(0,)
problem._bounds[0, 1] = 0 #beta ~ dflat()T(0,)
problem._bounds[:, 3] = 0.5, 1.0 #gamma ~ dunif(0.5, 1.0)
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["U3", "alpha", "beta", "gamma", "sigma"]


openbugs_result = """
      mean    sd      MC_error  2.5pc   median  97.5pc  start sample
U3    1.835   0.2818  0.01639   1.269   1.846   2.38    2001  10000
alpha 2.647   0.0728  0.004263  2.524   2.639   2.816   2001  10000
beta  0.9734  0.07886 0.003896  0.8215  0.9716  1.136   2001  10000
gamma 0.859   0.0344  0.002033  0.7805  0.8636  0.9153  2001  10000
sigma 0.0993  0.01493 2.907E-4  0.07523 0.09749 0.1328  2001  10000
"""
