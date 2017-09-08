"""
Inhaler: ordered categorical data

::

    model
    {
    #
    # Construct individual response data from contingency table
    #
        for (i in 1 : Ncum[1, 1]) {
            group[i] <- 1 for (t in 1 : T) { response[i, t] <- pattern[1, t] }
        }
        for (i in (Ncum[1,1] + 1) : Ncum[1, 2]) {
            group[i] <- 2 for (t in 1 : T) { response[i, t] <- pattern[1, t] }
        }

        for (k in 2  : Npattern) {
            for(i in (Ncum[k - 1, 2] + 1) : Ncum[k, 1]) {
                group[i] <- 1 for (t in 1 : T) { response[i, t] <- pattern[k, t] }
            }
            for(i in (Ncum[k, 1] + 1) : Ncum[k, 2]) {
                group[i] <- 2 for (t in 1 : T) { response[i, t] <- pattern[k, t] }
            }
        }
    #
    # Model
    #
        for (i in 1 : N) {
            for (t in 1 : T) {
                for (j in 1 : Ncut) {
    #
    # Cumulative probability of worse response than j
    #
                    logit(Q[i, t, j]) <- -(a[j] + mu[group[i], t] + b[i])
                }
    #
    # Probability of response = j
    #
                p[i, t, 1] <- 1 - Q[i, t, 1]
                for (j in 2 : Ncut) { p[i, t, j] <- Q[i, t, j - 1] - Q[i, t, j] }
                p[i, t, (Ncut+1)] <- Q[i, t, Ncut]

                response[i, t] ~ dcat(p[i, t, ])
            }
    #
    # Subject (random) effects
    #
            b[i] ~ dnorm(0.0, tau)
    }

    #
    # Fixed effects
    #
        for (g in 1 : G) {
            for(t in 1 : T) {
    # logistic mean for group i in period t
                mu[g, t] <- beta * treat[g, t] / 2 + pi * period[g, t] / 2 + kappa * carry[g, t]
            }
        }
        beta ~ dnorm(0, 1.0E-06)
        pi ~ dnorm(0, 1.0E-06)
        kappa ~ dnorm(0, 1.0E-06)

    # ordered cut points for underlying continuous latent variable
        a[1] ~ dflat()T(-1000, a[2])
        a[2] ~ dflat()T(a[1], a[3])
        a[3] ~ dflat()T(a[2],  1000)

        tau ~ dgamma(0.001, 0.001)
        sigma <- sqrt(1 / tau)
        log.sigma <- log(sigma)

    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

import sys
import numpy as np
from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dcat_llf, ilogit

#  data: N=286, T=2, G=2, Npattern=16, Ncut=3,
#        pattern[Npattern,T], Ncum[Npattern,G],
#        treat[G,T], period[G,T], carry[G,T]
_, data = load('../Inhalersdata.txt')
globals().update(load('../Inhalersdata.txt')[1])

# inits: beta, pi, kappa, a[Ncut], tau
_, init = load('../Inhalersinits.txt')
pars = ["beta", "pi", "kappa", "tau", "a"]
#init["b"] = np.zeros(N); pars.append("b")
p0, labels = define_pars(init, pars)

def pre():
    # Note: change to 0-origin indices for i and k and for group
    group = np.empty(N, 'i')
    response = np.empty((N, T), 'i')
    for i in range(0, Ncum[0, 0]):
        group[i] = 0
        response[i] = pattern[0]
    for i in range(Ncum[0, 0], Ncum[0, 1]):
        group[i] = 1
        response[i] = pattern[1]
    for k in range(1, Npattern):
        for i in range(Ncum[k-1, 1], Ncum[k, 0]):
            group[i] = 0
            response[i] = pattern[k]
        for i in range(Ncum[k, 0], Ncum[k, 1]):
            group[i] = 1
            response[i] = pattern[k]
    return group, response

group, response = pre()

MARGINALIZATION_COUNT = 10
#MARGINALIZATION_COUNT = 1

def nllf(p):
    beta, pi, kappa, tau = p[:4]
    a = p[4:7]
    b = p[7:]

    ## quick rejection of unordered a points
    if not(-1000 <= a[0] and a[0] <= a[1] and a[1] <= a[2] and a[2] <= 1000):
        return inf
    if tau <= 0:
        return inf

    sigma = 1 / sqrt(tau)
    mu = beta * treat/2 + pi * period/2 + kappa * carry
    prob = np.empty((N, T, Ncut+1))

    ## Marginalize over random effects (b[N] ~ N(0, tau))
    cost = 0
    for _ in range(MARGINALIZATION_COUNT):
        b = np.random.normal(0.0, sigma, size=N)
        #cost += np.sum(dnorm_llf(b, 0, tau))

        Q = ilogit(-(a[None, None, :] + mu[group, :, None] + b[:, None, None]))
        prob[:, :, 0] = 1 - Q[:, :, 0]
        for j in range(1, Ncut):
            prob[:, :, j] = Q[:, :, j-1] - Q[:, :, j]
        prob[:, :, -1] = Q[:, :, -1]

        cost += np.sum(dcat_llf(response, prob))
    cost /= MARGINALIZATION_COUNT

    cost += dnorm_llf(beta, 0, 1e-6)
    cost += dnorm_llf(pi, 0, 1e-6)
    cost += dnorm_llf(kappa, 0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)
    ## ordered cut points for underlying continuous latent variable
    #cost += dflat_llf(a[0]) if -1000 <= a[0] <= a[1] else -inf
    #cost += dflat_llf(a[1]) if a[0] <= a[1] <= a[2] else -inf
    #cost += dflat_llf(a[2]) if a[1] <= a[2] <= 1000 else -inf

    ## PAK: model looks over-parameterized: anchor a[1]
    #cost += dnorm_llf(a[0], 0.707, 1/0.1364**2)

    return -cost

def post(p):
    beta, pi, kappa, tau = p[:4]
    a = p[4:7]

    sigma = sqrt(1 / tau)
    log_sigma = log(sigma)

    return [sigma, log_sigma]

post_vars = ["sigma", "log.sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "a[1]", "a[2]", "a[3]", "beta", "kappa", "log.sigma", "pi", "sigma"
]

openbugs_result = """
          mean    sd      MC_error  2.5pc    median  97.5pc   start sample
a[1]      0.707   0.1364  0.003749   0.4554  0.7019   0.9876  2001  20000
a[2]      3.92    0.3322  0.01437    3.313   3.903    4.611   2001  20000
a[3]      5.266   0.4729  0.01681    4.408   5.24     6.267   2001  20000
beta      1.084   0.3252  0.01199    0.4797  1.077    1.735   2001  20000
kappa     0.2321  0.2523  0.00875   -0.2875  0.2341   0.7022  2001  20000
log.sigma 0.1791  0.2205  0.01305   -0.353   0.2041   0.5445  2001  20000
pi       -0.2364  0.1953  0.003197  -0.6112 -0.2387   0.1479  2001  20000
sigma     1.224   0.2537  0.01471    0.7026  1.226    1.724   2001  20000
"""
