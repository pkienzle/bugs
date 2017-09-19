"""
Orange Trees: Non-linear growth curve

::

    model {
        for (i in 1:K) {
            for (j in 1:n) {
                Y[i, j] ~ dnorm(eta[i, j], tauC)
                eta[i, j] <- phi[i, 1] / (1 + phi[i, 2] * exp(phi[i, 3] * x[j]))
            }
            phi[i, 1] <- exp(theta[i, 1])
            phi[i, 2] <- exp(theta[i, 2]) - 1
            phi[i, 3] <- -exp(theta[i, 3])
            for (k in 1:3) {
                theta[i, k] ~ dnorm(mu[k], tau[k])
            }
        }
        tauC ~ dgamma(1.0E-3, 1.0E-3)
        sigmaC <- 1 / sqrt(tauC)
        varC <- 1 / tauC
        for (k in 1:3) {
            mu[k] ~ dnorm(0, 1.0E-4)
            tau[k] ~ dgamma(1.0E-3, 1.0E-3)
            sigma[k] <- 1 / sqrt(tau[k])
        }
    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from numpy import expm1
from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf

#  data: n=7, K=5, x[N], Y[K,n]
_, data = load('../examples/Otreesdata.txt')
globals().update(data)
# init: theta[K,3], mu[3], tau[3], tauC
pars = "theta,mu,tau,tauC".split(',')
_, init = load('../examples/Otreesinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):
    theta = p[:K*3].reshape(K, 3)
    mu, tau = p[K*3:K*3+3], p[K*3+3:K*3+3+3]
    tauC = p[K*3+3+3]

    phi0, phi1, phi2 = exp(theta[:, 0]), expm1(theta[:, 1]), -exp(theta[:, 2])
    eta = phi0[:, None] / (1 + phi1[:, None] * exp(phi2[:, None] * x[None, :]))

    cost = 0
    cost += np.sum(dnorm_llf(Y, eta, tauC))
    cost += np.sum(dnorm_llf(theta, mu[None, :], tau[None, :]))
    cost += dgamma_llf(tauC, 0.001, 0.001)
    cost += np.sum(dnorm_llf(mu, 0, 1e-4))
    cost += np.sum(dgamma_llf(tau, 1e-3, 1e-3))

    return -cost

def post(p):
    #theta = p[:K*3].reshape(K, 3)
    mu, tau = p[K*3:K*3+3], p[K*3+3:K*3+3+3]
    tauC = p[K*3+3+3]
    sigma_C = 1 / sqrt(tauC)
    varC = 1 / tauC
    sigma = 1 / sqrt(tau)
    return np.vstack((sigma_C, varC, sigma))
post_vars = ["sigma.C", "varC", "sigma[1]", "sigma[2]", "sigma[3]"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "mu[1]", "mu[2]", "mu[3]",
    "sigma[1]", "sigma[2]", "sigma[3]", "sigma.C"]


openbugs_result = """
          mean     sd      MC_error 2.5pc    median  97.5pc    start   sample
mu[1]     5.253    0.1244  0.00237   5.007    5.254    5.505   15001   20000
mu[2]     2.218    0.1137  0.005027  1.999    2.227    2.433   15001   20000
mu[3]    -5.858    0.09592 0.004643 -6.048   -5.857   -5.686   15001   20000
sigma[1]  0.2359   0.1239  0.001806  0.1023   0.2054   0.5469  15001   20000
sigma[2]  0.1245   0.1126  0.004228  0.02446  0.0937   0.4044  15001   20000
sigma[3]  0.09705  0.08283 0.002275  0.02602  0.07677  0.2851  15001   20000
sigma.C   7.892    1.175   0.03214   5.977    7.76    10.57    15001   20000
"""
