"""
Rats: a normal hierarchical model

::

    model
    {
        for( i in 1 : N ) {
            for( j in 1 : T ) {
                Y[i , j] ~ dnorm(mu[i , j],tau.c)
                mu[i , j] <- alpha[i] + beta[i] * (x[j] - xbar)
            }
            alpha[i] ~ dnorm(alpha.c,alpha.tau)
            beta[i] ~ dnorm(beta.c,beta.tau)
        }
        tau.c ~ dgamma(0.001,0.001)
        sigma <- 1 / sqrt(tau.c)
        alpha.c ~ dnorm(0.0,1.0E-6)
        alpha.tau ~ dgamma(0.001,0.001)
        beta.c ~ dnorm(0.0,1.0E-6)
        beta.tau ~ dgamma(0.001,0.001)
        alpha0 <- alpha.c - xbar * beta.c
    }
"""


from bumps.names import *

## Rats model
from math import sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf


_, data = load('../Ratsdata.txt')
N, T = data["N"], data["T"]
x, xbar, Y = data["x"], data["xbar"], data["Y"]

pars =  'alpha,beta,alpha.c,alpha.tau,beta.c,beta.tau,tau.c'.split(',')
_, init = load('../Ratsinits.txt')
p0, labels = define_pars(init, pars)

def rats(p):
    alpha, beta = p[0:N], p[N:2*N]
    alpha_c, alpha_tau, beta_c, beta_tau, tau_c = p[2*N:2*N+5]
    mu = alpha[:, None] + beta[:, None]*(x[None, :] - xbar)

    cost = 0.
    cost += np.sum(dnorm_llf(Y, mu, tau_c))
    cost += np.sum(dnorm_llf(alpha, alpha_c, alpha_tau))
    cost += np.sum(dnorm_llf(beta, beta_c, beta_tau))
    cost += dgamma_llf(tau_c, 0.001, 0.001)
    cost += dnorm_llf(alpha_c, 0.0, 1e-6)
    cost += dgamma_llf(alpha_tau, 0.001, 0.001)
    cost += dnorm_llf(beta_c, 0.0, 1e-6)
    cost += dgamma_llf(beta_tau, 0.001, 0.001)
    return -cost

def post(p):
    alpha_c, beta_c, tau_c = p[2*N], p[2*N+2], p[2*N+4]
    alpha0 = alpha_c - xbar * beta_c
    sigma = 1./np.sqrt(tau_c)
    return alpha0, sigma
post_vars = ["alpha0", "sigma"]


problem = DirectProblem(rats, p0, labels=labels, dof=T*N-len(p0))
# limit tau_c, alpha_tau, beta_tau to [0,inf)
problem._bounds[0,2*N] = 0
problem._bounds[0,2*N+2] = 0
problem._bounds[0,2*N+4] = 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha0", "beta.c", "sigma"]


openbugs_result = """
        mean    sd      MC_error   2.5pc   median  97.5pc    start sample
alpha0  106.6   3.666   0.04102    99.29   106.6    113.7    1001  10000
beta.c    6.186 0.1088  0.001316    5.971    6.187    6.398  1001  10000
sigma     6.092 0.4672  0.007633    5.254    6.06     7.095  1001  10000
"""
