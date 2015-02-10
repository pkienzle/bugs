from bumps.names import *

## Rats model
from math import sqrt
from bumps import bugs
from bumps.bugsmodel import dnorm_llf, dgamma_llf


_,data = bugs.load('../Ratsdata.txt')
N,T = data["N"], data["T"]
x,xbar,Y = data["x"],data["xbar"],data["Y"]

pars =  ('alpha','beta','alpha.c','alpha.tau','beta.c','beta.tau', 'tau.c')
_,init = bugs.load('../Ratsinits.txt')
p0 = np.hstack([init[s] for s in pars])

def rats(p):
    alpha,beta = p[0:N],p[N:2*N]
    alpha_c, alpha_tau, beta_c, beta_tau, tau_c = p[2*N:2*N+5]

    mu = alpha[:,None] + beta[:,None]*(x[None,:]-xbar)
    alpha0 = alpha_c - xbar * beta_c
    sigma = 1/sqrt(tau_c)

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

problem = DirectPDF(rats, len(p0), T*N-len(p0))
# limit tau_c, alpha_tau, beta_tau to [0,inf)
problem._bounds[0,2*N] = 0
problem._bounds[0,2*N+2] = 0
problem._bounds[0,2*N+4] = 0
problem.setp(p0)
