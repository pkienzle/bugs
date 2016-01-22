from bumps.names import *

## Rats model
from math import sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf


_,data = load('../Ratsdata.txt')
N,T = data["N"], data["T"]
x,xbar,Y = data["x"],data["xbar"],data["Y"]

pars =  ('alpha','beta','alpha.c','alpha.tau','beta.c','beta.tau', 'tau.c')
_,init = load('../Ratsinits.txt')
p0, labels = define_pars(init, pars)

def rats(p):
    alpha,beta = p[0:N],p[N:2*N]
    alpha_c, alpha_tau, beta_c, beta_tau, tau_c = p[2*N:2*N+5]
    mu = alpha[:,None] + beta[:,None]*(x[None,:]-xbar)

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


problem = DirectPDF(rats, p0, labels=labels, dof=T*N-len(p0))
# limit tau_c, alpha_tau, beta_tau to [0,inf)
problem._bounds[0,2*N] = 0
problem._bounds[0,2*N+2] = 0
problem._bounds[0,2*N+4] = 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha0", "beta.c", "sigma"]