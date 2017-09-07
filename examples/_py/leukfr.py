"""
LeukFr: Cox regression with random effects

::

    model
    {
    # Set up data
        for(i in 1 : N) {
            for(j in 1 : T) {
    # risk set = 1 if obs.t >= t
                Y[i, j] <- step(obs.t[i] - t[j] + eps)
    # counting process jump = 1 if obs.t in [ t[j], t[j+1] )
    #                      i.e. if t[j] <= obs.t < t[j+1]
                dN[i, j] <- Y[i, j ] *step(t[j+1] - obs.t[i] - eps)*fail[i]
            }
        }
    # Model
        for(j in 1 : T) {
            for(i in 1 : N) {
                dN[i, j]   ~ dpois(Idt[i, j])
                Idt[i, j] <- Y[i, j] * exp(beta * Z[i]+b[pair[i]]) * dL0[j]
            }
            dL0[j] ~ dgamma(mu[j], c)
            mu[j] <- dL0.star[j] * c    # prior mean hazard
    # Survivor function = exp(-Integral{l0(u)du})^exp(beta * z)
            S.treat[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * -0.5))
            S.placebo[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * 0.5))
        }
        for(k in 1 : Npairs) {
            b[k] ~ dnorm(0.0, tau);
        }
        tau ~ dgamma(0.001, 0.001)
        sigma <- sqrt(1 / tau)
        c <- 0.001   r <- 0.1
        for (j in 1 : T) {
            dL0.star[j] <- r * (t[j+1]-t[j])
        }
        beta ~ dnorm(0.0,0.000001)
    }
"""

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dpois_llf, dgamma_llf, step

#  data: N=42, T=17, eps, Npairs, t[Npairs], obs.t[N], pair[2*Npair], fail[T], Z[N]
vars = "N,T,eps,Npairs,t,obs.t,pair,fail,Z".split(',')
_, data = load('../Leukfrdata.txt')
N, T, eps, Npairs, t, obs_t, pair, fail, Z = (data[p] for p in vars)
# init: beta, dL0[T]
pars = "beta,tau,dL0,b".split(',')
_, init = load('../Leukfrinits.txt')
init["b"] = np.zeros(Npairs)
p0, labels = define_pars(init, pars)

# constants
c = 0.001
r = 0.1
dL0_star = r * np.diff(t)
mu = dL0_star * c

def nllf(p):
    beta, tau, dL0, b = p[0], p[1], p[2:2+T], p[2+T:]
    Y = step(obs_t[0:N, None] - t[None, 0:T] + eps)
    dN = Y * step(t[None, 1:T+1] - obs_t[0:N, None] - eps) * fail[0:N, None]
    Idt = Y * exp(beta * Z[0:N, None] + b[pair[0:N, None]-1]) * dL0[None, 0:T]

    cost = 0
    cost += np.sum(dpois_llf(dN, Idt))
    cost += np.sum(dgamma_llf(dL0, mu, c))
    cost += np.sum(dnorm_llf(b, 0.0, tau))
    cost += dgamma_llf(tau, 0.001, 0.001)
    cost += dnorm_llf(beta, 0.0, 0.000001)

    return -cost

S_TREAT = ["S.treat[%d]"%j for j in range(1, T+1)]
S_PLACEBO = ["S.placebo[%d]"%j for j in range(1, T+1)]
def post(p):
    tau = p[1]
    sigma = sqrt(1 / tau)
    #S_placebo = exp(-np.cumsum(dL0, axis=0)) ** exp(beta * 0.5)
    #S_treat = exp(-np.cumsum(dL0, axis=0)) ** exp(beta * -0.5)
    #return np.vstack((sigma, S_placebo, S_treat))
    return [sigma]
post_vars = ["sigma"]
#post_vars = ["sigma"] + S_PLACEBO + S_TREAT

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["beta", "sigma"]


openbugs_result = """
      mean    sd      MC_error  2.5pc   median  97.5pc  start  sample
beta  1.59    0.4361  0.009325  0.7675  1.581   2.472   5001   20000
sigma 0.2106  0.1887  0.01093   0.02831 0.1447  0.7048  5001   20000
"""
