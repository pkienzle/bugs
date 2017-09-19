"""
Blocker: random effects meta-analysis of clinical trials

::

    model
    {
        for( i in 1 : Num ) {
            rc[i] ~ dbin(pc[i], nc[i])
            rt[i] ~ dbin(pt[i], nt[i])
            logit(pc[i]) <- mu[i]
            logit(pt[i]) <- mu[i] + delta[i]
            mu[i] ~ dnorm(0.0,1.0E-5)
            delta[i] ~ dt(d, tau, 4)
        }
        d ~ dnorm(0.0,1.0E-6)
        tau ~ dgamma(0.001,0.001)
        delta.new ~ dt(d, tau, 4)
        sigma <- 1 / sqrt(tau)
    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbin_llf, dt_llf, dgamma_llf, dnorm_llf, ilogit

#  data: Num=22, rt[Num], nt[Num], rc[Num], nc[Num]
vars = "rt,nt,rc,nc,Num".split(',')
_, data = load('../examples/Blockersdata.txt')
rt, nt, rc, nc, Num = (data[p] for p in vars)
# init: d, delta.new, tau, mu[Num], delta[Num]
pars = "d,delta.new,tau,mu,delta".split(',')
_, init = load('../examples/Blockersinits1.txt')
p0, labels = define_pars(init, pars)

def nllf(pars):
    d, delta_new, tau = pars[0:3]
    mu, delta = pars[3:3+Num], pars[3+Num:3+2*Num]

    pc = ilogit(mu)
    pt = ilogit(mu + delta)

    cost = 0
    cost += np.sum(dbin_llf(rc, pc, nc))
    cost += np.sum(dbin_llf(rt, pt, nt))
    cost += np.sum(dnorm_llf(mu, 0, 1e-5))
    cost += np.sum(dt_llf(delta, d, tau, 4))
    cost += dnorm_llf(d, 0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)
    cost += dt_llf(delta_new, d, tau, 4)

    return -cost

def post(pars):
    tau = pars[2]
    sigma = np.sqrt(1/tau)
    return [sigma]
post_vars = ["sigma"]

dof = 1
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem._bounds[0, 2] = 0  # tau in [0, inf]
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["d", "delta.new", "sigma"]

openbugs_result = """
          mean     sd       MC_error  2.5pc    median    97.5pc   start sample
d         -0.2536  0.06164  0.002288  -0.3718  -0.2545   -0.1266  1001  10000
delta.new -0.2536  0.1417   0.00249   -0.5384  -0.2571    0.04796 1001  10000
sigma      0.1114  0.06381  0.003067   0.02749  0.09871   0.2634  1001  10000
"""
