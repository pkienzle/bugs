"""
Surgical: Institutional ranking - random effects analysis

::

    model
    {
        for( i in 1 : N ) {
            b[i] ~ dnorm(mu,tau)
            r[i] ~ dbin(p[i],n[i])
            logit(p[i]) <- b[i]
        }
        pop.mean <- exp(mu) / (1 + exp(mu))
        mu ~ dnorm(0.0,1.0E-6)
        sigma <- 1 / sqrt(tau)
        tau ~ dgamma(0.001,0.001)
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbin_llf, dnorm_llf, dgamma_llf, ilogit

#  data: N=12, n[N], r[N]
vars = "n,r,N".split(',')
_, data = load('../Surgicalranddata.txt')
n, r, N = (data[p] for p in vars)
# init: p[N]
pars = "b,tau,mu".split(',')
_, init = load('../Surgicalrandinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):
    b, tau, mu = p[:-2], p[-2], p[-1]

    p = ilogit(b)

    cost = 0
    cost += np.sum(dnorm_llf(b, mu, tau))
    cost += np.sum(dbin_llf(r, p, n))
    cost += dnorm_llf(mu, 0.0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

PVARS = ["p[%d]"%i for i in range(1, N+1)]
def post(p):
    b, tau, mu = p[:-2], p[-2], p[-1]
    p = ilogit(b)
    pop_mean = ilogit(mu)
    sigma = 1.0 / sqrt(tau)
    return np.vstack((p, pop_mean, sigma))
post_vars = PVARS + ["pop.mean", "sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem._bounds[0, -2] = 0.0  # tau >= 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["mu"] + post_vars

openbugs_result = """
        mean      sd       MC_error  2.5pc     median    97.5pc   start sample
mu      -2.558    0.1535   0.002212  -2.887    -2.551    -2.27    1001  10000
p[1]     0.0528   0.01961  3.693E-4   0.01808   0.05184   0.09351 1001  10000
p[2]     0.103    0.0218   3.087E-4   0.06661   0.1007    0.1514  1001  10000
p[3]     0.07071  0.01761  1.967E-4   0.03945   0.06959   0.1094  1001  10000
p[4]     0.05925  0.007902 1.004E-4   0.04465   0.05888   0.07572 1001  10000
p[5]     0.05147  0.01334  2.515E-4   0.02754   0.05074   0.07966 1001  10000
p[6]     0.06915  0.01483  1.696E-4   0.04261   0.06857   0.1008  1001  10000
p[7]     0.06683  0.01586  1.978E-4   0.03815   0.06577   0.1008  1001  10000
p[8]     0.1237   0.02263  4.045E-4   0.08425   0.1222    0.1715  1001  10000
p[9]     0.06967  0.0145   1.629E-4   0.04397   0.06881   0.1003  1001  10000
p[10]    0.07849  0.02007  2.362E-4   0.04508   0.07691   0.1236  1001  10000
p[11]    0.1022   0.01769  2.439E-4   0.07143   0.1009    0.1407  1001  10000
p[12]    0.06865  0.01173  1.419E-4   0.04703   0.06816   0.093   1001  10000
pop.mean 0.07258  0.01016  1.443E-4   0.05282   0.07234   0.09361 1001  10000
sigma    0.4077   0.1611   0.004062   0.1664    0.3835    0.7937  1001  10000
"""
