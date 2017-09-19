"""
Mice: Weibull regression

::

    model
    {
        for(i in 1 : M) {
            for(j in 1 : N) {
                t[i, j] ~ dweib(r, mu[i])C(t.cen[i, j],)
            }
            mu[i] <- exp(beta[i])
            beta[i] ~ dnorm(0.0, 0.001)
            median[i] <- pow(log(2) * exp(-beta[i]), 1/r)
        }
        r ~ dexp(0.001)
        veh.control <- beta[2] - beta[1]
        test.sub <- beta[3] - beta[1]
        pos.control <- beta[4] - beta[1]
    }

"""

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dweib_C_llf, dexp_llf

#  data: M=4, N=20, t[M,N], t.cen[M,N]
vars = "t,t.cen,M,N".split(',')
_, data = load('../examples/Micedata.txt')
t, t_cen, M, N = (data[p] for p in vars)
# init: beta[M], r
pars = "beta,r".split(',')
_, init = load('../examples/Miceinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):
    beta, r = p[:M], p[M]
    mu = exp(beta)

    cost = 0
    cost += np.sum(dweib_C_llf(t, r, mu[:, None], lower=t_cen))
    cost += np.sum(dnorm_llf(beta, 0.0, 0.001))
    cost += dexp_llf(r, 0.001)

    return -cost

MEDIAN = ["median[%d]"%k for k in range(1, M+1)]
def post(p):
    beta, r = p[:M], p[M]
    median = (log(2) * exp(-beta))**(1./r)
    veh_control = beta[1] - beta[0]
    test_sub = beta[2] - beta[0]
    pos_control = beta[3] - beta[0]
    return np.vstack((median, veh_control, test_sub, pos_control))
post_vars = MEDIAN + ["veh.control", "test.sub", "pos.control"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0,M] = 0  # r >= 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = MEDIAN + ["pos.control", "r", "test.sub", "veh.control"]


openbugs_result = """
             mean    sd      MC_error  2.5pc    median   97.5pc   start  sample
median[1]    23.77   1.993   0.06917   20.1     23.67    28.03    2001   20000
median[2]    34.91   3.508   0.1201    28.93    34.59    42.7     2001   20000
median[3]    26.63   2.371   0.08281   22.32    26.52    31.63    2001   20000
median[4]    21.32   1.883   0.05613   18.0     21.19    25.4     2001   20000
pos.control   0.3208 0.3435  0.01172   -0.3314   0.3173   0.977   2001   20000
r             2.908  0.2981  0.02056    2.371    2.891    3.538   2001   20000
test.sub     -0.3283 0.3388  0.01176   -0.9825  -0.3333   0.36    2001   20000
veh.control  -1.108  0.3671  0.01373   -1.842   -1.105   -0.3707  2001   20000
"""
