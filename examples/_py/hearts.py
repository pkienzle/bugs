"""
Hearts: a mixture model for count data

::

    model
    {
        for (i in 1 : N) {
            y[i] ~ dbin(P[state1[i]], t[i])
            state[i] ~ dbern(theta)
            state1[i] <- state[i] + 1
            t[i] <- x[i] + y[i]
            prop[i] <- P[state1[i]]
        }
        P[1] <- p
        P[2] <- 0
        logit(p) <- alpha
        alpha ~ dnorm(0,1.0E-4)
        beta <- exp(alpha)
        logit(theta) <- delta
        delta ~ dnorm(0, 1.0E-4)
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbin_llf, dbern_llf, dnorm_llf, dgamma_llf, ilogit

#  data: N=12, x[N], y[N]
_, data = load('../Heartsdata.txt')
globals().update(data)
# init: delta, alpha
_, init = load('../Heartsinits.txt')
init['state'] = np.ones(N)
pars = "delta alpha state".split()
p0, labels = define_pars(init, pars)

t = x + y

def nllf(p):
    delta, alpha, state = p[0], p[1], p[2:]
    beta = exp(alpha)
    theta = ilogit(delta)

    P = np.array([ilogit(alpha), 0])
    state = np.asarray(np.floor(state), 'i')

    #state1 = state + 1  # zero-indexing in numpy
    #prop = P[state] # unused

    cost = 0
    cost += np.sum(dbin_llf(y, P[state], t))
    cost += np.sum(dbern_llf(state, theta))
    cost += dnorm_llf(alpha, 0, 1e-4)
    cost += dnorm_llf(delta, 0, 1e-4)

    return -cost

def post(p):
    delta, alpha, state = p[0], p[1], p[2:]
    beta = exp(alpha)
    theta = ilogit(delta)
    return beta, theta
post_vars = ["beta", "theta"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem._bounds[:, 2:N+2] = [0]*N, [2-1e-10]*N
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha", "beta", "delta", "theta"]


openbugs_result = """
       mean     sd      MC_error  2.5pc    median   97.5pc    start sample
alpha  -0.4787  0.2778  0.006036  -1.029   -0.4806   0.06089  1001  10000
beta    0.6438  0.1809  0.004069   0.3572   0.6184   1.063    1001  10000
delta   0.3191  0.6201  0.00758   -0.8863   0.3114   1.56     1001  10000
theta   0.5726  0.1393  0.001689   0.2919   0.5772   0.8263   1001  10000
"""
