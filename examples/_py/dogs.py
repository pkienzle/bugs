"""
Dogs: loglinear model for binary data

::

    model
    {
        for (i in 1 : N) {
            theta[i] ~ dgamma(alpha, beta)
            lambda[i] <- theta[i] * t[i]
            x[i] ~ dpois(lambda[i])
        }
        alpha ~ dexp(1)
        beta ~ dgamma(0.1, 1.0)
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbern_llf

#  data: Dogs=30, Trials=25, Y[Dogs,Trials]
vars = "Dogs,Trials,Y".split(',')
_, data = load('../Dogsdata.txt')
Dogs, Trials, Y = (data[p] for p in vars)
# init: alpha, beta
pars = "alpha,beta".split(',')
_, init = load('../Dogsinits.txt')
p0, labels = define_pars(init, pars)

def pre():
    y = 1 - Y
    xa = np.empty(Y.shape)
    xs = np.empty(Y.shape)
    for i in range(Dogs):
        xa[i, 0] = 0
        xs[i, 0] = 0
        for j in range(1, Trials):
            xa[i, j] = np.sum(Y[i, :j-1])
            xs[i, j] = j - xa[i, j]
    return y, xa, xs

y, xa, xs = pre()

def dogs(p):
    alpha, beta = p
    p = exp(alpha*xa + beta*xs)

    cost = 0
    cost += np.sum(dbern_llf(y, p))

    return -cost

def post(p):
    alpha, beta = p
    A = exp(alpha)
    B = exp(beta)
    return A, B
post_vars = ["A", "B"]

dof = 100
problem = DirectProblem(dogs, p0, labels=labels, dof=dof)

#alpha ~ dflat()T(, -0.00001)
#beta ~ dflat()T(, -0.00001)
problem._bounds[1, :] = -0.00001
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["A", "B", "alpha", "beta"]


openbugs_result = """
       mean    sd       MC_error  2.5pc    median    97.5pc    start sample
A      0.7827  0.01805  7.102E-4   0.7463   0.7832    0.8169   1001  10000
B      0.9248  0.01051  3.419E-4   0.9029   0.9249    0.945    1001  10000
alpha -0.2452  0.02308  9.094E-4  -0.2926  -0.2444   -0.2023   1001  10000
beta  -0.07829 0.01138  3.701E-4  -0.1021  -0.07802  -0.05661  1001  10000
"""