"""
Eyes: Normal Mixture Model

::

    model
    {
        for( i in 1 : N ) {
            y[i] ~ dnorm(mu[i], tau)
            mu[i] <- lambda[T[i]]
            T[i] ~ dcat(P[])
        }
        P[1:2] ~ ddirich(alpha[])
        theta ~ dunif(0.0, 1000)
        lambda[2] <- lambda[1] + theta
        lambda[1] ~ dnorm(0.0, 1.0E-6)
        tau ~ dgamma(0.001, 0.001) sigma <- 1 / sqrt(tau)
    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dunif_llf, ddirich_llf, dcat_llf

#  data: N=48, y[N], T[N], alpha[2]
_, data = load('../Eyesdata.txt')
globals().update(data)
# init: lambda[2], theta, tau
pars = "lambda[1] P[1] theta tau T".split()
_, init = load('../Eyesinits.txt')
init["lambda[1]"] = init["lambda"][0]
init["T"] = np.zeros(N-2)+2
init["P[1]"] = 0.5
p0, labels = define_pars(init, pars)

def nllf(p):
    lambda1, P1, theta, tau = p[:4]
    T[1:-1] = p[4:4+N-2]
    lambda2 = lambda1 + theta
    P2 = 1 - P1  # P must sum to 1 for dirichlet
    P = np.array([P1, P2])
    T_int = np.asarray(np.floor(T), 'i')
    mu = np.array([lambda1, lambda2])[T_int-1]

    cost = 0
    cost += np.sum(dnorm_llf(y, mu, tau))
    cost += np.sum(dcat_llf(T_int, P))
    cost += ddirich_llf(P, alpha)
    cost += dunif_llf(theta, 0, 1000)
    cost += dnorm_llf(lambda1, 0.001, 0.001)
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

def post(p):
    lambda1, P1, theta, tau = p[:4]
    lambda2 = lambda1 + theta
    P2 = 1 - P1
    sigma = 1 / sqrt(tau)
    return lambda2, P2, sigma
post_vars = ["lambda[2]", "P[2]", "sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[:, 1] = [0, 1]  # P[1] is a probability
problem._bounds[:, 2] = [0, 1000] # theta ~ dunif(0, 1000)
problem._bounds[:, 4:] = [1]*(N-2), [3-1e-10]*(N-2) # T is in 1,2
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["P[1]", "P[2]", "lambda[1]", "lambda[2]", "sigma"]


openbugs_result = """
          mean     sd      MC_error 2.5pc    median   97.5pc    start  sample
P[1]        0.5973 0.08633 0.001912   0.4219   0.6014   0.7562  1001   10000
P[2]        0.4027 0.08633 0.001912   0.2439   0.3986   0.5783  1001   10000
lambda[1] 536.7    0.9031  0.02074  535.0    536.7    538.5     1001   10000
lambda[2] 548.8    1.254   0.03458  546.3    548.9    551.2     1001   10000
sigma       3.768  0.6215  0.02004    2.929    3.662    5.403   1001   10000
"""
