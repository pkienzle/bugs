"""
Salm: extra - Poisson variation in dose - response study

::

    model
    {
        for( i in 1 : doses ) {
            for( j in 1 : plates ) {
                y[i , j] ~ dpois(mu[i , j])
                log(mu[i , j]) <- alpha + beta * log(x[i] + 10) +
                    gamma * x[i] + lambda[i , j]
                lambda[i , j] ~ dnorm(0.0, tau)
            }
        }
        alpha ~ dnorm(0.0,1.0E-6)
        beta ~ dnorm(0.0,1.0E-6)
        gamma ~ dnorm(0.0,1.0E-6)
        tau ~ dgamma(0.001, 0.001)
        sigma <- 1 / sqrt(tau)
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dpois_llf

#  data: doses, plates, y[doses,plates], x[doses]
vars = "doses,plates,y,x".split(',')
_, data = load('../Salmdata.txt')
doses, plates, y, x = (data[p] for p in vars)
# init: alpha, beta, gamma, tau
pars = "alpha,beta,gamma,tau,lambda".split(',')
_, init = load('../Salminits.txt')
# lambda not initialized; default to zero
init["lambda"] = np.zeros((doses, plates))
p0, labels = define_pars(init, pars)

def salm(p):
    alpha, beta, gamma, tau = p[:4]
    lambda_ = p[4:].reshape(doses, plates)

    mu = exp(alpha + beta*log(x[:, None]+10) + gamma*x[:, None] + lambda_)

    cost = 0
    cost += np.sum(dpois_llf(y, mu))
    cost += np.sum(dnorm_llf(lambda_, 0.0, tau))
    cost += dnorm_llf(alpha, 0.0, 1e-6)
    cost += dnorm_llf(beta, 0.0, 1e-6)
    cost += dnorm_llf(gamma, 0.0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

def post(p):
    alpha, beta, gamma, tau = p[:4]
    sigma = 1 / sqrt(tau)
    return [sigma]
post_vars = ["sigma"]

dof = 100
problem = DirectProblem(salm, p0, labels=labels, dof=dof)

problem._bounds[0, 3] = 0  # tau bounded below by 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha", "beta", "gamma", "sigma"]

openbugs_result = """
       mean      sd       MC_error  2.5pc    median    97.5pc    start  sample
alpha   2.174    0.3576   0.009239   1.455    2.173     2.872    1001   10000
beta    0.3096   0.09808  0.002719   0.1193   0.3095    0.5037   1001   10000
gamma  -9.587E-4 4.366E-4 1.194E-5  -0.00182 -9.565E-4 -1.194E-4 1001   10000
sigma   0.2576   0.07933  0.001941   0.1283   0.2491    0.4357   1001   10000
"""