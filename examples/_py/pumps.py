"""
Pumps: conjugate gamma-Poisson hierarchical model

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
from bugs.model import dgamma_llf, dpois_llf, dexp_llf

#  data: N=10, t[N], x[N]
vars = "t,x,N".split(',')
_, data = load('../Pumpsdata.txt')
t, x, N = (data[p] for p in vars)
# init: alpha, beta ;  theta is missing from inits, so assume zeros
pars = "alpha,beta,theta".split(',')
_, init = load('../Pumpsinits.txt')
init['theta'] = np.zeros(N)
p0, labels = define_pars(init, pars)


def pumps(pars):
    alpha, beta, theta = pars[0], pars[1], pars[2:]
    lambda_ = theta * t

    cost = 0
    cost += np.sum(dgamma_llf(theta, alpha, beta))
    cost += np.sum(dpois_llf(x, lambda_))
    cost += dexp_llf(alpha, 1)
    cost += dgamma_llf(beta, 0.1, 1.0)

    return -cost

dof = 1
problem = DirectProblem(pumps, p0, labels=labels, dof=dof)
problem._bounds[0, :] = 0.
problem.setp(p0)


openbugs_result = """
          mean    sd       MC_error   2.5pc    median   97.5pc  start sample
alpha     0.6951  0.2764   0.005396   0.2812   0.6529   1.351   1001  10000
beta      0.9189  0.542    0.01017    0.1775   0.8138   2.265   1001  10000
theta[1]  0.05981 0.02518  2.629E-4   0.02127  0.05621  0.1176  1001  10000
theta[2]  0.1027  0.08174  9.203E-4   0.00808  0.08335  0.3138  1001  10000
theta[3]  0.08916 0.03802  4.144E-4   0.03116  0.08399  0.1798  1001  10000
theta[4]  0.1157  0.0301   3.152E-4   0.06443  0.1128   0.1818  1001  10000
theta[5]  0.5977  0.3124   0.003209   0.1491   0.5426   1.359   1001  10000
theta[6]  0.6104  0.1376   0.00145    0.3726   0.6007   0.9089  1001  10000
theta[7]  0.9035  0.7396   0.007221   0.07521  0.7072   2.844   1001  10000
theta[8]  0.9087  0.7523   0.007056   0.07747  0.7094   2.887   1001  10000
theta[9]  1.583   0.7647   0.007846   0.4667   1.461    3.446   1001  10000
theta[10] 1.984   0.4212   0.004278   1.24     1.953    2.891   1001  10000
"""
