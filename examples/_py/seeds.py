"""
Seeds: Random effect logistic regression

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
from bugs.model import dbin_llf, dnorm_llf, dgamma_llf, ilogit

# data: N=21, r[N], n[N], x1[N], x2[N]
vars = "r,n,x1,x2,N".split(',')
_, data = load('../Seedsdata.txt')
r, n, x1, x2, N = (data[p] for p in vars)
# init: alpha0, alpha1, alpha2, alpha12, tau
pars = "alpha0,alpha1,alpha2,alpha12,tau,b".split(',')
_, init = load('../Seedsinits.txt')
# implicit parameter b initialized to zero
init['b'] = np.zeros(N)
p0, labels = define_pars(init, pars)

def nllf(p):
    alpha0, alpha1, alpha2, alpha12, tau = p[:5]
    b = p[5:]

    p = ilogit(alpha0 + alpha1*x1 + alpha2*x2 + alpha12*x1*x2 + b)

    cost = 0
    cost += np.sum(dbin_llf(r, p, n))
    cost += np.sum(dnorm_llf(b, 0., tau))
    cost += dnorm_llf(alpha0, 0.0, 1e-6)
    cost += dnorm_llf(alpha1, 0.0, 1e-6)
    cost += dnorm_llf(alpha2, 0.0, 1e-6)
    cost += dnorm_llf(alpha12, 0.0, 1e-6)
    cost += dgamma_llf(tau, 0.001, 0.001)

    return -cost

def post(p):
    alpha0, alpha1, alpha2, alpha12, tau = p[:5]
    sigma = 1.0 / sqrt(tau)
    return [sigma]
post_vars = ["sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem.setp(p0)
problem._bounds[0, 4] = 0  # tau >= 0
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha0", "alpha1", "alpha12", "alpha2", "sigma"]

openbugs_result = """
         mean      sd      MC_error  2.5pc    median   97.5pc    start  sample
alpha0   -0.5499   0.1965  0.004298  -0.9433  -0.5522  -0.1596   1001   10000
alpha1    0.08902  0.3124  0.005997  -0.5504   0.09795  0.6812   1001   10000
alpha12  -0.841    0.4372  0.008725  -1.736   -0.8265   0.008258 1001   10000
alpha2    1.356    0.2772  0.006133   0.8298   1.351    1.914    1001   10000
sigma     0.2922   0.1467  0.007297   0.04439  0.2838   0.6104   1001   10000
"""
