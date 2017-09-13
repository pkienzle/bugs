"""
Cervix: case - control study with errors in covariates

::

    model
    {
        for (i in 1 : N) {
            x[i]   ~ dbern(q)         # incidence of HSV
            logit(p[i]) <- beta0C + beta * x[i]     # logistic model
            d[i]  ~ dbern(p[i])        # incidence of cancer
            x1[i] <- x[i] + 1
            d1[i] <- d[i] + 1
            w[i]  ~ dbern(phi[x1[i], d1[i]])     # incidence of w
        }
        q      ~ dunif(0.0, 1.0)           # prior distributions
        beta0C ~ dnorm(0.0, 0.00001);
        beta   ~ dnorm(0.0, 0.00001);
        for(j in 1 : 2) {
            for(k in 1 : 2){
                phi[j, k] ~ dunif(0.0, 1.0)
            }
        }
    # calculate gamma1 = P(x=1|d=0) and gamma2 = P(x=1|d=1)
        gamma1 <- 1 / (1 + (1 + exp(beta0C + beta)) / (1 + exp(beta0C)) * (1 - q) / q)
        gamma2 <- 1 / (1 + (1 + exp(-beta0C - beta)) / (1 + exp(-beta0C)) * (1 - q) / q)
    }
"""

# Another effects model that is too big to succeed...
raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dbern_llf, ilogit

#  data: N=2044, d[N], x[N], w[N] ; many x are NA
_, data = load('../Cervixdata.txt')
globals().update(data)
x_index = np.isnan(x)
# init: beta0C, beta, phi[2,2]
pars = "beta0C beta phi q x".split()
_, init = load('../Cervixinits.txt')
init['q'] = 1.
init['x'] = np.ones(np.sum(x_index))
p0, labels = define_pars(init, pars)


def nllf(p):
    beta0C, beta, phi, q, x[x_index] = p[0], p[1], p[2:6].reshape(2, 2), p[6], p[7:]

    p = ilogit(beta0C + beta*np.floor(x))
    x_int, d_int = [np.asarray(np.floor(v), 'i') for v in (x, d)]
    cost = 0
    cost += np.sum(dbern_llf(d, q))
    cost += np.sum(dbern_llf(d, p))
    cost += np.sum(dbern_llf(w, phi[x_int, d_int]))
    cost += dnorm_llf(beta0C, 0, 0.00001)
    cost += dnorm_llf(beta, 0, 0.00001)

    return -cost

def post(p):
    beta0C, beta, q = p[0], p[1], p[6]
    # calculate gamma1 = P(x=1|d=0) and gamma2 = P(x=1|d=1)
    gamma1 = 1 / (1 + (1 + exp(beta0C + beta)) / (1 + exp(beta0C)) * (1 - q) / q)
    gamma2 = 1 / (1 + (1 + exp(-beta0C - beta)) / (1 + exp(-beta0C)) * (1 - q) / q)
    return gamma1, gamma2
post_vars = ["gamma1", "gamma2"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[:, 2:6] = [0]*4, [1]*4 #phi ~ dunif(0, 1)
problem._bounds[:, 6] = 0, 1 #q ~ dunif(0, 1)
problem._bounds[0, 7:] = 0
problem._bounds[1, 7:] = 2-1e-10 #x ~ dbern(q)
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "beta0C", "gamma1", "gamma2",
    "phi[1,1]", "phi[1,2]", "phi[2,1]", "phi[2,2]", "q"
    ]


openbugs_result = """
         mean     sd      MC_error  2.5pc    median    97.5pc    start   sample
beta0C   -0.9061  0.1951  0.01105   -1.318   -0.9001   -0.5474    1501   10000
gamma1    0.4341  0.05513 0.002808   0.3292   0.4312    0.5502    1501   10000
gamma2    0.5861  0.06675 0.003997   0.4536   0.5862    0.7123    1501   10000
phi[1,1]  0.3214  0.05266 0.002882   0.2134   0.3221    0.4191    1501   10000
phi[1,2]  0.2242  0.0849  0.00499    0.07721  0.2178    0.4012    1501   10000
phi[2,1]  0.5667  0.06455 0.003372   0.4426   0.5654    0.6964    1501   10000
phi[2,2]  0.764   0.06507 0.00374    0.6403   0.7624    0.8928    1501   10000
q         0.4885  0.04398 0.002382   0.4065   0.4869    0.5802    1501   10000
"""