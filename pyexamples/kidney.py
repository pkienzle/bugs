"""
Kidney: Weibull regression with random efects

::

    model
    {
        for (i in 1 : N) {
            for (j in 1 : M) {
    # Survival times bounded below by censoring times:
                t[i,j] ~ dweib(r, mu[i,j])C(t.cen[i, j], );
                log(mu[i,j ]) <- alpha + beta.age * age[i, j]
                        + beta.sex  *sex[i]
                        + beta.dis[disease[i]] + b[i];
            }
    # Random effects:
            b[i] ~ dnorm(0.0, tau)
        }
    # Priors:
        alpha ~ dnorm(0.0, 0.0001);
        beta.age ~ dnorm(0.0, 0.0001);
        beta.sex ~ dnorm(0.0, 0.0001);
    #    beta.dis[1] <- 0;  # corner-point constraint
        for(k in 2 : 4) {
            beta.dis[k] ~ dnorm(0.0, 0.0001);
        }
        tau ~ dgamma(1.0E-3, 1.0E-3);
        r ~ dgamma(1.0, 1.0E-3);
        sigma <- 1 / sqrt(tau); # s.d. of random effects
    }
"""
import warnings

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dweib_C_llf, dgamma_llf

# TODO: fix kidney model
raise NotImplementedError("Model fails to reproduce the OpenBUGS result")
# Note: censored Weibull was tested in mice.py

#  data: N=38, M=2, t[N,M], t.cen[N,M], age[N,M], beta.dis[4], sex[N], disease[N]
vars = "N,M,t,t.cen,age,beta.dis,sex,disease".split(',')
_, data = load('../examples/Kidneydata.txt')
N, M, t, t_cen, age, beta_dis, sex, disease = (data[p] for p in vars)
# init: beta.age, beta.sex, beta.dis[4], alpha, r, tau
pars = "beta.age,beta.sex,beta.dis,alpha,r,tau,b".split(',')
_, init = load('../examples/Kidneyinits.txt')
init['b'] = np.zeros(N)
p0, labels = define_pars(init, pars)
# TODO: add notion of fixed parameters to DirectProblem
# remove beta.dis[1] since it is not a fitting parameter
del labels[2]
p0 = np.hstack((p0[:2], p0[3:]))

# beta_dis[0] = 0  # corener-point constraint

def nllf(p):
    beta_age, beta_sex = p[:2]
    beta_dis_ = np.empty(4)
    beta_dis_[0] = beta_dis[0]
    beta_dis_[1:] = p[2:5]
    alpha, r, tau = p[5:8]
    b = p[8:]

    mu = exp(alpha
             + beta_age*age
             + beta_sex*sex[:, None]
             + beta_dis_[disease[:, None]-1]  # numpy is 0-origin
             + b[:, None])

    cost = 0
    cost += np.sum(dweib_C_llf(t, r, mu, lower=t_cen))
    cost += np.sum(dnorm_llf(b, 0.0, tau))
    cost += dnorm_llf(alpha, 0, 0.0001)
    cost += dnorm_llf(beta_age, 0, 0.0001)
    cost += dnorm_llf(beta_sex, 0, 0.0001)
    cost += np.sum(dnorm_llf(beta_dis_[1:], 0, 0.0001))
    cost += dgamma_llf(tau, 1e-3, 1e-3)
    cost += dgamma_llf(r, 1, 1e-3)

    return -cost

def post(p):
    tau = p[7]
    sigma = 1/sqrt(tau)
    return [sigma]
post_vars = ["sigma"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0,7] = 0  # tau >= 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["alpha", "beta.dis[2]", "beta.dis[3]", "beta.dis[4]",
                        "beta.sex", "r", "sigma"]

openbugs_result = """
            mean    sd     MC_error 2.5pc   median  97.5pc  start sample
alpha       -4.693  1.045  0.05452  -6.971  -4.579  -2.927  5001  20000
beta.dis[2]  0.1362 0.5865 0.01461  -0.9617  0.1147  1.372  5001  20000
beta.dis[3]  0.6575 0.6062 0.0184   -0.4679  0.627   1.956  5001  20000
beta.dis[4] -1.205  0.8619 0.02398  -2.899  -1.219   0.5216 5001  20000
beta.sex    -2.005  0.5574 0.02368  -3.207  -1.979  -1.007  5001  20000
r            1.236  0.1976 0.0127    0.9128  1.213   1.657  5001  20000
sigma        0.6843 0.3868 0.02288   0.04943 0.6941  1.44   5001  20000
"""
