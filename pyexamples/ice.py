"""
Ice: non-parametric smoothing in an age-cohort model

::

    model
    {
        for (i in 1:I)  {
            cases[i]        ~ dpois(mu[i])
            log(mu[i])     <- log(pyr[i]) + alpha[age[i]] + beta[year[i]]
        }
        betamean[1]    <- 2 * beta[2] - beta[3]
        Nneighs[1]     <- 1
        betamean[2]    <- (2 * beta[1] + 4 * beta[3] - beta[4]) / 5
        Nneighs[2]     <- 5
        for (k in 3 : K - 2)  {
            betamean[k]    <- (4 * beta[k - 1] + 4 * beta[k + 1]- beta[k - 2] - beta[k + 2]) / 6
            Nneighs[k]     <- 6
        }
        betamean[K - 1]  <- (2 * beta[K] + 4 * beta[K - 2] - beta[K - 3]) / 5
        Nneighs[K - 1]   <- 5
        betamean[K]    <- 2 * beta[K - 1] - beta[K - 2]
        Nneighs[K]     <- 1
        for (k in 1 : K)  {
            betaprec[k]    <- Nneighs[k] * tau
        }
        for (k in 1 : K)  {
            beta[k]        ~ dnorm(betamean[k], betaprec[k])
            logRR[k]      <- beta[k] - beta[5]
            tau.like[k]   <- Nneighs[k] * beta[k] * (beta[k] - betamean[k])
        }
        alpha[1]      <- 0.0
        for (j in 2 : Nage)  {
            alpha[j]       ~ dnorm(0, 1.0E-6)
        }
        d <- 0.0001 + sum(tau.like[]) / 2
        r <- 0.0001 + K / 2
        tau  ~ dgamma(r, d)
        sigma <- 1 / sqrt(tau)
    }
"""
from __future__ import division

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dpois_llf, dgamma_llf, dnorm_llf

# data: I=77, Nage=13, K=11, age[I], year[I], cases[I], pyr[I]
_, data = load('../examples/Icedata.txt')
globals().update(data)
# init: tau, alpha[Nage], beta[K]
pars = "tau alpha beta".split()
_, init = load('../examples/Iceinits.txt')
init['alpha'][0] = 0. # set alpha[1] to its fixed value
p0, labels = define_pars(init, pars)

age_index = np.asarray(age, 'i') - 1
year_index = np.asarray(year, 'i') - 1

def nllf(p):
    tau, alpha, beta = p[0], p[1:Nage+1], p[Nage+1:Nage+1+K]
    alpha[0] = 0.  # alpha 0 is not a fitting parameter

    mu = exp(log(pyr) + alpha[age_index] + beta[year_index])

    betamean = np.empty(beta.shape)
    betamean[0] = 2 * beta[1] - beta[2]
    betamean[1] = (2 * beta[0] + 4 * beta[2] - beta[3]) / 5
    betamean[2:K-2] = (4 * beta[1:K-3] + 4 * beta[3:K-1] - beta[0:K-4] - beta[4:K]) / 6
    betamean[-2] = (2 * beta[-1] + 4 * beta[-3] - beta[-4]) / 5
    betamean[-1] = 2 * beta[-2] - beta[-3]
    logRR = beta - beta[4]

    Nneighs = np.empty(beta.shape, 'i')
    Nneighs[0] = 1
    Nneighs[1] = 5
    Nneighs[2:K-2] = 6
    Nneighs[-2] = 5
    Nneighs[-1] = 1

    betaprec = Nneighs * tau
    tau_like = Nneighs * beta * (beta - betamean)
    d = 0.0001 + np.sum(tau_like) / 2
    r = 0.0001 + K / 2

    cost = 0
    cost += np.sum(dpois_llf(cases, mu))
    cost += np.sum(dnorm_llf(beta, betamean, betaprec))
    cost += np.sum(dnorm_llf(alpha[1:], 0, 1e-6))
    cost += dgamma_llf(tau, r, d)

    return -cost

LOGRR = ["logRR[%d]"%(k+1) for k in range(K)]
def post(p):
    tau, alpha, beta = p[0], p[1:K+1], p[K+1:K+1+K]
    alpha[0] = 0.  # alpha 0 is not a fitting parameter

    betamean = np.empty(beta.shape)
    betamean[0] = 2 * beta[1] - beta[2]
    betamean[1] = (2 * beta[0] + 4 * beta[2] - beta[3]) / 5
    betamean[2:K-2] = (4 * beta[1:K-3] + 4 * beta[3:K-1] - beta[0:K-4] - beta[4:K]) / 6
    betamean[-2] = (2 * beta[-1] + 4 * beta[-3] - beta[-4]) / 5
    betamean[-1] = 2 * beta[-2] - beta[-3]
    logRR = beta - beta[4]

    sigma = 1 / sqrt(tau)
    return np.vstack((sigma, logRR))
post_vars = ["sigma"] + LOGRR

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0, 0] = 0 # tau ~ dgamma(r, d) => tau >= 0
# be lazy and leave alpha[1] as fitted even though it is fixed
problem._bounds[:, 1] = 0, 1e-10 # alpha[1] <- 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = LOGRR[:4] + LOGRR[5:]


openbugs_result = """
          mean    sd      MC_error 2.5pc   median    97.5pc  start sample
logRR[1]  -1.038  0.252   0.0143   -1.577  -1.004    -0.6829 1001  20000
logRR[2]  -0.748  0.1608  0.009239 -1.082  -0.7317   -0.5123 1001  20000
logRR[3]  -0.4615 0.08272 0.004348 -0.6405 -0.452    -0.3355 1001  20000
logRR[4]  -0.2006 0.03653 0.001021 -0.2758 -0.1982   -0.1241 1001  20000
logRR[6]   0.1616 0.04168 0.001625  0.0552  0.1719   0.2206  1001  20000
logRR[7]   0.3217 0.06385 0.003012  0.1724  0.3354   0.4188  1001  20000
logRR[8]   0.4837 0.0803  0.004211  0.3024  0.4964   0.6134  1001  20000
logRR[9]   0.6428 0.1036  0.005897  0.4178  0.6574   0.8112  1001  20000
logRR[10]  0.819  0.1293  0.007686  0.5361  0.839    1.035   1001  20000
logRR[11]  1.004  0.1757  0.009986  0.6182  1.026    1.302   1001  20000
"""
