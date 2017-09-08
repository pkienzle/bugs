"""
Sensitivity to prior distributions: application to Magnesium meta-analysis

::

model
{
#        j indexes alternative prior distributions
        for (j in 1:6) {
                mu[j] ~ dunif(-10, 10)
                OR[j] <- exp(mu[j])

#        k indexes study number
                for (k in 1:8) {
                        theta[j, k] ~ dnorm(mu[j], inv.tau.sqrd[j])
                        rtx[j, k] ~ dbin(pt[j, k], nt[k])
                        rtx[j, k] <- rt[k]
                        rcx[j, k] ~ dbin(pc[j, k], nc[k])
                        rcx[j, k] <- rc[k]
                        logit(pt[j, k]) <- theta[j, k] + phi[j, k]
                        phi[j, k] <- logit(pc[j, k])
                        pc[j, k] ~ dunif(0, 1)
                }
        }

#        k  again indexes study number
        for (k in 1:8) {
                # log-odds ratios:
                y[k] <- log(((rt[k] + 0.5) / (nt[k] - rt[k] + 0.5)) / ((rc[k] + 0.5) / (nc[k] - rc[k] + 0.5)))
#         variances & precisions:
                sigma.sqrd[k] <- 1 / (rt[k] + 0.5) + 1 / (nt[k] - rt[k] + 0.5) + 1 / (rc[k] + 0.5) +
                                        1 / (nc[k] - rc[k] + 0.5)
                prec.sqrd[k] <- 1 / sigma.sqrd[k]
        }
        s0.sqrd <- 1 / mean(prec.sqrd[1:8])

# Prior 1: Gamma(0.001, 0.001) on inv.tau.sqrd

        inv.tau.sqrd[1] ~ dgamma(0.001, 0.001)
        tau.sqrd[1] <- 1 / inv.tau.sqrd[1]
        tau[1] <- sqrt(tau.sqrd[1])

# Prior 2: Uniform(0, 50) on tau.sqrd

        tau.sqrd[2] ~ dunif(0, 50)
        tau[2] <- sqrt(tau.sqrd[2])
        inv.tau.sqrd[2] <- 1 / tau.sqrd[2]

# Prior 3: Uniform(0, 50) on tau

        tau[3] ~ dunif(0, 50)
        tau.sqrd[3] <- tau[3] * tau[3]
        inv.tau.sqrd[3] <- 1 / tau.sqrd[3]

# Prior 4: Uniform shrinkage on tau.sqrd

        B0 ~ dunif(0, 1)
        tau.sqrd[4] <- s0.sqrd * (1 - B0) / B0
        tau[4] <- sqrt(tau.sqrd[4])
        inv.tau.sqrd[4] <- 1 / tau.sqrd[4]

# Prior 5: Dumouchel on tau

        D0 ~ dunif(0, 1)
        tau[5] <- sqrt(s0.sqrd) * (1 - D0) / D0
        tau.sqrd[5] <- tau[5] * tau[5]
        inv.tau.sqrd[5] <- 1 / tau.sqrd[5]

# Prior 6: Half-Normal on tau.sqrd

        p0 <- phi(0.75) / s0.sqrd
        tau.sqrd[6] ~ dnorm(0, p0)T(0, )
        tau[6] <- sqrt(tau.sqrd[6])
        inv.tau.sqrd[6] <- 1 / tau.sqrd[6]

}
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *

from math import sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dbin_llf, phi, logit, ilogit

if len(sys.argv) < 0:
    raise RuntimeError("use 'bumps magnesium.py PRIOR MARG_N' to select prior and internal marginalization count")
PRIOR = int(sys.argv[1])
MARGINALIZATION_COUNT = int(sys.argv[2])

Nprior = 6 if PRIOR == 0 else 1
Nstudy = 8

# data: rt[Nstudy], nt[Nstudy], rc[Nstudy], nc[Nstudy]
_, data = load('../Magnesiumdata.txt')
rt, nt, rc, nc = data["rt"], data["nt"], data["rc"], data["nc"]

# inits: mu[Nprior], tau[Nprior], tau.sqrd[Nprior], inv.tau.sqrd[Nprior]
_, init = load('../Magnesiuminits.txt')

## PAK: simplifying the parameter structure so it only contains active pars
simple_init = [
    ('inv.tau.sqrd[1]', init['inv.tau.sqrd'][0]),
    ('tau.sqrd[2]', init['tau.sqrd'][1]),
    ('tau[3]', init['tau'][2]),
    ('B0', 0.5),
    ('D0', 0.5),
    ('tau.sqrd[6]', init['tau.sqrd'][5]),
    ('mu', init['mu']),
    ]
if PRIOR > 0:
    simple_init = [simple_init[PRIOR-1], ('mu', init['mu'][PRIOR-1])]
init = dict(simple_init)
pars = [name for name, value in simple_init]
p0, labels = define_pars(init, pars)

# log-odds ratios:
y = log(((rt + 0.5) / (nt - rt + 0.5)) / ((rc + 0.5) / (nc - rc + 0.5)))

# variances & precisions:
sigma_sqrd = (1/(rt + 0.5) + 1/(nt-rt + 0.5) + 1/(rc + 0.5) + 1/(nc-rc + 0.5))
prec_sqrd = 1 / sigma_sqrd
s0_sqrd = 1 / np.mean(prec_sqrd)
prior_6_p0 = phi(0.75) / s0_sqrd

# === Free parameters ===
# inv_tau_sqrd[1] ~ dgamma(0.001, 0.001)
# tau.sqrd[2] ~ dunif(0, 50)
# tau[3] ~ dunif(0, 50)
# B0 ~ dunif(0, 1)
# D0 ~ dunif(0, 1)
# tau_sqrd[6] ~ dnorm(0, p0)T(0, )
# mu[Nprior] ~ unif(-10, 10)
# theta[Nprior, Nstudy] ~ norm(mu[Nprior], inv.tau.sqrd[Nprior])
# pc[Nprior, Nstudy] ~ unif(0,1)

sigma_prior = [
    lambda inv_tau_sqrd_1: 1 / np.sqrt(inv_tau_sqrd_1),
    lambda tau_sqrd_2: np.sqrt(tau_sqrd_2),
    lambda tau_3: tau_3,
    lambda B0: np.sqrt(s0_sqrd) * (1 - B0) / B0,
    lambda D0: np.sqrt(s0_sqrd) * (1 - D0) / D0,
    lambda tau_sqrd_6: np.sqrt(tau_sqrd_6),
    ]

def nllf(p):
    if PRIOR > 0:
        sigma = np.array([sigma_prior[PRIOR-1](p[0])])
        mu = np.array([p[1]])
    else:
        #inv_tau_sqrd_1, tau_sqrd_2, tau_3, B0, D0, tau_sqrd_6 = p[:6]
        sigma = np.array([f(v) for f, v in zip(sigma_prior, p[:6])])
        mu = p[6:12]

    tau = 1/np.sqrt(sigma)

    cost = 0
    for _ in range(MARGINALIZATION_COUNT):
        theta = np.random.normal(mu[:, None], sigma[:, None], size=(Nprior, Nstudy))
        pc = np.random.rand(Nprior, Nstudy)
        pt = ilogit(theta + logit(pc))
        cost += np.sum(dnorm_llf(theta, mu[:, None], tau[:, None]))
        cost += np.sum(dbin_llf(rt[None, :], pt, nt[None, :]))
        cost += np.sum(dbin_llf(rc[None, :], pc, nc[None, :]))
    cost /= MARGINALIZATION_COUNT

    #cost += np.sum(dunif_llf(mu, -10, 10))

    # Prior 1: Gamma(0.001, 0.001) on inv.tau.sqrd
    if PRIOR == 1: cost += dgamma_llf(p[0], 0.001, 0.001)
    elif PRIOR == 0: cost += dgamma_llf(p[0], 0.001, 0.001)

    # Prior 2: Uniform(0, 50) on tau.sqrd
    #cost += dunif_llf(tau_sqrd_2, 0, 50)

    # Prior 3: Uniform(0, 50) on tau
    #cost += dunif_llf(tau_3, 0, 50)

    # Prior 4: Uniform shrinkage on tau.sqrd
    #cost += dunif_llf(B0, 0, 1)

    # Prior 5: Dumouchel on tau
    #cost += dunif_llf(D0, 0, 1)

    # Prior 6: Half-Normal on tau.sqrd
    if PRIOR == 6: cost += dnorm_llf(p[0], 0, prior_6_p0)
    elif PRIOR == 0: cost += dnorm_llf(p[5], 0, prior_6_p0)

    return -cost

def post(p):
    if PRIOR > 0:
        v, mu = p
        sigma = [sigma_prior[PRIOR-1](v)]
    else:
        v = p[:6]
        mu = p[6:12]
        sigma = [f(v) for f, v in zip(sigma_prior, v)]

    OR = exp(mu)
    return np.vstack([OR] + sigma)
if PRIOR == 0:
    post_vars = (["OR[%d]"%k for k in range(1, Nprior+1)]
                 + ["tau[%d]"%k for k in range(1, Nprior+1)])
else:
    post_vars = ["OR[%d]"%PRIOR, "tau[%d]"%PRIOR]


problem = DirectProblem(nllf, p0, labels=labels, dof=100)
if PRIOR == 0:
    problem._bounds[0, 0] = 0 # inv_tau_sqrd[1] ~ dgamma(0.001, 0.001)
    problem._bounds[:, 1] = 0, 50 # tau.sqrd[2] ~ dunif(0, 50)
    problem._bounds[:, 2] = 0, 50 # tau[3] ~ dunif(0, 50)
    problem._bounds[:, 3] = 0, 1 # B0 ~ dunif(0, 1)
    problem._bounds[:, 4] = 0, 1 # D0 ~ dunif(0, 1)
    problem._bounds[0, 5] = 0 # tau_sqrd[6] ~ dnorm(0, p0)T(0, )
    problem._bounds[:, 6:12] = [-10]*6, [10]*6 # mu[Nprior] ~ unif(-10, 10)
else:
    problem._bounds[:, 1] = -10, 10 # mu[Nprior] ~ unif(-10, 10)
if PRIOR == 1:
    problem._bounds[0, 0] = 0 # inv_tau_sqrd[1] ~ dgamma(0.001, 0.001)
elif PRIOR == 2:
    problem._bounds[:, 0] = 0, 50 # tau.sqrd[2] ~ dunif(0, 50)
elif PRIOR == 3:
    problem._bounds[:, 0] = 0, 50 # tau[3] ~ dunif(0, 50)
elif PRIOR == 4:
    problem._bounds[:, 0] = 0, 1 # B0 ~ dunif(0, 1)
elif PRIOR == 5:
    problem._bounds[:, 0] = 0, 1 # D0 ~ dunif(0, 1)
elif PRIOR == 6:
    problem._bounds[0, 0] = 0 # tau_sqrd[6] ~ dnorm(0, p0)T(0, )

# theta[Nprior, Nstudy] ~ norm(mu[Nprior], inv.tau.sqrd[Nprior])
# pc[Nprior, Nstudy] ~ unif(0,1)
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = post_vars

openbugs_result = """
        mean    sd      MC_error  2.5pc    median  97.5pc  start  sample
OR[1]   0.4733  0.1516  0.003591  0.1959   0.4709  0.7655  2001   20000
OR[2]   0.4231  0.2567  0.006416  0.1063   0.3871  0.9769  2001   20000
OR[3]   0.4379  0.1843  0.005731  0.146    0.4233  0.8126  2001   20000
OR[4]   0.4648  0.1359  0.004725  0.2169   0.4613  0.7375  2001   20000
OR[5]   0.4792  0.1624  0.006243  0.2126   0.4756  0.7783  2001   20000
OR[6]   0.4486  0.1447  0.004673  0.2108   0.4321  0.7546  2001   20000
tau[1]  0.5267  0.3833  0.01159   0.04487  0.454   1.473   2001   20000
tau[2]  1.13    0.6169  0.02279   0.3549   0.985   2.784   2001   20000
tau[3]  0.8023  0.514   0.02051   0.08684  0.7085  2.099   2001   20000
tau[4]  0.494   0.2665  0.009859  0.1197   0.4456  1.141   2001   20000
tau[5]  0.5142  0.3508  0.01583   0.02291  0.4627  1.365   2001   20000
tau[6]  0.5584  0.1972  0.007635  0.1575   0.5602  0.9354  2001   20000
"""
