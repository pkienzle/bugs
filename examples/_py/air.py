# model
# {
#         for(j in 1 : J) {
#                 y[j] ~ dbin(p[j], n[j])
#                 logit(p[j]) <- theta[1] + theta[2] * X[j]
#                 X[j] ~ dnorm(mu[j], tau)
#                 mu[j] <- alpha + beta * Z[j]
#         }
#         theta[1] ~ dnorm(0.0, 0.001)
#         theta[2] ~ dnorm(0.0, 0.001)
# }

from bumps.names import *
from bumps import bugs
from bumps.bugsmodel import dbin_llf, dnorm_llf, ilogit

# J, y[J], n[J], Z[J], tau, alpha, beta
_,data = bugs.load('../Airdata.txt')
J = data["J"]
# theta[2], X[J]
_,init = bugs.load('../Airinits.txt')
p0 = np.hstack([init[s] for s in ("theta","X")])

def air(pars):
    theta, X = pars[0:2], pars[2:]
    p = np.array([ilogit(theta[0] + theta[1]*Xj) for Xj in X])
    mu = data["alpha"] + data["beta"] * data["Z"]
    cost = 0
    cost += np.sum(dbin_llf(data["y"], p, data["n"]))
    cost += np.sum(dnorm_llf(X, mu, data["tau"]))
    cost += dnorm_llf(theta[0], 0.0, 0.001)
    cost += dnorm_llf(theta[1], 0.0, 0.001)
    return -cost


problem = DirectPDF(air, len(p0), 1)
problem.setp(p0)

# From Air.txt
#	mean	sd	median	2.5pc	97.5pc
#theta1	-0.9591	1.981	-0.6974	-4.282	0.3374
#theta2	0.04771	0.0813	0.03844	-0.0023	0.1728
#X1	13.37	8.438	13.55	-3.644	29.42
#X2	27.36	7.365	27.38	12.95	41.81
#X3	41.05	8.655	40.93	24.55	58.47
