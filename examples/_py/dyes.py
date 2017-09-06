#model
#{
#	for(i in 1 : batches) {
#		mu[i] ~ dnorm(theta, tau.btw)
#		for(j in 1 : samples) {
#			y[i , j] ~ dnorm(mu[i], tau.with)
#		}
#	}
#	sigma2.with <- 1 / tau.with
#	sigma2.btw <- 1 / tau.btw
#	tau.with ~ dgamma(0.001, 0.001)
#	tau.btw ~ dgamma(0.001, 0.001)
#	theta ~ dnorm(0.0, 1.0E-10)
#}

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf

#  data: batches=6, samples=5, y[batches,samples]
vars = "batches,samples,y".split(',')
_, data = load('../Dyesdata.txt')
batches, samples, y = (data[p] for p in vars)
# init: theta, tau.with, tau.btw, mu[batches]=0
pars = "theta,tau.with,tau.btw,mu".split(',')
_, init = load('../Dyesinits.txt')
init["mu"] = np.zeros(batches)
p0, labels = define_pars(init, pars)

def dyes(p):
    theta, tau_with, tau_btw = p[:3]
    mu = p[3:]

    cost = 0
    cost += np.sum(dnorm_llf(mu, theta, tau_btw))
    cost += np.sum(dnorm_llf(y, mu[:,None], tau_with))
    cost += dgamma_llf(tau_with, 0.001, 0.001)
    cost += dgamma_llf(tau_btw, 0.001, 0.001)
    cost += dnorm_llf(theta, 0, 1e-6)

    return -cost

def post(p):
    theta, tau_with, tau_btw = p[:3]
    sigma2_with = 1 / tau_with
    sigma2_btw = 1 / tau_btw
    return [sigma2_with, sigma2_btw]
post_vars = ["sigma2.with", "sigma2.btw"]

dof = 100
problem = DirectProblem(dyes, p0, labels=labels, dof=dof)

problem._bounds[0, 1:3] = 0  # tau.with, tau.btw bounded below by 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["sigma2.btw", "sigma2.with", "theta"]


#	mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
#sigma2.btw	2252.0	3974.0	39.33	0.00895	1341.0	10250.0	1001	100000
#sigma2.with	3009.0	1100.0	18.96	1550.0	2777.0	5745.0	1001	100000
#theta	1527.0	21.73	0.171	1484.0	1527.0	1571.0	1001	100000