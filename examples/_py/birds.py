#model {
## prior distributions
#	psi~dunif(0,1)
#	mu~dnorm(0,0.001)
#	tau~dgamma(.001,.001) # zero-inflated binomial mixture model for
#	                                          # the augmented data
#	for(i in 1: nind + nz){
#		z[i] ~ dbin(psi,1)
#		eta[i]~ dnorm(mu, tau)
#		logit(p[i])<- eta[i]
#		muy[i]<-p[i] * z[i]
#		y[i] ~ dbin(muy[i], J)
#	}
#	# Derived parameters
#	N<-sum(z[1 : nind+nz])
#	sigma<-sqrt(1  /tau)
#}

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbin_llf, dunif_llf, dnorm_llf, dgamma_llf, ilogit

#  data: y[nind+nz], nind, nz, J
vars = "y,nind,nz,J".split(',')
_, data = load('../Birdsdata.txt')
y, nind, nz, J = (data[p] for p in vars)
# init: mu, tau, psi, z[nind+nz]
pars = "mu,tau,psi,z".split(',')
_, init = load('../Birdsinits1.txt')
p0, labels = define_pars(init, pars)

# eta missing from data and init.  Assume it is initialized to mu
p0 = np.hstack((p0, np.zeros(nind+nz) + p0[0]))
labels.extend("eta[%d]"%k for k in range(1, nind+nz+1))

def birds(pars):
    mu, tau, psi, z = pars[0], pars[1], pars[2], pars[3:3+nind+nz]
    eta = pars[3+nind+nz:]

    p = ilogit(eta)  # logit(p) = eta
    muy = p*z

    cost = 0
    cost += dunif_llf(psi, 0, 1)
    cost += dnorm_llf(mu, 0, 0.001)
    cost += dgamma_llf(tau, 0.001, 0.001)
    cost += np.sum(dbin_llf(z, psi, 1))
    cost += np.sum(dnorm_llf(eta, mu, tau))
    cost += np.sum(dbin_llf(y, muy, J))

    return -cost

def post(pars):
    mu, tau, psi, z = pars[0], pars[1], pars[2], pars[3:3+nind+nz]
    N = np.sum(z, axis=0)
    sigma = np.sqrt(1/tau)
    return [N, sigma]
post_vars = ["N", "sigma"]

dof = 1
problem = DirectProblem(birds, p0, labels=labels, dof=dof)
problem._bounds[0, 1] = 0  # tau in [0, inf]
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["N", "mu", "psi", "sigma"]


#	mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
#N	89.85	12.85	0.7614	76.0	87.0	122.0	1001	20000
#mu	-2.692	0.4506	0.02712	-3.78	-2.606	-2.064	1001	20000
#psi	0.2815	0.04708	0.002392	0.2124	0.2744	0.3909	1001	20000
#sigma	1.668	0.3017	0.01815	1.226	1.614	2.361	1001	20000