#model
#{
#	for( i in 1 : N ) {
#		r[i] ~ dbin(p[i],n[i])
#		logit(p[i]) <- alpha.star + beta * (x[i] - mean(x[]))
#		rhat[i] <- n[i] * p[i]
#	}
#	alpha <- alpha.star - beta * mean(x[])
#	beta ~ dnorm(0.0,0.001)
#	alpha.star ~ dnorm(0.0,0.001)
#}

from bumps.names import *
from bumps import bugs
from bumps.bugsmodel import ilogit, dnorm_llf, dbin_llf

#  data: x[N],n[N],r[N],N
vars = "x,n,r,N".split(',')
_,data = bugs.load('../Beetlesdata.txt')
x,n,r,N = (data[p] for p in vars)
xbar = np.mean(x)
# init: alpha.star, beta
pars = 'alpha.star,beta'.split(',')
_,init = bugs.load('../Beetlesinits.txt')
p0 = np.hstack([init[s] for s in pars])


def beetles(pars):
    cost = 0
    alpha_star, beta = pars
    p = np.array([ilogit(alpha_star + beta*(xi-xbar)) for xi in x])
    cost += np.sum(dbin_llf(r,p,n))
    cost += dnorm_llf(beta, 0.0, 0.001)
    cost += dnorm_llf(alpha_star, 0.0, 0.001)
    return -cost

def transform(pars):
    alpha_star, beta = pars
    alpha = alpha_star - beta*np.mean(x)
    p = ilogit(alpha_star + beta*(x-xbar))
    rhat = n*p
    return np.hstack([alpha, beta, rhat])

num_pars = len(p0)
problem = DirectPDF(beetles, num_pars, N-num_pars)
problem.setp(p0)

#	mean	sd	median	2.5pc	97.5pc
#alpha	-60.76	5.172	-60.59	-71.41	-51.19
#beta	34.29	2.906	34.19	28.92	40.24
#rhat1	3.565	0.9533	3.491	1.945	5.633
#rhat2	9.936	1.696	9.894	6.814	13.41
#rhat3	22.47	2.131	22.44	18.31	26.65
#rhat4	33.85	1.78	33.85	30.34	37.27
#rhat5	50.01	1.66	50.04	46.65	53.14
#rhat6	53.2	1.107	53.26	50.88	55.2
#rhat7	59.13	0.7358	59.21	57.52	60.39
#rhat8	58.68	0.4248	58.74	57.72	59.36
