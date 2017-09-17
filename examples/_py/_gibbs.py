# Gibbs sampler for the change-point model described in a Cognition cheat sheet titled "Gibbs sampling."
# This is a Python implementation of the procedure at http://www.cmpe.boun.edu.tr/courses/cmpe58n/fall2009/
# Written by Ilker Yildirim, September 2012.

# Paul Kienzle <pkienzle@gmail.com> 2017-09-16
# * add "temperature" parameter T for generating flattened posteriors
# * limit status updates to once per second
# * update to python 2/3 compatibility

from __future__ import print_function, division

from scipy.stats import uniform, gamma, poisson
import matplotlib.pyplot as plt
import numpy
from numpy import log,exp
from numpy.random import multinomial
import time

# fix the random seed for replicability.
numpy.random.seed(123456789)

# Generate data

# Hyperparameters
N=50
a=2
b=1
T=1

# Change-point: where the intensity parameter changes.
n=int(round(uniform.rvs()*N))
print("Change point " + str(n))

# Intensity values
lambda1=gamma.rvs(a,scale=1./b) # We use 1/b instead of b because of the way Gamma distribution is parametrized in the package random.
lambda2=gamma.rvs(a,scale=1./b)

lambdas=[lambda1]*n
lambdas[n:N-1]=[lambda2]*(N-n)

# Observations, x_1 ... x_N
x=poisson.rvs(lambdas)

# make one big subplots and put everything in it.
f, (ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,1)
# Plot the data
ax1.stem(range(N),x,linefmt='b-', markerfmt='bo')
ax1.plot(range(N),lambdas,'r--')
ax1.set_ylabel('Counts')

# Gibbs sampler
E=5200
BURN_IN=200

# Initialize the chain
n=int(round(uniform.rvs()*N))
lambda1=gamma.rvs(a,scale=1./b)
lambda2=gamma.rvs(a,scale=1./b)

# Store the samples
chain_n=numpy.array([0.]*(E-BURN_IN))
chain_lambda1=numpy.array([0.]*(E-BURN_IN))
chain_lambda2=numpy.array([0.]*(E-BURN_IN))

def gamma_T(shape, scale, T=1):
	# (k-1)/T = k' - 1 => k' = (k-1)/T + 1
	shape_p = (shape - 1)/T + 1
	scale_p = scale*T
	return gamma.rvs(shape_p, scale=scale_p)

next_update = time.time() + 1
for e in range(E):
	now = time.time()
	if now > next_update:
		print("At iteration "+str(e+1)+" of "+str(E))
		next_update = now + 1

	# sample lambda1 and lambda2 from their posterior conditionals, Equation 8 and Equation 9, respectively.
	#lambda1=gamma.rvs(a+sum(x[0:n]), scale=1./(n+b))
	#lambda2=gamma.rvs(a+sum(x[n:N]), scale=1./(N-n+b))
	lambda1=gamma_T(a+sum(x[0:n]), scale=1./(n+b), T=T)
	lambda2=gamma_T(a+sum(x[n:N]), scale=1./(N-n+b), T=T)

	# sample n, Equation 10
	mult_n=numpy.array([0]*N)
	for i in range(N):
		mult_n[i]=sum(x[0:i])*log(lambda1)-i*lambda1+sum(x[i:N])*log(lambda2)-(N-i)*lambda2
	mult_n=exp((mult_n-max(mult_n))/T)
	n=numpy.where(multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]

	# store
	if e>=BURN_IN:
		chain_n[e-BURN_IN]=n
		chain_lambda1[e-BURN_IN]=lambda1
		chain_lambda2[e-BURN_IN]=lambda2
print("Done.")

# Store the results in _gibbs_<T>.json
with open("_gibbs_%g.json"%T, "w") as fd:
	import json
	json.dump({
		'lambda1': chain_lambda1.tolist(),
		'lambda2': chain_lambda2.tolist(),
		'n': chain_n.tolist(),
	}, fd)


ax2.plot(chain_lambda1,'b',chain_lambda2,'g')
ax2.set_ylabel('$\lambda$')
ax3.hist(chain_lambda1,numpy.linspace(0,12,50),normed=True)
ax3.set_xlabel('$\lambda_1$')
ax3.set_xlim([0,12])
ax4.hist(chain_lambda2,numpy.linspace(0,12,50),color='g',normed=True)
ax4.set_xlim([0,12])
ax4.set_xlabel('$\lambda_2$')
ax5.hist(chain_n,bins=numpy.arange(51)+0.5,normed=True)
ax5.set_xlabel('n')
#ax5.set_xlim([1,50])
plt.show()


