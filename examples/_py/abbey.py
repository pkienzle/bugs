# model
# {
#	for(i in 2 : N){
#		z[i] ~ dstable(alpha, beta, gamma, delta)
#		z[i] <- price[i] / price[i - 1] - 1
#	}
#
#	alpha ~ dunif(1.1, 2)
#	beta ~ dunif(-1, 1)
#	gamma ~ dunif(-0.05, 0.05)
#	delta ~ dunif(0.001, 0.5)
#
#	mean.z <- mean(z[2:50])
#	sd.z <- sd(z[2:50])
#}
from __future__ import division

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dstable_llf

N = 50
price = np.array([
    296, 296, 300, 302, 300, 304, 303, 299, 293, 294, 294, 293, 295,
    287, 288, 297, 305, 307, 307, 304, 303, 304, 304, 309, 309, 309,
    307, 306, 304, 300, 296, 301, 298, 295, 295, 293, 292, 297, 294,
    293, 306, 303, 301, 303, 308, 305, 302, 301, 297, 299,
    ])
z = price[1:] / price[:-1] - 1
mean_z, sd_z = np.mean(z), np.std(z, ddof=1)

labels = "alpha beta gamma delta".split()
def abbey(alpha, beta, gamma, delta):
    log_likelihood = np.sum(dstable_llf(z, alpha, beta, gamma, delta))
    return -log_likelihood

M = PDF(abbey, dof=len(z)-4, alpha=1.7, beta=0.5, gamma=0.00029, delta=0.0065)
M.alpha.range(1.1, 2)
M.beta.range(-1, 1)
M.gamma.range(-0.05, 0.05)
M.delta.range(0.001, 0.5)

problem = FitProblem(M)

# init1 alpha = 1.7, beta = 0.5, gamma = 0.00029, delta = 0.0065
# init2 alpha = 1.2, beta = -0.5, gamma = 0.00029, delta = 0.0065

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import levy
    t = np.linspace(-0.05, 0.05, 100)
    plt.hist(z, normed=True)
    plt.plot(t, levy.levy(t, 1.52569, 0.725142, -0.00315616, 0.00680555))
    plt.show()

#
#        mean     sd        MC_error  val2.5pc  median   val97.5pc  start  sample
# alpha	 1.558    0.1995    0.008487  1.17      1.56     1.923      1001   20000
# beta	-0.5499   0.3628    0.01235  -0.9743   -0.65     0.3909	    1001   20000
# delta	 0.008204 0.00158   6.989E-5  0.005828  0.007991 0.0121     1001   20000
# gamma	 7.936E-4 0.002908  1.506E-4 -0.003841  1.917E-4 0.007961   1001   20000
