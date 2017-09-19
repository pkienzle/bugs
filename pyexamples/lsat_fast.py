"""
LSAT: item response

[PAK: This is a numerical experiment that is not part of the openbugs
distribution, and is not a recommendation for handling random effects models.]

Modified LSAT to treat students with the same response as identical.  So
rather than 1000 parameters representing the individual student ability
we only need 32, which is one for each response.  This is not the same as
marginalizing across the individual groups of students with the same set
of responses.  With correlated error, this is no longer a correct random effects
model.  The solution is similar, but the width of the distribution is
much reduced.

The equivalent OpenBUGS model would be adjusted as follows:

::

    model
    {
    # Calculate individual (binary) responses to each test from multinomial data
        for (j in 1 : culm[1]) {
            for (k in 1 : T) {
                r[j, k] <- response[1, k]
            }
        }
        for (i in 2 : R) {
            for (j in culm[i - 1] + 1 : culm[i]) {
                for (k in 1 : T) {
                    r[j, k] <- response[i, k]
                }
            }
        }
    # Treat students with the same reponse profile as having the same latent ability
        for (i in 1 : R) {
            ability[i] ~ dnorm(0, 1)
        }
        for (j in 1 : culm[1]) {
            theta[j] <- ability[i]
        }
        for (i in 2 : R) {
            for (j in culm[i-1] + 1 : culm[i]) {
                theta[j] <- ability[i]
            }
        }
    # Rasch model
        for (j in 1 : N) {
            for (k in 1 : T) {
                logit(p[j, k]) <- beta * theta[j] - alpha[k]
                r[j, k] ~ dbern(p[j, k])
            }
        }
    # Priors
        for (k in 1 : T) {
            alpha[k] ~ dnorm(0, 0.0001)
            a[k] <- alpha[k] - mean(alpha[])
        }
        beta ~ dflat()T(0, )
    }

As a further trick, rather than forming the full student and response matrices,
we simply multiply the response log-likelihood by the response frequency, which
is computationally equivalent but much faster.
"""
raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dbern_llf, dflat_llf, ilogit

#  data: N=1000, R=32, T=5, culm[R], response[R,T]
vars = "N,R,T,culm,response".split(',')
_, data = load('../examples/Lsatdata.txt')
N, R, T, culm, response = (data[p] for p in vars)
# init: alpha[T], beta, theta[N]
pars = "alpha,beta,theta".split(',')
_, init = load('../examples/Lsatinits.txt')
init["theta"] = np.zeros(R)
p0, labels = define_pars(init, pars)

def pre():
    frequency = np.diff(np.hstack((0, culm)))
    return frequency
frequency = pre()

def nllf(p):
    alpha, beta, theta = p[:T], p[T], p[T+1:]

    p = ilogit(beta*theta[:, None] - alpha[None, :])

    cost = 0
    cost += np.sum(frequency[:, None]*dbern_llf(response, p))
    cost += np.sum(frequency*dnorm_llf(theta, 0.0, 1.0))
    cost += np.sum(dnorm_llf(alpha, 0, 0.0001))
    cost += dflat_llf(beta)

    return -cost

def post(p):
    alpha, beta, theta = p[:T], p[T], p[T+1:]
    a = alpha - np.mean(alpha, axis=0)
    return a
post_vars = ["a[%d]"%k for k in range(1, T+1)]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0, T] = 0  # beta = dflat()T(0, )
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = post_vars + ["beta"]


openbugs_result = """
         mean     sd       MC_error   2.5pc    median    97.5pc   start  sample
  a[1]  -1.26     0.1053   0.001266  -1.474    -1.26     -1.056    1001   10000
  a[2]   0.4776   0.0698   8.158E-4   0.3412    0.4776    0.6168   1001   10000
  a[3]   1.239    0.0687   9.116E-4   1.106     1.239     1.374    1001   10000
  a[4]   0.1696   0.07325  8.07E-4    0.02688   0.1692    0.313    1001   10000
  a[5]  -0.6256   0.08617  0.001083  -0.7961   -0.6239   -0.4563   1001   10000
  beta   0.7582   0.07181  0.001678   0.6125    0.7601    0.895    1001   10000
"""
