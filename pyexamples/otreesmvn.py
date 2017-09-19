"""
Orange Trees: Non-linear growth curve (MVN)

::

    model {
        for (i in 1:K) {
            for (j in 1:n) {
                Y[i, j] ~ dnorm(eta[i, j], tauC)
                eta[i, j] <- phi[i, 1] / (1 + phi[i, 2] * exp(phi[i, 3] * x[j]))
            }
            phi[i, 1] <- exp(theta[i, 1])
            phi[i, 2] <- exp(theta[i, 2]) - 1
            phi[i, 3] <- -exp(theta[i, 3])
            theta[i, 1:3] ~ dmnorm(mu[1:3], tau[1:3, 1:3])
        }
        mu[1:3] ~ dmnorm(mean[1:3], prec[1:3, 1:3])
        tau[1:3, 1:3] ~ dwish(R[1:3, 1:3], 3)
        sigma2[1:3, 1:3] <- inverse(tau[1:3, 1:3])
        for (i in 1 : 3) {sigma[i] <- sqrt(sigma2[i, i]) }
        tauC ~ dgamma(1.0E-3, 1.0E-3)
        sigmaC <- 1 / sqrt(tauC)
    }

"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from numpy import expm1
from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dmnorm_llf, dwish_llf
from bugs.model import compress_pd, expand_pd, pd_size

#  data: n=7, K=5, x[N], Y[K,n], mean[3], R[3,3], prec[3,3]
_, data = load('../examples/OtreesMVNdata.txt')
globals().update(data)
# init: theta[K,3], mu[3], tau[3,3], tauC
pars = "theta,mu,tau.L,tauC".split(',')
_, init = load('../examples/OtreesMVNinits.txt')
# Convert tau into L for cholesky decomposition, and sample only over the lower
# triangular portion; store diagonally, starting with main diagonal, so that
# it is easy to constrain the diagonal entries to be positive.
init['tau.L'] = compress_pd(init['tau'])
p0, labels = define_pars(init, pars)

n_tau_L = pd_size(3)

def nllf(p):
    theta = p[:K*3].reshape(K, 3)
    mu, tau_L = p[K*3:K*3+3], p[K*3+3:K*3+3+n_tau_L]
    tauC = p[K*3+3+n_tau_L]

    tau = expand_pd(tau_L, 3)

    phi0, phi1, phi2 = exp(theta[:, 0]), expm1(theta[:, 1]), -exp(theta[:, 2])
    eta = phi0[:, None] / (1 + phi1[:, None] * exp(phi2[:, None] * x[None, :]))

    cost = 0
    cost += np.sum(dnorm_llf(Y, eta, tauC))
    cost += sum(dmnorm_llf(theta_k, mu, tau) for theta_k in theta)
    cost += dgamma_llf(tauC, 0.001, 0.001)
    cost += dmnorm_llf(mu, mean, prec)
    cost += dwish_llf(tau, R, 3)

    return -cost

def post(p):
    #theta = p[:K*3].reshape(K, 3)
    mu, tau_L = p[K*3:K*3+3], p[K*3+3:K*3+3+n_tau_L]
    tauC = p[K*3+3+3]
    sigma_C = 1 / sqrt(tauC)
    # TODO: return sigm = sqrt(diag(inv(L.T L)))
    sigma = 1 / tau_L[:3]
    return np.vstack((sigma_C, sigma))
post_vars = ["sigmaC", "sigma[1]", "sigma[2]", "sigma[3]"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem.setp(p0)
problem._bounds[0, K*3+3:K*3+3+n_tau_L] = 0
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "mu[1]", "mu[2]", "mu[3]",
    "sigma[1]", "sigma[2]", "sigma[3]", "sigmaC"]


openbugs_result = """
         mean     sd      MC_error  2.5pc    median   97.5pc  start sample
mu[1]     5.268   0.1353  0.002557   4.997    5.268    5.535  5001  20000
mu[2]     2.194   0.162   0.005338   1.879    2.192    2.515  5001  20000
mu[3]    -5.887   0.1409  0.004828  -6.169   -5.887   -5.61   5001  20000
sigma[1]  0.261   0.1173  0.001881   0.1278   0.2345   0.55   5001  20000
sigma[2]  0.2661  0.1306  0.003348   0.1184   0.2339   0.593  5001  20000
sigma[3]  0.2264  0.1074  0.002382   0.1062   0.2007   0.4947 5001  20000
sigmaC    7.841   1.169   0.02945    5.92     7.713    10.49  5001  20000
"""
