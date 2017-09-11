"""
Birats: a bivariate normal hierarchical model

::

    model
    {
        for( i in 1 : N ) {
            beta[i , 1 : 2] ~ dmnorm(mu.beta[], R[ , ])
            for( j in 1 : T ) {
                Y[i, j] ~ dnorm(mu[i , j], tauC)
                mu[i, j] <- beta[i, 1] + beta[i, 2] * x[j]
            }
        }

        mu.beta[1 : 2] ~ dmnorm(mean[], prec[ , ])
        R[1 : 2 , 1 : 2] ~ dwish(Omega[ , ], 2)
        tauC ~ dgamma(0.001, 0.001)
        sigma <- 1 / sqrt(tauC)
    }

Because the precision matrix $R$ must be positive definite, we cannot sample
its elements independently.  Instead, we sample from a lower triangular
matrix L with positive diagonal elements, and use $R = L L^T$.  The matrix
$L$ is initialized with the cholesky decomposition of the initial $R$ value.
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *

from math import sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, dmnorm_llf, dwish_llf
import scipy.sparse


# N=30, T=5, x[T], Omega[2,2], mean[2], prec[2,2], Y[N,T]
_, data = load('../BiRatsdata.txt')
x, N, T = data["x"], data["N"], data["T"]
Omega, mean, prec, Y = data["Omega"], data["mean"], data["prec"], data["Y"]

# Convert R into L for cholesky decomposition, and sample only over the lower
# triangular portion; store diagonally, starting with main diagonal, so that
# it is easy to constrain the diagonal entries to be positive.
pars =  'mu.beta,tauC,beta,L'.split(',')
_, init = load('../BiRatsinits.txt')
L = np.linalg.cholesky(init['R'])
init['L'] = np.array([L[0, 0], L[1, 1], L[1, 0]])
p0, labels = define_pars(init, pars)


def birats(p):
    mu_beta, tauC = p[0:2], p[2]
    beta = p[3:30*2+3].reshape(30, 2)
    L11, L22, L21 = p[30*2+3:30*2+3+3]
    R = np.array([[L11*L11, L11*L21], [L11*L21, L21*L21 + L22*L22]])
    mu = beta[:, 0:1] + beta[:, 1:2]*x[None, :]

    cost = 0.
    cost += sum(dmnorm_llf(beta_k, mu_beta, R) for beta_k in beta)
    cost += np.sum(dnorm_llf(Y, mu, tauC))
    cost += np.sum(dmnorm_llf(mu_beta, mean, prec))
    cost += np.sum(dwish_llf(R, Omega, 2))
    cost += dgamma_llf(tauC, 0.001, 0.001)
    return -cost

def post(p):
    tauC = p[2]
    sigma = 1./np.sqrt(tauC)
    L11, L21, L22 = p[30*2+3:30*2+3+3]
    R = [L11*L11, L11*L21, L11*L21, L21*L21 + L22*L22]
    return [sigma] + R
post_vars = ["sigma", "R[1,1]", "R[2,1]", "R[1,2]", "R[2,2]"]

problem = DirectProblem(birats, p0, labels=labels, dof=T*N-len(p0))
problem._bounds[0,2] = 0 # limit tau_c to [0,inf)
problem._bounds[0,-3:-1] = 0 # limit R=LL^T diagonal to positive values
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["mu.beta[1]", "mu.beta[2]", "sigma"]


openbugs_result = """
           mean    sd        MC_error  2.5pc    median   97.5pc   start sample
mu.beta[1] 106.6   2.325     0.03173   101.9    106.6    111.1    1001  10000
mu.beta[2]   6.186 0.1062    0.001397    5.974    6.186    6.395  1001  10000
sigma        6.156 0.4746    0.008901    5.326    6.121    7.181  1001  10000
"""
