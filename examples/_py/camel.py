"""
Camel: Multivariate normal with structured missing data

::

    model
    {
        for (i in 1 : N){
            Y[i, 1 : 2] ~ dmnorm(mu[], tau[ , ])
        }
        mu[1] <- 0
        mu[2] <- 0
        tau[1 : 2,1 : 2] ~ dwish(R[ , ], 2)
        R[1, 1] <- 0.001
        R[1, 2] <- 0
        R[2, 1] <- 0;
        R[2, 2] <- 0.001
        Sigma2[1 : 2,1 : 2] <- inverse(tau[ , ])
        rho <- Sigma2[1, 2] / sqrt(Sigma2[1, 1] * Sigma2[2, 2])
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

# data: N=12, Y[N,2]; note that many Y are fitted
_, data = load('../Cameldata.txt')
N, Y = data["N"], data["Y"]

# Convert R into L for cholesky decomposition, and sample only over the lower
# triangular portion; store diagonally, starting with main diagonal, so that
# it is easy to constrain the diagonal entries to be positive.
pars =  'tau.L,Y.fit'.split(',')
# init: tau[2,2], Y[N,2]
_, init = load('../Camelinits.txt')
L = np.linalg.cholesky(init['tau'])
init['tau.L'] = np.array([L[0, 0], L[1, 1], L[1, 0]])
Y_fit_index = np.nonzero(np.isfinite(init['Y']))
init['Y.fit'] = init['Y'][Y_fit_index]
p0, labels = define_pars(init, pars)
active = np.isfinite(p0)

R = np.array([[0.001, 0], [0, 0.001]])
mu = np.array([0., 0.])

def nllf(p):
    L11, L22, L21 = p[0:3]
    Y[Y_fit_index] = p[3:]
    tau = np.array([[L11*L11, L11*L21], [L11*L21, L21*L21+L22*L22]])

    cost = 0.
    cost += sum(dmnorm_llf(Y[k], mu, tau) for k in range(N))
    cost += dwish_llf(tau, R, 2)
    return -cost

def matrix_labels(base, n=2):
    return ["%s[%d,%d]"%(base, i+1, j+1) for i in range(n) for j in range(n)]

def post(p):
    L11, L22, L21 = p[0:3]
    tau = np.array([L11*L11, L11*L21, L11*L21, L21*L21+L22*L22])
    # convert columns to an array of tau arrays, invert the arrays, and convert
    # back to columns.  Since reshape and transpose create views on the
    # original array, there is no need to worry about extra memory.
    Sigma2 = np.linalg.inv(tau.reshape(2, 2, -1).transpose(2, 0, 1)).transpose(1, 2, 0).reshape(4, -1)
    # Construct rho
    rho = Sigma2[1] / np.sqrt(Sigma2[0] * Sigma2[3])
    return np.vstack((Sigma2, rho, tau))
post_vars = matrix_labels("Sigma2", 2) + ["rho"] + matrix_labels("tau", 2)

problem = DirectProblem(nllf, p0, labels=labels, dof=100)
problem._bounds[0,:2] = 0 # force tau.L diagonal to be positive
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = post_vars


openbugs_result = """
            mean      sd     MC_error   2.5pc    median    97.5pc   start sample
Sigma2[1,1]  3.203    2.103  0.01152     1.124    2.653     8.58    1001  100000
Sigma2[1,2]  0.03249  2.479  0.03659    -4.676    0.08471   4.695   1001  100000
Sigma2[2,1]  0.03249  2.479  0.03659    -4.676    0.08471   4.695   1001  100000
Sigma2[2,2]  3.199    2.074  0.01134     1.112    2.66      8.491   1001  100000
rho          0.01128  0.6585 0.01105    -0.9079   0.04066   0.9079  1001  100000
tau[1,1]     0.8617   0.5155 0.003321    0.2217   0.7401    2.189   1001  100000
tau[1,2]    -0.009638 0.7154 0.01092    -1.421   -0.01699   1.411   1001  100000
tau[2,1]    -0.009638 0.7154 0.01092    -1.421   -0.01699   1.411   1001  100000
tau[2,2]     0.8625   0.5164 0.003253    0.2229   0.7406    2.201   1001  100000
"""
