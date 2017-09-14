"""
Equiv: bioequivalence in a cross-over trial

::

    model
    {
        for( k in 1 : P ) {
            for( i in 1 : N ) {
                Y[i , k] ~ dnorm(m[i , k], tau1)
                m[i , k] <- mu + sign[T[i , k]] * phi / 2 + sign[k] * pi / 2 + delta[i]
                T[i , k] <- group[i] * (k - 1.5) + 1.5
            }
        }
        for( i in 1 : N ) {
            delta[i] ~ dnorm(0.0, tau2)
        }
        tau1 ~ dgamma(0.001, 0.001) sigma1 <- 1 / sqrt(tau1)
        tau2 ~ dgamma(0.001, 0.001) sigma2 <- 1 / sqrt(tau2)
        mu ~ dnorm(0.0, 1.0E-6)
        phi ~ dnorm(0.0, 1.0E-6)
        pi ~ dnorm(0.0, 1.0E-6)
        theta <- exp(phi)
        equiv <- step(theta - 0.8) - step(theta - 1.2)
    }
"""

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, step

#  data: N=10, P=2, group[N], Y[N,P], sign[2]
vars = "N,P,group,Y,sign".split(',')
_, data = load('../Equivdata.txt')
N, P, group, Y, sign = (data[p] for p in vars)
# init: alpha, beta, gamma, tau
pars = "mu,phi,pi,tau1,tau2,delta".split(',')
_, init = load('../Equivinits.txt')
init["delta"] = np.zeros(N)
p0, labels = define_pars(init, pars)

def pre():
    k = np.arange(1.0, P+1)
    T = group[:, None] * (k[None, :] - 1.5) + 1.5
    k = np.asarray(k, 'i')
    T = np.asarray(T, 'i')
    sign_k = sign[k - 1]
    sign_T = sign[T - 1]
    return sign_k, sign_T
sign_k, sign_T = pre()

def nllf(p):
    mu, phi, pi, tau1, tau2 = p[:5]
    delta = p[5:]
    m = mu + sign_T * phi / 2 + sign_k[None, :] * pi / 2 + delta[:, None]

    cost = 0
    cost += np.sum(dnorm_llf(Y, m, tau1))
    cost += np.sum(dnorm_llf(delta, 0.0, tau2))
    cost += dgamma_llf(tau1, 0.001, 0.001)
    cost += dgamma_llf(tau2, 0.001, 0.001)
    cost += dnorm_llf(mu, 0, 1e-6)
    cost += dnorm_llf(phi, 0, 1e-6)
    cost += dnorm_llf(pi, 0, 1e-6)

    return -cost

def post(p):
    mu, phi, pi, tau1, tau2 = p[:5]
    theta = exp(phi)
    equiv = step(theta - 0.8) - step(theta - 1.2)
    sigma1 = 1 / sqrt(tau1)
    sigma2 = 1 / sqrt(tau2)
    return [equiv, sigma1, sigma2, theta]
post_vars = ["equiv", "sigma1", "sigma2", "theta"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem._bounds[0, 3:5] = 0  # tau1, tau2 bounded below by 0
problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = ["equiv", "mu", "phi", "pi", "sigma1", "sigma2", "theta"]

openbugs_result = """
        mean     sd        MC_error  2.5pc     median   97.5pc   start   sample
equiv   0.998    0.04468   4.161E-4   1.0       1.0      1.0      1001   10000
mu      1.436    0.05751   0.001952   1.323     1.436    1.551    1001   10000
phi    -0.008613 0.05187   4.756E-4  -0.1132   -0.00806  0.09419  1001   10000
pi     -0.18     0.05187   5.131E-4  -0.2841   -0.1801  -0.07464  1001   10000
sigma1  0.1102   0.03268   9.532E-4   0.06501   0.1035   0.1915   1001   10000
sigma2  0.1412   0.05366   0.00141    0.04701   0.1359   0.2666   1001   10000
theta   0.9928   0.05145   4.74E-4    0.893     0.992    1.099    1001   10000
"""
