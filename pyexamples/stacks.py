"""
Stacks: robust regression

::

    model
    {
    # Standardise x's and coefficients
        for (j in 1 : p) {
            b[j] <- beta[j] / sd(x[ , j ])
            for (i in 1 : N) {
                z[i, j] <- (x[i, j] -  mean(x[, j])) / sd(x[ , j])
            }
        }
        b0 <- beta0 - b[1] * mean(x[, 1]) - b[2] * mean(x[, 2]) - b[3] * mean(x[, 3])

    # Model
        d <- 4;                                # degrees of freedom for t
    for (i in 1 : N) {
            Y[i] ~ dnorm(mu[i], tau)
    #        Y[i] ~ ddexp(mu[i], tau)
    #        Y[i] ~ dt(mu[i], tau, d)

            mu[i] <- beta0 + beta[1] * z[i, 1] + beta[2] * z[i, 2] + beta[3] * z[i, 3]
            stres[i] <- (Y[i] - mu[i]) / sigma
            outlier[i] <- step(stres[i] - 2.5) + step(-(stres[i] + 2.5) )
        }
    # Priors
        beta0 ~  dnorm(0, 0.00001)
        for (j in 1 : p) {
            beta[j] ~ dnorm(0, 0.00001)      # coeffs independent
    #        beta[j] ~ dnorm(0, phi)     # coeffs exchangeable (ridge regression)
        }
        tau ~ dgamma(1.0E-3, 1.0E-3)
        phi ~ dgamma(1.0E-2,1.0E-2)
    # standard deviation of error distribution
        sigma <- sqrt(1 /  tau)                  # normal errors
    #    sigma <- sqrt(2) / tau                     # double exponential errors
    #    sigma <- sqrt(d / (tau * (d - 2)));    # t errors on d degrees of freedom
    }
"""

import sys
import numpy as np
from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dgamma_llf, ddexp_llf, dt_llf, step

if len(sys.argv) < 2:
    print("""
usage: bumps stacks.py mode

    mode a: normal error model, no ridge regression
    mode b: double exponential error model, no ridge regression
    mode c: T(4) error model, no ridge regression
    mode d: normal error model, with ridge regression
    mode e: double exponential error model, with ridge regression
    mode f: T(4) error model, with ridge regression
""")
    sys.exit()

mode = sys.argv[1]
if mode in 'ad':
    # normal errors
    Y_model = lambda Y, mu, tau: dnorm_llf(Y, mu, tau)
    sigma_model = lambda tau: np.sqrt(1 / tau)
elif mode in 'be':
    # double exponential
    Y_model = lambda Y, mu, tau: ddexp_llf(Y, mu, tau)
    sigma_model = lambda tau: np.sqrt(2) / tau
elif mode in 'cf':
    # t errors on d degrees of freedom
    Y_model = lambda Y, mu, tau: dt_llf(Y, mu, tau, d)
    sigma_model = lambda tau: np.sqrt(d / (tau * (d-2)))
else:
    raise RuntimeError("unknown mode %r"%mode)

if mode in 'abc':
    # coeffs independent
    beta_model = lambda beta, phi: dnorm_llf(beta, 0, 0.00001)
elif mode in 'def':
    # coeffs exchangeable (ridge regression)
    beta_model = lambda beta, phi: dnorm_llf(beta, 0, phi)
else:
    raise RuntimeError("unknown mode %r"%mode)

p=N=Y=x=None # silence lint
#  data: p=3, N=21, Y[N], x[N,p]
globals().update(load('../examples/Stacksdata.txt')[1])
del p  # using p as parameter vector to nllf; avoid conflict
# inits: beta0, beta[p], tau, phi
_, init = load('../examples/Stacksinits.txt')
pars = list(sorted(init))
p0, labels = define_pars(init, pars)

d = 4  # Student T degrees of freedom
x_bar = np.mean(x, axis=0)
x_dev = np.std(x, axis=0, ddof=1)
z = (x - x_bar) / x_dev

def nllf(p):
    beta0, beta, tau, phi = p[0], p[1:4], p[4], p[5]

    b = beta / x_dev
    mu = beta0 + np.dot(z, beta)
    #print("=>", beta.shape, b.shape, mu.shape)

    cost = 0

    #cost += np.sum(dnorm_llf(Y, mu, tau)) # normal errors
    #cost += np.sum(ddexp_llf(Y, mu, tau)) # double exponential
    #cost += np.sum(dt_llf(Y, mu, tau, d)) # t(d) errors
    cost += np.sum(Y_model(Y, mu, tau))

    #cost += np.sum(dnorm_llf(beta, 0, 0.00001)) # coeffs independent
    #cost += np.sum(dnorm_llf(beta, 0, phi)) # coeffs exchangeable (ridge regression)
    cost += np.sum(beta_model(beta, phi))

    cost += dnorm_llf(beta0, 0, 0.00001)
    cost += dgamma_llf(tau, 1e-3, 1e-3)
    cost += dgamma_llf(phi, 1e-2, 1e-2)

    return -cost

def post(p):
    beta0, beta, tau, phi = p[0], p[1:4], p[4], p[5]

    # Standard deviation of error distribution
    #a,d sigma = sqrt(1 / tau) # normal errors
    #b,e sigma = sqrt(2) / tau # double exponential
    #c,f sigma = sqrt(d / (tau * (d-2))) # t(d) errors
    sigma = sigma_model(tau)

    b = beta / x_dev[:, None]
    mu = beta0 + np.dot(z, beta)
    #print("=>", beta.shape, b.shape, mu.shape)

    b0 = beta0 - np.dot(x_bar, b)
    #print(b0.shape)

    stres = (Y[:, None] - mu)/sigma
    outlier = step(stres - 2.5) + step(-(stres + 2.5))

    return np.vstack((sigma, b, b0, outlier))

post_vars = (["sigma", "b[1]", "b[2]", "b[3]", "b0"]
             + ["outlier[%d]"%k for k in range(1, N+1)])

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "b[1]", "b[2]", "b[3]",
    "b0",
    "outlier[1]", "outlier[3]", "outlier[4]", "outlier[21]",
    "sigma",
]


openbugs_result = """
a) Normal error
             mean    sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.7185  0.1436  0.002928   0.4292    0.72     0.9992  1001  10000
b[2]         1.29    0.3915  0.007735   0.5136    1.29     2.068   1001  10000
b[3]        -0.1561  0.1653  0.00218   -0.4799   -0.1571   0.1678  1001  10000
b0         -39.64   12.63    0.1428   -64.38    -39.78   -14.35    1001  10000
outlier[3]   0.0101  0.09999 9.779E-4   0.0       0.0      0.0     1001  10000
outlier[4]   0.056   0.2299  0.002399   0.0       0.0      1.0     1001  10000
outlier[21]  0.3324  0.4711  0.006548   0.0       0.0      1.0     1001  10000
sigma        3.385   0.63    0.007726   2.42      3.302    4.853   1001  10000

b) Double exponential error
             mean    sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.8404  0.1373  .007101    0.5774    0.8433   1.11    1001  10000
b[2]         0.7292  0.3559  0.01692    0.1664    0.6938   1.479   1001  10000
b[3]        -0.1197  0.1175  0.003615  -0.3582   -0.1189   0.1148  1001  10000
b0         -38.41    8.756   0.2011   -55.93    -38.3    -20.77    1001  10000
outlier[1]   0.0387  0.1929  0.002834   0.0       0.0      1.0     1001  10000
outlier[3]   0.0541  0.2262  0.003018   0.0       0.0      1.0     1001  10000
outlier[4]   0.3054  0.4606  0.009034   0.0       0.0      1.0     1001  10000
outlier[21]  0.618   0.4859  0.01354    0.0       1.0      1.0     1001  10000
sigma        3.47    0.865   0.01688    2.178     3.34     5.519   1001  10000

c) t4 error
             mean    sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.8397  0.1433  0.003869   0.5489    0.8412   1.119   1001  10000
b[2]         0.8478  0.38    0.009911   0.1528    0.8278   1.661   1001  10000
b[3]        -0.1255  0.1302  0.00189   -0.3871   -0.1238   0.1325  1001  10000
b0         -40.24    9.872   0.1164   -60.58    -40.23   -20.87    1001  10000
outlier[3]   0.0385  0.1924  0.002889   0.0       0.0      1.0     1001  10000
outlier[4]   0.2418  0.4282  0.007181   0.0       0.0      1.0     1001  10000
outlier[21]  0.5942  0.491   0.01023    0.0       1.0      1.0     1001  10000
sigma        3.495   0.8837  0.01689    2.145     3.374    5.522   1001  10000

d) Normal error ridge regression
             mean    sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.6805  0.1363  0.002577   0.4074    0.6823   0.944   1001  10000
b[2]         1.319   0.3663  0.006692   0.5942    1.318    2.047   1001  10000
b[3]        -0.126   0.1664  0.002176  -0.4508   -0.1273   0.2074  1001  10000
b0         -40.54   12.7     0.1469   -65.81    -40.63   -15.17    1001  10000
outlier[3]   0.0173  0.1304  0.001225   0.0       0.0      0.0     1001  10000
outlier[4]   0.0479  0.2136  0.002289   0.0       0.0      1.0     1001  10000
outlier[21]  0.281   0.4495  0.005972   0.0       0.0      1.0     1001  10000
sigma        3.404   0.624   0.007759   2.44      3.312    4.885   1001  10000

e) Double exponential error  ridge regression
             mean    sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.7891  0.1275  0.005877   0.5118    0.7955   1.021   1001  10000
b[2]         0.7975  0.3406  0.01479    0.2196    0.7694   1.529   1001  10000
b[3]        -0.09416 0.1169  0.00329   -0.327    -0.09324  0.14    1001  10000
b0         -39.03    8.87    0.1869   -57.3     -38.96   -21.59    1001  10000
outlier[1]   0.0621  0.2413  0.004219   0.0       0.0      1.0     1001  10000
outlier[3]   0.0776  0.2675  0.004324   0.0       0.0      1.0     1001  10000
outlier[4]   0.2902  0.4539  0.009295   0.0       0.0      1.0     1001  10000
outlier[21]  0.539   0.4985  0.01375    0.0       1.0      1.0     1001  10000
sigma        3.498   0.8781  0.01731    2.19      3.36     5.56    1001  10000

f) t4 error ridge regression
            mean     sd      MC_error  2.5pc     median   97.5pc   start sample
b[1]         0.7921  0.1412  0.003735   0.5       0.7969   1.054   1001  10000
b[2]         0.9121  0.3663  0.009125   0.2516    0.8905   1.686   1001  10000
b[3]        -0.1069  0.1304  0.001909  -0.3729   -0.107    0.153   1001  10000
b0         -40.39    9.895   0.1172   -60.54    -40.24   -21.0     1001  10000
outlier[1]   0.0349  0.1835  0.002632   0.0       0.0      1.0     1001  10000
outlier[3]   0.0476  0.2129  0.003032   0.0       0.0      1.0     1001  10000
outlier[4]   0.2169  0.4121  0.00781    0.0       0.0      1.0     1001  10000
outlier[21]  0.5202  0.4996  0.0105     0.0       1.0      1.0     1001  10000
sigma        3.519   0.874   0.01806    2.152     3.4      5.547   1001  10000
"""
