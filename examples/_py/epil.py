"""
Epilepsy: repeated measures on Poisson counts

::

    model
    {
        for (i in 1 : N) {
            theta[i] ~ dgamma(alpha, beta)
            lambda[i] <- theta[i] * t[i]
            x[i] ~ dpois(lambda[i])
        }
        alpha ~ dexp(1)
        beta ~ dgamma(0.1, 1.0)
    }
        model
    {
        for(j in 1 : N) {
            for(k in 1 : T) {
                log(mu[j, k]) <- a0 + alpha.Base * (log.Base4[j] - log.Base4.bar)
                      + alpha.Trt * (Trt[j] - Trt.bar)
                      + alpha.BT  * (BT[j] - BT.bar)
                      + alpha.Age * (log.Age[j] - log.Age.bar)
                      + alpha.V4  * (V4[k] - V4.bar)
                      + b1[j] + b[j, k]
                y[j, k] ~ dpois(mu[j, k])
                b[j, k] ~ dnorm(0.0, tau.b);       # subject*visit random effects
            }
            b1[j]  ~ dnorm(0.0, tau.b1)        # subject random effects
            BT[j] <- Trt[j] * log.Base4[j]    # interaction
            log.Base4[j] <- log(Base[j] / 4) log.Age[j] <- log(Age[j])
        }

    # covariate means:
        log.Age.bar <- mean(log.Age[])
        Trt.bar  <- mean(Trt[])
        BT.bar <- mean(BT[])
        log.Base4.bar <- mean(log.Base4[])
        V4.bar <- mean(V4[])
    # priors:

        a0 ~ dnorm(0.0,1.0E-4)
        alpha.Base ~ dnorm(0.0,1.0E-4)
        alpha.Trt  ~ dnorm(0.0,1.0E-4);
        alpha.BT   ~ dnorm(0.0,1.0E-4)
        alpha.Age  ~ dnorm(0.0,1.0E-4)
        alpha.V4   ~ dnorm(0.0,1.0E-4)
        tau.b1     ~ dgamma(1.0E-3,1.0E-3); sigma.b1 <- 1.0 / sqrt(tau.b1)
        tau.b      ~ dgamma(1.0E-3,1.0E-3); sigma.b  <- 1.0/  sqrt(tau.b)

    # re-calculate intercept on original scale:
        alpha0 <- a0 - alpha.Base * log.Base4.bar - alpha.Trt * Trt.bar
        - alpha.BT * BT.bar - alpha.Age * log.Age.bar - alpha.V4 * V4.bar
    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dpois_llf, dnorm_llf, dgamma_llf

#  data: N=59, T=4, y[N,T], Trt[N], Base[N], Age[N], V4[T]
globals().update(load('../Epildata.txt')[1])
# inits: a0, alpha.Base, alpha.Trt, alpha.BT, alpha.Age, alpha.V4, tau.b1, tau.b
_, init = load('../Epilinits.txt')
pars = list(sorted(init)) + ["b", "b1"]
init["b"] = np.zeros((N, T))
init["b1"] = np.zeros(N)
p0, labels = define_pars(init, pars)

log_Age = log(Age)
log_Base4 = log(Base / 4)
BT = Trt * log_Base4

log_Age_bar = np.mean(log_Age)
log_Base4_bar = np.mean(log_Base4)
V4_bar = np.mean(V4)
Trt_bar = np.mean(Trt)
BT_bar = np.mean(BT)

def nllf(p):
    a0, alpha_Base, alpha_Trt, alpha_BT, alpha_Age, alpha_V4, tau_b1, tau_b = p[:8]
    b = p[8:8+N*T].reshape(N, T)
    b1 = p[8+N*T:8+N*T+N]

    mu = exp(
        a0
        + alpha_Base * (log_Base4[:, None] - log_Base4_bar)
        + alpha_BT * (BT[:, None] - BT_bar)
        + alpha_Age * (log_Age[:, None] - log_Age_bar)
        + alpha_V4 * (V4[None, :] - V4_bar)
        + b1[:, None]
        + b
    )

    cost = 0
    cost += np.sum(dpois_llf(y, mu))
    cost += np.sum(dnorm_llf(b, 0.0, tau_b))
    cost += np.sum(dnorm_llf(b1, 0.0, tau_b1))
    cost += dnorm_llf(a0, 0, 1e-4)
    cost += dnorm_llf(alpha_Base, 0, 1e-4)
    cost += dnorm_llf(alpha_Trt, 0, 1e-4)
    cost += dnorm_llf(alpha_BT, 0, 1e-4)
    cost += dnorm_llf(alpha_Age, 0, 1e-4)
    cost += dnorm_llf(alpha_V4, 0, 1e-4)
    cost += dgamma_llf(tau_b1, 1e-3, 1e-3)
    cost += dgamma_llf(tau_b, 1e-3, 1e-3)

    return -cost

def post(p):
    a0, alpha_Base, alpha_Trt, alpha_BT, alpha_Age, alpha_V4, tau_b1, tau_b = p[:8]
    alpha0 = (a0
              - alpha_Base * log_Base4_bar
              - alpha_Trt * Trt_bar
              - alpha_BT * BT_bar
              - alpha_Age * log_Age_bar
              - alpha_V4 * V4_bar)
    sigma_b1 = 1.0 / sqrt(tau_b1)
    sigma_b = 1.0/ sqrt(tau_b)
    return [alpha0, sigma_b1, sigma_b]
post_vars = ["alpha0", "sigma.b1", "sigma.b"]

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "alpha.Age", "alpha.BT", "alpha.Base", "alpha.Trt", "alpha.V4",
    "alpha0", "sigma.b", "sigma.b1",
]


openbugs_result = """
            mean     sd      MC_error  2.5pc     median    97.5pc   start sample
alpha.Age   0.4816   0.3626   0.01466   -0.2639   0.4889    1.182    2001  10000
alpha.BT    0.3484   0.2146   0.01183   -0.06699  0.3473    0.7835   2001  10000
alpha.Base  0.8933   0.1403   0.007015   0.6209   0.8918    1.172    2001  10000
alpha.Trt  -0.9485   0.4318   0.02043   -1.808   -0.9501   -0.1064   2001  10000
alpha.V4   -0.1047   0.08826  0.001711  -0.2781  -0.1043    0.06825  2001  10000
alpha0     -1.407    1.253    0.05115   -3.832   -1.436     1.097    2001  10000
sigma.b     0.3627   0.04409  0.001594   0.2804   0.3609    0.4535   2001  10000
sigma.b1    0.4979   0.07081  0.001797   0.3722   0.4939    0.6469   2001  10000
"""