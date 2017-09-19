"""
Leuk: Cox regression

::

    model
    {
    # Set up data
        for(i in 1:N) {
            for(j in 1:T) {
    # risk set = 1 if obs.t >= t
                Y[i,j] <- step(obs.t[i] - t[j] + eps)
    # counting process jump = 1 if obs.t in [ t[j], t[j+1] )
    #                      i.e. if t[j] <= obs.t < t[j+1]
                dN[i, j] <- Y[i, j] * step(t[j + 1] - obs.t[i] - eps) * fail[i]
            }
        }
    # Model
        for(j in 1:T) {
            for(i in 1:N) {
                dN[i, j]   ~ dpois(Idt[i, j])              # Likelihood
                Idt[i, j] <- Y[i, j] * exp(beta * Z[i]) * dL0[j]     # Intensity
            }
            dL0[j] ~ dgamma(mu[j], c)
            mu[j] <- dL0.star[j] * c    # prior mean hazard

    # Survivor function = exp(-Integral{l0(u)du})^exp(beta*z)
            S.treat[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * -0.5));
            S.placebo[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * 0.5));
        }
        c <- 0.001
        r <- 0.1
        for (j in 1 : T) {
            dL0.star[j] <- r * (t[j + 1] - t[j])
        }
        beta ~ dnorm(0.0,0.000001)
    }
"""

from bumps.names import *
from numpy import exp, sqrt
from bugs.parse import load, define_pars
from bugs.model import dnorm_llf, dpois_llf, dgamma_llf, step

#  data: N=42, T=17, eps, obs.t[N], fail[T], Z[N], t[T+1]
vars = "N,T,eps,obs.t,fail,Z,t".split(',')
_, data = load('../examples/Leukdata.txt')
N, T, eps, obs_t, fail, Z, t = (data[p] for p in vars)
# init: beta, dL0[T]
pars = "beta,dL0".split(',')
_, init = load('../examples/Leukinits.txt')
p0, labels = define_pars(init, pars)

# constants
c = 0.001
r = 0.1
dL0_star = r * np.diff(t)
mu = dL0_star * c

def nllf(p):
    beta, dL0 = p[0], p[1:]
    Y = step(obs_t[:N, None] - t[None, :T] + eps)
    dN = Y * step(t[None, 1:] - obs_t[:, None] - eps) * fail[:, None]
    Idt = Y*exp(beta * Z[:, None]) * dL0[None, :]

    cost = 0
    cost += np.sum(dpois_llf(dN, Idt))
    cost += np.sum(dgamma_llf(dL0, mu, c))
    cost += dnorm_llf(beta, 0.0, 0.000001)

    return -cost

S_TREAT = ["S.treat[%d]"%j for j in range(1, T+1)]
S_PLACEBO = ["S.placebo[%d]"%j for j in range(1, T+1)]
def post(p):
    beta, dL0 = p[0], p[1:]
    S_placebo = exp(-np.cumsum(dL0, axis=0)) ** exp(beta * 0.5)
    S_treat = exp(-np.cumsum(dL0, axis=0)) ** exp(beta * -0.5)
    return np.vstack((S_placebo, S_treat))
post_vars = S_PLACEBO + S_TREAT

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)

problem.setp(p0)
problem.derive_vars = post, post_vars
problem.visible_vars = post_vars + ["beta"]


openbugs_result = """
              mean    sd       MC_error  2.5pc    median   97.5pc   start    sample
S.placebo[1]  0.9268  0.04934  4.916E-4  0.8041   0.937    0.9909   1001  10000
S.placebo[2]  0.8541  0.06696  7.232E-4  0.703    0.8632   0.9573   1001  10000
S.placebo[3]  0.8169  0.0736   8.222E-4  0.6537   0.825    0.9357   1001  10000
S.placebo[4]  0.7433  0.08401  9.528E-4  0.5638   0.7496   0.8854   1001  10000
S.placebo[5]  0.6705  0.09086  0.001067  0.4835   0.6753   0.8349   1001  10000
S.placebo[6]  0.5634  0.09744  0.001146  0.3686   0.5647   0.7469   1001  10000
S.placebo[7]  0.5303  0.09811  0.001146  0.3362   0.5301   0.7164   1001  10000
S.placebo[8]  0.4147  0.095    0.001148  0.2369   0.4117   0.6058   1001  10000
S.placebo[9]  0.3816  0.0943   0.001144  0.2052   0.3781   0.5755   1001  10000
S.placebo[10] 0.3209  0.09054  0.001116  0.1576   0.3154   0.509    1001  10000
S.placebo[11] 0.2592  0.08513  0.0011    0.1136   0.2526   0.4422   1001  10000
S.placebo[12] 0.2266  0.08184  0.001113  0.08899  0.2191   0.4057   1001  10000
S.placebo[13] 0.1963  0.07856  0.001105  0.06861  0.188    0.3712   1001  10000
S.placebo[14] 0.167   0.07434  0.001102  0.04946  0.1575   0.3368   1001  10000
S.placebo[15] 0.1407  0.06902  0.001008  0.03725  0.131    0.2991   1001  10000
S.placebo[16] 0.08767 0.05583  8.095E-4  0.01418  0.07627  0.2262   1001  10000
S.placebo[17] 0.0452  0.04025  6.235E-4  0.002561 0.03378  0.1518   1001  10000
S.treat[1]    0.9825  0.01412  2.133E-4  0.9466   0.9862   0.9981   1001  10000
S.treat[2]    0.9642  0.02157  3.779E-4  0.9106   0.9689   0.9921   1001  10000
S.treat[3]    0.9543  0.02543  4.507E-4  0.8909   0.9593   0.9883   1001  10000
S.treat[4]    0.9339  0.03237  6.151E-4  0.8551   0.9397   0.9793   1001  10000
S.treat[5]    0.9121  0.03936  7.785E-4  0.8185   0.9183   0.9699   1001  10000
S.treat[6]    0.8766  0.04956  0.001022  0.7622   0.884    0.9527   1001  10000
S.treat[7]    0.8645  0.053    0.001138  0.7422   0.8721   0.9464   1001  10000
S.treat[8]    0.8171  0.0657   0.001441  0.6685   0.8251   0.9239   1001  10000
S.treat[9]    0.8016  0.06942  0.001549  0.646    0.8093   0.9155   1001  10000
S.treat[10]   0.7703  0.07732  0.001712  0.5967   0.7785   0.8986   1001  10000
S.treat[11]   0.7332  0.08575  0.001921  0.548    0.7411   0.8767   1001  10000
S.treat[12]   0.7106  0.09033  0.001979  0.5219   0.7187   0.8641   1001  10000
S.treat[13]   0.6872  0.09456  0.002055  0.4853   0.6945   0.8497   1001  10000
S.treat[14]   0.6616  0.09872  0.00215   0.4527   0.6681   0.8332   1001  10000
S.treat[15]   0.6353  0.1027   0.002258  0.4203   0.6417   0.8189   1001  10000
S.treat[16]   0.5662  0.1121   0.00248   0.3379   0.5705   0.771    1001  10000
S.treat[17]   0.476   0.1196   0.002402  0.247    0.4754   0.7064   1001  10000
beta          1.532   0.4246   0.0108    0.709    1.522    2.387    1001  10000
"""
