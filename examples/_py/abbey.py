"""
Abbey National: A stable distribution

::

    model
    {
        for(i in 2 : N){
            z[i] ~ dstable(alpha, beta, gamma, delta)
            z[i] <- price[i] / price[i - 1] - 1
        }

        alpha ~ dunif(1.1, 2)
        beta ~ dunif(-1, 1)
        gamma ~ dunif(-0.05, 0.05)
        delta ~ dunif(0.001, 0.5)

        mean.z <- mean(z[2:50])
        sd.z <- sd(z[2:50])
    }

Note: this posterior reported by Bumps does not match that from OpenBUGS,
however it does match that from a numerical integral over a dense mesh
for the Levy Stable distribution used for this model.  Some simple checks
of the dstable in OpenBUGS (using z ~ dstable(alpha, beta, gamma, delta)
with fixed parameters) show that it is consistent with this implementation
of the Levy distribution.
"""
from __future__ import division

#raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dstable_llf, mean, sd

N = 50
price = np.array([
    296, 296, 300, 302, 300, 304, 303, 299, 293, 294, 294, 293, 295,
    287, 288, 297, 305, 307, 307, 304, 303, 304, 304, 309, 309, 309,
    307, 306, 304, 300, 296, 301, 298, 295, 295, 293, 292, 297, 294,
    293, 306, 303, 301, 303, 308, 305, 302, 301, 297, 299,
    ])
z = price[1:] / price[:-1] - 1
mean_z, sd_z = mean(z), sd(z)

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

def brute_force():
    import numpy as np
    from numpy import exp

    N = 12
    alpha = np.linspace(1.1, 2, N-1)
    beta = np.linspace(-1+1e-10, 1-1e-10, N+1)
    gamma = np.linspace(-0.05, 0.05, N-2)
    delta = np.linspace(0.001, 0.5, N+2)

    #N = 40  # about 3 hrs
    #alpha = np.linspace(1.1, 2, N/2)
    #beta = np.linspace(-1+1e-10, 1-1e-10, N/2)
    #gamma = np.linspace(-0.005, 0.02, 2*N)
    #delta = np.linspace(0.001, 0.03, 2*N)

    dalpha = alpha[1]-alpha[0]
    dbeta = beta[1]-beta[0]
    dgamma = gamma[1]-gamma[0]
    ddelta = delta[1]-delta[0]
    A, B = np.meshgrid(alpha, beta)
    scale = np.empty((len(gamma), len(delta)))
    gamma_delta = np.empty((len(gamma), len(delta)))
    alpha_beta = np.zeros((len(alpha), len(beta)))
    alpha_hist = np.zeros(len(alpha))
    beta_hist = np.zeros(len(beta))
    llf = np.empty((len(alpha), len(beta)))
    for i, g in enumerate(gamma):
        print("i", i)
        for j, d in enumerate(delta):
            """
            llf = np.zeros((len(alpha), len(beta)))
            for zi in z:
                llf += np.log(dstable_llf(zi, A, B, g, d))
            """
            for k, a in enumerate(alpha):
                for l, b in enumerate(beta):
                    llf[k, l] = -abbey(a, b, g, d)
                    if not np.isfinite(llf[k,l]):
                        print("levy(z,%g,%g,%g,%g)"%(a, b, g, d))
            #llf = -abbey(A, B, g, d)
            llf_max = llf.max()
            alpha_beta += exp(llf)
            gamma_delta[i, j] = np.trapz(np.trapz(exp(llf-llf_max), dx=dbeta), dx=dalpha)
            scale[i, j] = llf_max
            alpha_hist += np.trapz(exp(llf-llf_max), dx=dalpha)*exp(llf_max)
            beta_hist += np.trapz(exp(llf-llf_max).T, dx=dbeta)*exp(llf_max)


    norm = scale.max()
    gamma_delta *= exp(scale-norm)
    gamma_hist = np.trapz(gamma_delta, dx=dgamma)
    delta_hist = np.trapz(gamma_delta.T, dx=ddelta)
    alpha_beta *= exp(-norm)
    alpha_hist *= exp(-norm)
    beta_hist *= exp(-norm)

    import json
    with open('abbey_pdf.json', 'w') as fd:
        state = dict(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            alpha_beta=alpha_beta,
            gamma_delta=gamma_delta,
            alpha_hist=alpha_hist,
            beta_hist=beta_hist,
            gamma_hist=gamma_hist,
            delta_hist=delta_hist,
        )
        state = {k: v.tolist() for k, v in state.items()}
        json.dump(state, fd)

    #print(integral)
    #print(scale)
    #print(norm)
    import matplotlib.pyplot as plt
    #plt.pcolor(edges(delta), edges(gamma), np.log10(gamma_delta))
    #plt.colorbar()
    #plt.xlabel("delta")
    #plt.ylabel("gamma")
    #plt.figure()
    plt.pcolor(edges(delta), edges(gamma), gamma_delta)
    plt.colorbar()
    plt.xlabel("delta")
    plt.ylabel("gamma")
    plt.figure()
    plt.pcolor(edges(beta), edges(alpha), alpha_beta)
    plt.colorbar()
    plt.xlabel("beta")
    plt.ylabel("alpha")
    plt.figure()
    plt.subplot(221); plt.plot(alpha, alpha_hist); plt.xlabel("alpha")
    plt.subplot(222); plt.plot(beta, beta_hist); plt.xlabel("beta")
    plt.subplot(223); plt.plot(gamma, gamma_hist); plt.xlabel("gamma")
    plt.subplot(224); plt.plot(delta, delta_hist); plt.xlabel("delta")
    plt.show()

def edges(c):
    mid = (c[:-1] + c[1:])/2
    return np.hstack((2*c[0] - mid[0], mid, 2*c[-1]-mid[-1]))

def check_levy():
    import matplotlib.pyplot as plt
    import levy
    t = np.linspace(-0.05, 0.05, 100)
    plt.hist(z, normed=True)
    plt.plot(t, levy.levy(t, 1.52569, 0.725142, -0.00315616, 0.00680555))
    plt.show()

if __name__ == "__main__":
    brute_force()

openbugs_result = """
        mean      sd        MC_error  2.5pc     median    97.5pc   start sample
alpha    1.558    0.1995    0.008487   1.17      1.56     1.923    1001  20000
beta    -0.5499   0.3628    0.01235   -0.9743   -0.65     0.3909   1001  20000
delta    0.008204 0.00158   6.989E-5   0.005828  0.007991 0.0121   1001  20000
gamma    7.936E-4 0.002908  1.506E-4  -0.003841  1.917E-4 0.007961 1001  20000
"""
