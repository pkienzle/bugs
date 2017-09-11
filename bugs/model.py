r"""
OpenBUGS library
================

Reimplementation of the functions and distributions provided in OpenBUGS
so that bugs models can be translated into python negative log likelihood
functions.

See the OpenBugs manual for details:

    http://openbugs.net/Manuals/ModelSpecification.html

Censored distributions
----------------------

Note on censored data, such as *xd ~ dweib(v, L)C(xl, xr)*

Censoring is only in effect if xd is nan.  For weibull, this corresponds
to no failure being observed. Otherwise, xd is the observed failure time
which is follows the Weibull log likelihood function.  Implementing
generalized censoring will require significant effort, requiring the
cdf of each distribution that is censored.

Given distribution with::

    probability density function pdf $f(x)$
    cumulative density function cdf $F(x) = \int_{-\infty}^x f(v)\, dv$
    survival function $S(x) = 1-F(x)$
    hazard function $\lambda(x) = f(x)/S(x)$, so $f(x) = \lambda(x)S(x)$

then censored log likelihood data from an experiment with::

    observed failures $d$ having a measured lifetime $x_d$
    surviving samples $r$ not observed to fail within measured time $x_r$

has the form:

.. math:

    \log(L) = \sum_d \log(f(x_d)) + \sum_r \log(S(x_r))

Generalizing to measurements where failures happen before observation or
between observerations::

    unobserved failures $l$ having lifetime less than $x_l$
    unobserved failures $i$ having lifetime in $[x_l,x_r]$

this becomes:

.. math:

    \log(L) = \sum_d \log(f(x_d)) + \sum_r \log(S(x_r))
        + \sum_l \log(1 -S(x_r)) + \sum_i \log(S(x_l) - S(x_r))

The weibull distribution has a convenient definition with closed
form expressions for hazard and survival:

.. math::

    \lambda(x) = v \lambda x^{v-1}
    S(x) = \exp(-\lambda x^v)
    f(x) = \lambda(x) S(x)

Trucated distributions
----------------------

Note on trucated data, such as x ~ dnorm(mu, tau)T(0,)

Truncated distributions can be simply formed by correcting llf with
-log(F(xl)) if left truncated, -log(S(xr)) if right truncated or
-log(F(xl) + S(xr)) if truncated to an interval.  If the truncation
region is fixed then this will not affect the MCMC.  If however the
trunctation region is a fitted parameter, then the correct truncation
normalization factor will need to be applied to the distribution.

References
----------

OpenBugs (2014). Computer Software. http://www.openbugs.net/

Gelfand, A. E. and Smith, A. F. M. (1990)
*Sampling-based approaches to calculating marginal densities*,
Journal of the American Statistical Association 85: 398--409

Lunn, D., Spiegelhalter, D., Thomas, A. and Best, N. (2009)
The BUGS project: Evolution, critique and future directions (with discussion),
Statistics in Medicine 28: 3049--3082.

Zhang, D. (2005) Likelihood and Censored (or Truncated) Survival Data
Lecture notes for Analysis of Survival Data (ST745), Spring 2005, 3
http://www4.stat.ncsu.edu/%7Edzhang2/st745/chap3.pdf

"""
from __future__ import division, print_function

import scipy.special
from scipy.special import betaln, gammaln, xlogy, xlog1py
import math
import numpy as np
from numpy import exp, log, pi, inf, nan, mean
LOG_1_ROOT_2PI = -0.5*log(2*pi)
ROOT_2 = math.sqrt(2)



def binomln(n, k):
    return -betaln(1 + n - k, 1 + k) - log(n + 1)
def logfact(k):
    return gammaln(k+1)
def logdet(A):
    sign, logdet = np.linalg.slogdet(A)
    return logdet if sign >= 0 else nan
def logit(x):
    return log(x/(1-x))
def cloglog(x):
    return log(-log(1-x))
gammap = scipy.special.gammainc
def ilogit(x):
    if np.isscalar(x):
        ret = math.exp(x)/(1+math.exp(x)) if x < 0 else 1/(1+math.exp(-x))
    else:
        ret = np.empty_like(x)
        idx = (x < 0)
        neg = x[idx]
        ret[idx] = exp(neg)/(1.0+exp(neg))
        ret[~idx] = 1.0/(1.0+exp(-x[~idx]))
    return ret
def iclogit(x):
    return  -np.expm1(-np.exp(x))
def probit(x):
    return ROOT_2 * scipy.special.erfinv(2*x-1)
def phi(x):
    return 0.5*(1+scipy.special.erf(x/ROOT_2))
def round(x):
    return np.round(x)
#def step(x): return 1.0 if x >=0 else 0
def step(x):
    return 1.0*(x>=0)
def rank(v, s):
    return len(vi for vi in v if v < s)
def sd(x):
    return np.std(x, ddof=1)
def sort(v):
    return list(sorted(v))

inverse = {
    logit: ilogit,
    probit: phi,
    cloglog: iclogit,
    math.log: math.exp,
    math.sin: math.asin,
    math.cos: math.acos,
    math.tan: math.atan,
    }
inverse.update(dict((v, k) for k, v in inverse.items()))

def wrap(fn):
    @staticmethod
    def wrapped(x):
        return fn(x.value)
    #wrapped.__name__ = fn.__name__
    #wrapped.__doc__ = fn.__doc__
    return wrapped

def wrap2(fn):
    @staticmethod
    def wrapped(x, y):
        return fn(x.value, y.value)
    #wrapped.__name__ = fn.__name__
    #wrapped.__doc__ = fn.__doc__
    return wrapped

wrapv = wrap
wrap2v = wrap2

class BugsContext:
    # scalar
    abs = wrap(abs)
    arccos = wrap(math.acos)
    arccosh = wrap(math.acosh)
    arcsin = wrap(math.asin)
    arcsinh = wrap(math.asinh)
    arctan = wrap(math.atan)
    arctanh = wrap(math.atanh)
    cloglog = wrap(cloglog)
    cos = wrap(math.cos)
    cosh = wrap(math.cosh)
    exp = wrap(math.exp)
    equals = wrap2(lambda x, y: 1 if x == y else 0)
    gammap = wrap2(scipy.special.gammainc)
    ilogit = wrap(ilogit)
    iclogit = wrap(iclogit)
    log = wrap(math.log)
    logfact = wrap(logfact)
    loggam = wrap(gammaln)
    logit = wrap(logit)
    max = wrap2(max)
    min = wrap2(min)
    phi = wrap(phi)
    probit = wrap(probit)
    pow = wrap2(pow)
    round = wrap(round)
    sin = wrap(math.sin)
    sinh = wrap(math.sinh)
    sqrt = wrap(math.sqrt)
    step = wrap(step)
    tan = wrap(math.tan)
    tanh = wrap(math.tanh)
    trunc = wrap(math.trunc)

    dot = wrap2v(np.dot)
    inverse = wrapv(np.linalg.inv)
    mean = wrapv(np.mean)
    eigen_values = wrapv(np.linalg.eigvals)
    prod = wrapv(np.prod)
    rank = wrap2v(rank)
    sd = wrapv(sd)
    sort = wrapv(sort)
    sum = wrapv(np.sum)
    logdet = wrapv(logdet)

    # dist functions require s1, s2 be distribution objects with cdf, llf, and value
    @staticmethod
    def cumulative(s1, s2):
        return s1.cdf(s2.value)
    @staticmethod
    def density(s1, s2):
        return math.exp(s1.llf(s2.value))
    @staticmethod
    def deviance(s1, s2):
        return -2*s1.llf(s2.value)
    @staticmethod
    def interp_lin(x, xp, fxp):
        return np.interp(x.value, xp.value, fxp.value)

    # TODO: missing
    # cut(s1)
    #     don't consider s1 when estimating likelihood
    # post.p.value(s)
    #     returns one if a sample from the prior is less than the value of s
    # prior.p.value(s)
    #     returns one if a sample from the prior after resampling its
    #     stochastic parents is less than value of s
    # replicate.post(s)
    #     replicate from distribution of s
    # replicate.prior(s)
    #     replicate from distribution of s after replicating from its
    #     stochastic parents
    # p_valueM(s)
    #     return a vector of ones and zeros depending on if a sample from the
    #     prior is less than the value of the corresponding component of s
    # replicate.postM(s)
    #     replicate from multivariate distribution of v

    # integral(F, low, high, tolerance)
    #     definite integral of F between low and high with accuracy within tolerance
    # solution(F, low, high, tolerance)
    #     s in [low, high] such that F(s) = 0, where sign(F(low)) != sign(F(high))
    # ode(x_0, grid, D(C,t), t_0, tolerance)
    #     solve D over grid starting with x_0 at time t_0
    # ranked(v, k)
    #     kth smallest component of v

# TODO: support censored distributions other than weibull
# TODO: support truncated distributions
# TODO: test parameter ranges in distributions to avoid numpy warnings
# TODO: remove _llf tags on the distribution calculators

# ==== discrete univariate distributions ====
def dbern_llf(x, p):
    """bernoulli(x;p); x = 0,1"""
    return log((1-p)*(x == 0) + p*(x == 1))

def dbin_llf(r, p, n):
    r"""
    binomial distribution

    r ~ binomial(p, n); r = 0, ..., n

    ..math::

        P(r; p, n) = \frac{n!}{r!(n-r)!} p^r (1-p)^{n-r}

    """
    return binomln(n, r) + xlogy(r, p) + xlog1py(n-r, -p)

def dcat_llf(x, p):
    """categorical(x;[p_{x=1},p_{x=2},...,p_{x=k}]); sum(p)=1"""
    return log(p[x-1])

def dnegbin_llf(x, p, r):
    """negative binomial(x; p, r); x=0,1,2,..."""
    return binomln(x+r-1, x) + xlogy(r, p) + xlog1py(x, -p)

def dpois_llf(x, L):
    """poisson(x; lambda); x=0,1,2,..."""
    x, L = np.broadcast_arrays(x, L)
    index = L >= 0
    if index.all():
        llf = -L + xlogy(x, L) - logfact(x)
    else:
        #print("L", L[L < 0])
        llf = np.empty(x.shape)
        llf[index] = -L[index] + xlogy(x[index], L[index]) - logfact(x[index])
        llf[~index] = -inf
    return llf

def dgeom_llf(x, p):
    """geometric(x; p); x=1,2,3,..."""
    return xlog1py(x-1, -p) + log(p)

def dgeom0_llf(x, p):
    """geometric(x; p); x = 0,1,2,..."""
    return xlogy(x, 1-p) + log(p)

def dhypr_llf(x, n, m, N, psi):
    """non-central hypergeometric(x; n, m, N, psi); x in [max(0,m+n-N),min(n,m)]"""
    u0 = max(0, m-N+n)
    u1 = min(n, m)
    norm = sum(exp(binomln(n, u)+binomln(N-n, m-u)+xlogy(u, psi))
               for u in range(u0, u1+1))
    return binomln(n, x) + binomln(N-n, m-x) + xlogy(x, psi) - log(norm)

# ==== continuous univariate distributions ====
def dbeta_llf(x, alpha, beta):
    """beta(x; alpha, beta); x in (0,1)"""
    return gammaln(alpha+beta)-gammaln(alpha)-gammaln(beta) \
           + xlogy(alpha-1, x) + xlog1py(beta-1, -x)

def dchisqr_llf(x, k):
    """chi-squared(x; k); x in (0, inf)"""
    return 0.5*(-k*log(2) + xlogy(k-2, x) - x) - gammaln(0.5*k)

def ddexp_llf(x, mu, tau):
    """double exponential(x; mu, tau); x in (-inf, inf)"""
    return log(0.5*tau) - tau*abs(x-mu)

def dexp_llf(x, L):
    """exponential(x; lambda); x in (0, inf)"""
    return log(L) - L*x

def dflat_llf(x):
    """uniform(x; -inf, inf); x in (-inf, inf)"""
    return 0

def dgamma_llf(x, alpha, beta):
    """gamma(x; alpha=k, beta=1/theta); x in (0, inf)"""
    return alpha * log(beta) + xlogy(alpha-1, x) - beta*x - gammaln(alpha)
    ## range protections if we want them (3x slower)
    #try:
    #    with np.errstate(all='raise'):
    #        return alpha * log(beta) + xlogy(alpha-1, x) - beta*x - gammaln(alpha)
    #except Exception:
    #    x, alpha, beta = np.broadcast_arrays(x, alpha, beta)
    #    index = (x <= 0) | (alpha <= 0) | (beta <= 0)
    #    llf = np.empty(x.shape)
    #    llf[index] = -inf
    #    index = ~index
    #    alpha, beta, x = alpha[index], beta[index], x[index]
    #    llf[index] = alpha * log(beta) + xlogy(alpha-1, x) - beta*x - gammaln(alpha)
    #    return llf

def dgpar_llf(x, mu, sigma, eta):
    """generalized pareto(x; mu, sigma, eta); x in [mu, sigma/eta - mu]"""
    return xlogy(-(1+1/eta), 1 + eta/sigma*(x-mu)) - log(sigma)

def dloglik_llf(x, p):
    """
    P(x;exp(p))

    Define a dummy observed variable set to 0, and an expression P(x) with
    returns the log probability of x given the context of the rest of the
    defined variables.  This is equivalent to setting x as an observed
    variable with distribution exp(P(x)).

    For example,

    ::

        model {
            dummy.x <- 0             # observed dummy set to 0
            dummy.x ~ dloglik(P.x)   # dummy is distributed like exp(P.x)
            # Set P.x as log(normal(mu,sigma)); note that the missing
            # constant -log(2 pi) doesn't affect the posterior and so it
            # can be dropped from the expression.
            P.x <- -log(sigma) - 0.5*pow((x - mu)/sigma, 2)
            mu ~ dunif(-10, 10)
            sigma ~ dunif(0, 10)
        }

    is equivalent to::

        model {
            x ~ dnorm(mu, tau)
            tau <- 1 / (sigma * sigma)
            mu ~ dunif(-10, 10)
            sigma ~ dunif(0, 10)
        }
    """
    return p

def dlnorm_llf(x, mu, tau):
    """log normal(x; mu, tau=1/sigma**2); x in (0, inf)"""
    return LOG_1_ROOT_2PI - log(x) + 0.5*log(tau) - 0.5*(log(x)-mu)**2*tau

def dlogis_llf(x, mu, tau):
    """logistic(x; mu, tau); x in (-inf, inf)"""
    return log(tau) + tau*(x-mu) - 2*log(1 + exp(tau*(x-mu)))

def dnorm_llf(x, mu, tau):
    """normal(x; mu, tau=1/sigma**2)"""
    return LOG_1_ROOT_2PI + 0.5*log(tau) - 0.5*(x-mu)**2*tau
    ## range protections, if we want them (3x slower)
    #try:
    #    with np.errstate(all='raise'):
    #        return LOG_1_ROOT_2PI + 0.5*log(tau) - 0.5*(x-mu)**2*tau
    #except Exception:
    #    x, mu, tau = np.broadcast_arrays(x, mu, tau)
    #    index = tau <= 0
    #    llf = np.empty(x.shape)
    #    llf[index] = -inf
    #    index = ~index
    #    x, mu, tau = x[index], mu[index], tau[index]
    #    llf[index] = LOG_1_ROOT_2PI + 0.5*log(tau) - 0.5*(x-mu)**2*tau
    #    return llf

def dpar_llf(x, alpha, c):
    """pareto(x; alpha, c); x in (c, inf)"""
    return log(alpha) + xlogy(alpha, c) - xlogy(alpha+1, x)

def dstable_llf(x, alpha, beta, gamma, delta):
    """stable(x; alpha, beta, gamma, delta)"""
    try:
        import levy
    except ImportError:
        raise ImportError("Requires pylevy package https://github.com/pkienzle/pylevy")
    return log(levy.levy(x, alpha, beta, mu=gamma, sigma=delta, par=1))

def dt_llf(x, mu, tau, k):
    """Student-t(x; tau, k); x in (-inf, inf); k=1,2,3,..."""
    return gammaln((k+1)/2) - gammaln(k/2) + 0.5*log(tau/k/pi) \
           - (k+1)/2 * log(1 + tau/k*(x-mu)**2)

def dunif_llf(x, a, b):
    """uniform(x;a,b); x in (a, b)"""
    # x==x to force vector return value for vector inputs
    return -log((x == x)*(b - a))

def dweib_llf(x, v, L):
    """weibull(x; v, lambda); x in (0, inf)"""
    return log(v*L) + xlogy(v-1, x) - L*x**v

def dweib_C_llf(x, v, L, lower=None, upper=None):
    """censored weibull(x; v, lambda); x in (0, inf)"""
    # normalize the data
    x, v, L = np.broadcast_arrays(x, v, L)

    # preallocate space for the result
    llf = np.empty(x.shape)

    # uncensored data
    index = np.isfinite(x)
    xp, vp, Lp = x[index], v[index], L[index]
    llf[index] = log(vp*Lp) + xlogy(vp-1, xp) - Lp*xp**vp

    # censored data
    index = ~index
    if upper is None: # right censored has an accurate form
        lower = np.broadcast_to(lower, x.shape)
        xp, vp, Lp = lower[index], v[index], L[index]
        llf[index] = -Lp*xp**vp
    else: # left censored is equivalent to interval censored with lower=0
        if lower is None:
            lower = 0
        lower = np.broadcast_to(lower, x.shape)
        upper = np.broadcast_to(upper, x.shape)
        # minimize underflow risk by normalizing to start time zero:
        #     log [ e^(-Lx_l^v) - e^(-Lx_r^v) ]
        #     = log [ e(-Lx_l^v) (1 - e^(-Lx_r^v)/e^(-Lx_l^v))]
        #     = -Lx_l^v + log[1 - e^-L(x_r^v-x_l^v)]
        xpl, xpu, vp, Lp = lower[index], upper[index], v[index], L[index]
        llf[index] = log1p(-exp(-Lp*(xpu**vp - xpl**vp))) - L*xpl**vp
    return llf


# ==== discrete multivariate distributions ====
def dmulti_llf(x, p, n):
    r"""
    mulinomial([x1,...,xk]; [p1,...,pk], n)

    Note: $x$ values for $p \leq 0$ are ignored.
    """
    x, p = x[p > 0], p[p > 0]
    return gammaln(n+1) - np.sum(gammaln(x+1)) + np.sum(xlogy(x, p))

# ==== continuous multivariate distributions ====
def ddirich_llf(x, alpha):
    """dirichlet([x1,...,xk]; [alpha1,...,alphak]); x_i in (0,1), sum(x) = 1"""
    # x,alpha are both of dimension n.
    return -np.sum(gammaln(alpha)) + gammaln(sum(alpha)) + np.sum(xlogy(alpha-1, x))

def dmnorm_llf(x, mu, T):
    """multivariate normal([x1,...,xk]; mu, T); x_i in (-inf, inf); T = inv(Sigma)"""
    # pdf = det(2 pi Sigma)^(-1/2) exp(-1/2 (x-mu)'Sigma^-1(x-mu))
    # T = inv(Sigma), det(cA) = c^N det(A), det(A^-1) = det(A)^-1
    # => pdf = (2 pi)^(-N/2) det(T)^(1/2) exp(-1/2 (x-mu)'T(x-mu))
    N = len(x)
    sign_det_T, logdet_T = np.linalg.slogdet(T)
    llf = N*LOG_1_ROOT_2PI + logdet_T/2 - np.dot(np.dot(x-mu, T), x-mu)/2
    #cov = np.linalg.inv(T)
    #scipy_llf = scipy.stats.multivariate_normal.logpdf(x, mean=mu, cov=cov)
    #print("mvn", llf, scipy_llf, scipy_llf-llf, T.flatten(), mu, x, logdet(T)/2)
    return llf if sign_det_T >= 0 else -inf

def dmt_llf(x, mu, T, k):
    """multivariate Student-t(x; mu, T, k); x_i in (-inf, inf); T = inv(Sigma)"""
    N = len(x)
    sign_det_T, logdet_T = np.linalg.slogdet(T)
    llf = (gammaln(k/2) + N/2*(log(k)+log(pi)) + logdet_T/2
           - (k+N)/2 * log(1 + np.dot(np.dot(x-mu, T), x-mu)/k))
    return llf if sign_det_T >= 0 else -inf

def dwish_llf(x, V, n):
    """wishart(x; V, n); X,V in p x p positive definite; n > p-1"""
    try:
        return scipy.stats.wishart.logpdf(x, n, V)
    except Exception as exc:
        #print(x, V, n, exc)
        return -inf

    ## X, V positive definite => X = X^T => tr(invV X) = sum_ij invV_ij X_ij
    ## should use cholesky decomp instead of inv
    #tr = np.sum(np.linalg.inv(V) * x)  # element-wise multiply
    #p = len(x)
    #sign_det_V, logdet_V = np.linalg.slogdet(V)
    #sign_det_x, logdet_x = np.linalg.slogdet(x)
    #llf = ((n-p-1)*logdet_x - tr - n*p*log(2) - n*logdet_V)/2 \
    #      - scipy.special.multigammaln(n/2, p)
    #return llf if sign_det_V >= 0 and sign_det_x >= 0 else -inf

