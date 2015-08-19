##############################################################################
# Distribution log likelihood functions
##############################################################################
from __future__ import division

import scipy.special
from scipy.special import betaln, gammaln
import math
import numpy as np
from numpy import exp, log, pi, inf
LOG_1_ROOT_2PI = -0.5*log(2*pi)
ROOT_2 = math.sqrt(2)



def binomln(n, k): return -betaln(1 + n - k, 1 + k) - log(n + 1)
def logfact(k): return gammaln(k+1)
def logdet(A): return math.log(np.linalg.det(A))
def logit(x):  return math.log(x/(1-x))
def cloglog(x): return log(-log(1-x))
gammap = scipy.special.gammainc
def ilogit(x): return math.exp(x)/(1+math.exp(x)) if x < 0 else 1/(1+math.exp(-x))
def iclogit(x): return 1 - math.exp(-math.exp(x))
def probit(x): return ROOT_2 * scipy.special.erfinv(2*x-1)
def phi(x): return 0.5*(1+scipy.special.erf(x/ROOT_2))
def round(x): return math.floor(x+1)
def step(x): return 1 if x >=0 else 0
def rank(v, s): return len(vi for vi in v if v < s)
def sort(v): return list(sorted(v))

inverse = {
    logit: ilogit,
    probit: phi,
    cloglog: iclogit,
    math.log: math.exp,
    math.sin: math.asin,
    math.cos: math.acos,
    math.tan: math.atan,
    }
inverse.update((v,k) for k,v in inverse.items())

def wrap(fn):
    @staticmethod
    def wrapped(x):
        return fn(x.value)
    #wrapped.__name__ = fn.__name__
    #wrapped.__doc__ = fn.__doc__
    return wrapped

def wrap2(fn):
    @staticmethod
    def wrapped(x,y):
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
    equals = wrap2(lambda x,y: 1 if x == y else 0)
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
    sd = wrapv(np.std)
    sort = wrapv(sort)
    sum = wrapv(np.sum)

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
    # logdet(A)
    #     log of determinant of positive-definite A
    # ranked(v, k)
    #     kth smallest component of v

# ==== discrete univariate distributions ====
def dbern_llf(x, p):
    """bernoulli(x;p); x = 0,1"""
    return log( (1-p)*(x==0) + p*(x==1) )

def dbin_llf(r, p, n):
    r"""
    binomial distribution

    r ~ binomial(p, n); r = 0, ..., n

    ..math::

        P(r; p, n) = \frac{n!}{r!(n-r)!} p^r (1-p)^{n-r}

    """
    return binomln(n, r) + r*log(p) + (n-r)*log(1-p)

def dcat_llf(x, p):
    """categorical(x;[p_{x=1},p_{x=2},...,p_{x=k}]); sum(p)=1"""
    return log(p[x-1])

def dnegbin_llf(x, p, r):
    """negative binomial(x; p, r); x=0,1,2,..."""
    return binomln(x+r-1, x) + r*log(p) + x*log(1-p)

def dpois_llf(x, L):
    """poisson(x; lambda); x=0,1,2,..."""
    if (L<0).any(): print "L",L[L<0]
    return -L + x*log(L) - logfact(x)

def dgeom_llf(x, p):
    """geometric(x; p); x=1,2,3,..."""
    return (x-1)*log(1-p) + log(p)

def dgeom0_llf(x, p):
    """geometric(x; p); x = 0,1,2,..."""
    return x*log(1-p) + log(p)

def dhypr_llf(x, n, m, N, psi):
    """non-central hypergeometric(x; n, m, N, psi); x in [max(0,m+n-N),min(n,m)]"""
    u0 = max(0, m-N+n)
    u1 = min(n,m)
    norm = sum(exp(binomln(n,u)+binomln(N-n,m-u)+u*log(psi))
               for u in range(u0,u1+1))
    binomln(n,x) + binomln(N-n,m-x) + x*log(psi) - log(norm)

# ==== continuous univariate distributions ====
def dbeta_llf(x, alpha, beta):
    """beta(x; alpha, beta); x in (0,1)"""
    return gammaln(alpha+beta)-gammaln(alpha)-gammaln(beta) \
           + (alpha-1)*log(x) + (beta-1)*log(1-x)

def dchisqr_llf(x, k):
    """chi-squared(x; k); x in (0, inf)"""
    return 0.5*(-k*log(2) + (k-2)*log(x) - x) - gammaln(0.5*k)

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
    #VECTORIZE: if alpha <= 0 or beta <= 0: return -inf
    return alpha*log(beta) + (alpha-1)*log(x) - beta*x - gammaln(alpha)

def dgpar_llf(x, mu, sigma, eta):
    """generalized pareto(x; mu, sigma, eta); x in [mu, sigma/eta - mu]"""
    return -(1+1/eta) * log(1 + eta/sigma*(x-mu)) - log(sigma)

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
    #VECTORIZE: if tau <= 0: return 0 if tau == 0 and mu == x else -inf
    return LOG_1_ROOT_2PI + 0.5*log(tau) - 0.5*(x-mu)**2*tau

def dpar_llf(x, alpha, c):
    """pareto(x; alpha, c); x in (c, inf)"""
    return log(alpha) + alpha*log(c) - (alpha+1) * log(x)

def dt_llf(x, mu, tau, k):
    """Student-t(x; tau, k); x in (-inf, inf); k=1,2,3,..."""
    return gammaln((k+1)/2) - gammaln(k/2) + 0.5*log(tau/k/pi) \
           - (k+1)/2 * log(1 + tau/k*(x-mu)**2)

def dunif_llf(x, a, b):
    """uniform(x;a,b); x in (a, b)"""
    # x==x to force vector return value for vector inputs
    return -log((x==x)*(b - a))

def dweib_llf(x, v, L):
    """weibull(x; v, lambda); x in (0, inf)"""
    return log(v*L) + (v-1)*log(x) - L*x**v

# ==== discrete multivariate distributions ====
def dmulti_llf(x, p, n):
    """mulinomial([x1,...,xk]; [p1,...,pk], n)"""
    return gammaln(n+1) - np.sum(gammaln(x+1)) + np.sum(x*log(p))


# ==== continuous multivariate distributions ====
def ddirich_llf(x, alpha):
    """dirichlet([x1,...,xk]; [alpha1,...,alphak]); x_i in (0,1), sum(x) = 1"""
    # x,alpha are both of dimension n.
    return -np.sum(gammaln(alpha)) + gammaln(sum(alpha)) + np.sum((alpha-1)*log(x))

def dmnorm_llf(x, mu, T):
    """multivariate normal([x1,...,xk]; mu, T); x_i in (-inf, inf); T = inv(Sigma)"""
    N = len(x)
    return N*LOG_1_ROOT_2PI + logdet(T)/2 - np.dot(np.dot(x-mu, T), x-mu)/2

def dmt_llf(x, mu, T, k):
    """multivariate Student-t(x; mu, T, k); x_i in (-inf, inf); T = inv(Sigma)"""
    N = len(x)
    return gammaln(k/2) + N/2*(log(k)+log(pi)) + logdet(T)/2 \
        - (k+N)/2 * log(1 + np.dot(np.dot(x-mu, T), x-mu)/k)

def dwish_llf(x, V, n):
    """wishart(x; V, n); X,V in p x p positive definite; n > p-1"""
    p = len(x)
    tr = np.sum(np.diag(np.linalg.dot(np.linalg.inv(V),x)))
    return (n-p-1)/2*logdet(x) - tr/2 - n*p/2*log(2) - n/2*logdet(V) \
        - scipy.special.multigammaln(n/2,p)

