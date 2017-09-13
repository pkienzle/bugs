#!/usr/bin/env python
from __future__ import print_function, division
import sys
import time

from numpy import inf
import numpy as np
import emcee

def load_problem(filename, options=None):
    """
    Load a problem definition from a python script file.

    sys.argv is set to ``[file] + options`` within the context of the script.

    The user must define ``problem=FitProblem(...)`` within the script.

    Raises ValueError if the script does not define problem.
    """
    ctx = dict(__file__=filename, __name__="model")
    old_argv = sys.argv
    sys.argv = [filename] + options if options else [filename]
    source = open(filename).read()
    code = compile(source, filename, 'exec')
    exec(code, ctx)
    sys.argv = old_argv

    return ctx

def eps_init(n, initial, bounds, use_point=False, eps=1e-6):
    # type: (int, np.ndarray, np.ndarray, bool, float) -> np.ndarray
    """
    Generate a random population using an epsilon ball around the current
    value.

    Since the initial population is contained in a small volume, this
    method is useful for exploring a local minimum around a point.  Over
    time the ball will expand to fill the minimum, and perhaps tunnel
    through barriers to nearby minima given enough burn-in time.

    eps is in proportion to the bounds on the parameter, or the current
    value of the parameter if the parameter is unbounded.  This gives the
    initialization a bit of scale independence.

    If *use_point* is True, then the current value of the parameters
    is returned as the first point in the population.
    """
    # Set the scale from the bounds, or from the initial value if the value
    # is unbounded.
    xmin, xmax = bounds
    scale = _get_scale_factor(eps, bounds, initial)
    #print("= scale", scale)
    initial = np.clip(initial, xmin, xmax)
    population = initial + scale * (2 * np.random.rand(n, len(xmin)) - 1)
    population = reflect(population, xmin, xmax)
    if use_point:
        population[0] = initial
    return population

def _get_scale_factor(scale, bounds, initial):
    # type: (float, np.ndarray, np.ndarray) -> np.ndarray
    xmin, xmax = bounds
    dx = (xmax - xmin) * scale  # type: np.ndarray
    dx[np.isinf(dx)] = abs(initial[np.isinf(dx)]) * scale
    dx[~np.isfinite(dx)] = scale
    dx[dx == 0] = scale
    #print("min,max,dx",xmin,xmax,dx)
    return dx

def reflect(v, low, high):
    """
    Reflect v off the boundary, then clip to be sure it is within bounds
    """
    index = v < low
    v[index] = (2*low - v)[index]
    index = v > high
    v[index] = (2*high - v)[index]
    return np.clip(v, low, high)

class Monitor(object):
    def __init__(self, num_iterations):
        self.timer, self.iteration = time.time(), 0
        self.index, self.best, self.best_x = 0, -inf, None
        self.num_iterations = num_iterations
    def monitor(self, p, lnlike):
        self.index += 1
        index = np.argmax(lnlike)
        if lnlike.flat[index] > self.best:
            self.best, self.best_x = lnlike.flat[index], p.reshape((-1,p.shape[-1]))[index]
        now = time.time()
        if now > self.timer+1:
            print(self.index, "of", self.num_iterations, self.best)
            self.timer = now
    __call__ = monitor

def run_emcee(nllf, p0, bounds, burn, nsamples, nwalkers, temperatures):
    if np.isscalar(temperatures):
        ntemps, betas = temperatures, None
    else:
        ntemps, betas = len(temperatures), 1/temperatures
    log_prior = lambda p: 0 if ((p>=bounds[0])&(p<=bounds[1])).all() else -inf
    log_likelihood = lambda p: -nllf(p)
    sampler = emcee.PTSampler(
        ntemps=ntemps, nwalkers=nwalkers, dim=len(p0),
        logl=log_likelihood, logp=log_prior,
        betas=betas,
        )

    nthin = 1
    steps = nsamples//(nwalkers*nthin)

    monitor = Monitor(burn+steps)

    # Burn-in
    pop = eps_init(nwalkers*ntemps, p0, bounds).reshape(ntemps, nwalkers, -1)
    for p, lnprob, lnlike in sampler.sample(pop, iterations=burn):
        monitor(p, lnlike)
    sampler.reset()
    print("== after burn ==", monitor.index, monitor.best)

    # Collect
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                            lnlike0=lnlike,
                                            iterations=nthin*steps, thin=nthin):
        monitor(p, lnlike)
    print("== after sample ==", monitor.index, monitor.best)

    assert sampler.chain.shape == (ntemps, nwalkers, steps, len(p0))
    #import pprint; pprint.pprint(sampler.__dict__)
    #[print(k, v.shape) for k, v in sampler.__dict__.items() if isinstance(v, np.ndarray)]
    return sampler

def build_state(sampler, problem):
    raise NotImplementedError("not done yet...")
    from bumps.dream.state import MCMCDraw
    T1 = MCMCDraw(0, 0, 0, 0, 0, 0, 1)
    state.draws = steps * nwalkers
    state.generation = steps
    state._gen_index = 0

    state.labels = problem.labels

class Draw(object):
    def __init__(self, logp, points, weights, labels, vars=None, integers=None):
        self.logp = logp
        self.weights = weights
        self.points = points[:,vars] if vars else points
        self.labels = [labels[v] for v in vars] if vars else labels
        if integers is not None:
            self.integers = integers[vars] if vars else integers
        else:
            self.integers = None

def process_vars(T, draw):
    import matplotlib.pyplot as plt
    from bumps.dream import stats, views
    title = "T=%g"%T
    vstats = stats.var_stats(draw)
    print("=== %s ==="%title)
    print(stats.format_vars(vstats))
    plt.figure()
    views.plot_vars(draw, vstats)
    plt.suptitle(title)
    plt.figure()
    views.plot_corrmatrix(draw)
    plt.suptitle(title)


def main():
    ## Hard-coded options
    #temperatures = 5
    temperatures = np.array([1, 3], 'd')

    ## command line options
    filename = sys.argv[1]
    burn, nsamples = int(sys.argv[2]), int(sys.argv[3])
    options = sys.argv[4:]

    # Load the problem
    model = load_problem(filename, options=options)
    problem = model['problem']
    nllf = problem.nllf
    ## reload from /tmp/t1; handy for continuing from a bumps fit
    #import bumps.cli; bumps.cli.load_best(problem, "/tmp/t1")
    p0 = problem.getp()
    bounds = problem._bounds
    labels = problem.labels()
    visible_vars = getattr(problem, 'visible_vars', None)
    integer_vars = getattr(problem, 'integer_vars', None)
    derived_vars, derived_labels = getattr(problem, 'derive_vars', (None, None))

    # Perform MCMC
    dim = len(p0)
    nwalkers = 2*dim
    sampler = run_emcee(nllf, p0, bounds, burn, nsamples, nwalkers, temperatures=temperatures)
    ntemps = len(sampler.betas)
    samples = np.reshape(sampler.chain, (ntemps, -1, dim))
    logp = np.reshape(sampler.lnlikelihood, (ntemps, -1))

    # process derived parameters
    if derived_vars:
        samples = np.reshape(samples, (-1, dim))
        new_vars = np.asarray(derived_vars(samples.T)).T
        samples= np.hstack((samples, new_vars))
        labels += derived_labels
        dim += len(derived_labels)
        samples = np.reshape(samples, (ntemps, -1, dim))

    # identify visible and integer variables
    visible = [labels.index(p) for p in visible_vars] if visible_vars else None
    integers = np.array([var in integer_vars for var in labels]) if integer_vars else None

    # plot the results, but only for the lowest and highest temperature
    draw = Draw(logp[0], samples[0], None, labels, vars=visible, integers=integers)
    process_vars(1/sampler.betas[0], draw)
    if len(sampler.betas) > 1:
        draw = Draw(logp[-1], samples[-1], None, labels, vars=visible, integers=integers)
        process_vars(1/sampler.betas[-1], draw)

    # show the plots
    import matplotlib.pyplot as plt; plt.show()

if __name__ == "__main__":
    main()
