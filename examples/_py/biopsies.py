# model
# {
#	for (i in 1 : ns){
#		nbiops[i] <- sum(biopsies[i, ])
#		true[i]  ~ dcat(p[])
#		biopsies[i, 1 : 4]  ~ dmulti(error[true[i], ], nbiops[i])
#	}
#	error[2,1 : 2] ~ ddirich(prior[1 : 2])
#	error[3,1 : 3] ~ ddirich(prior[1 : 3])
#	error[4,1 : 4] ~ ddirich(prior[1 : 4])
#	p[1 : 4] ~ ddirich(prior[]);     # prior for p
#}

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dcat_llf, ddirich_llf, dmulti_llf

# data: biopsies, error, ns, prior
_, data = load('../Biopsiesdata.txt')
_, init = load('../Biopsiesinits.txt')
#_,init = load('../Biopsiesinits1.txt')

ns = data['ns']
error = data['error']
prior = data['prior']
biopsies = data['biopsies']
nbiops = np.sum(biopsies, axis=1)

#error_init = init['error']
#error[~np.isnan(error_init)] = error_init[~np.isnan(error_init)]

# The variable true is not initialized, but it seems to be a fitting variable
# The model description says:
#
#   The sampling procedure may not detect the area of maximum rejection,
#   which is considered the true underlying state at the time of the session.
#
#   It is then assumed that each of the observed biopsies are conditionally
#   independent given this truestate with the restriction that there are
#   no 'false positives': i.e. one cannot observe a biopsy worse than the
#   true state.
#
#   No initial values are provided for the latent states, since the forward
#   sampling procedure will find a configuration of starting values that is
#   compatible with the expressed constraints.
#
# I am taking this to mean that true is is obvious from the data and doesn't
# need to be fitted (e.g., by counting the number of zeros at the end of
# each row); if we allow it to be fitted then it could possibly break the
# no false positive restriction.
#
# I have no idea how the BUGS interpreter can possibly figure this out.

def find_true(s):
    if s[3] != 0:
        return 4
    elif s[2] != 0:
        return 3
    elif s[1] != 0:
        return 2
    else:
        return 1
true = np.array([find_true(s) for s in biopsies])


# dirichlet inputs must sum to 1; this means that we must use
#    x[n] = 1 - sum(x[:n-1])
#    if x[n] < 0, nllf = inf
# so we can't use the normal parameter initialization:
#p0, labels = define_pars(init, ["p", "error"])
#active_index = ~np.isnan(p0)
#labels = [label for active,label in zip(active_index,labels) if active]
p = init["p"]
ei = init["error"]
p0 = np.hstack((p[0:3], ei[1, 0:1], ei[2, 0:2], ei[3, 0:3]))
labels = "p[1] p[2] p[3] error[2,1] error[3,1] error[3,2] error[4,1] error[4,2] error[4,3]".split()

# Reconstruct the missing dirichlet component for display
def post(pars):
    p4 = np.sum(pars[0:3], axis=0)
    e22 = 1.0 - pars[3]
    e33 = 1.0 - (pars[4] + pars[5])
    e44 = 1.0 - np.sum(pars[6:9], axis=0)
    return p4, e22, e33, e44
post_labels = list("p[4] error[2,2] error[3,3] error[4,4]".split())

def model(pars):
    p = np.zeros(4, 'd')
    error = np.zeros((4, 4), 'd')
    p[:3] = pars[:3]
    error[1, 0:1] = pars[3:4]
    error[2, 0:2] = pars[4:6]
    error[3, 0:3] = pars[6:9]
    p[3] = 1. - np.sum(p[:3])
    error[0, 0] = 1.
    error[1, 1] = 1. - error[1, 0]
    error[2, 2] = 1. - (error[2, 0] + error[2, 1])
    error[3, 3] = 1. - (error[3, 0] + error[3, 1] + error[3, 2])
    if p[3] < 0. or error[2, 2] < 0. or error[3, 3] < 0.:
        # Note: error[1,1] >= 0 since error[1,0] in [0,1]
        # error[0,0] is always more than 0.
        return np.inf

    cost = 0.
    cost += sum(dmulti_llf(biopsies[i], error[true[i]-1], nbiops[i])
                for i in range(ns))
    cost += ddirich_llf(error[1, :1], prior[:1])
    cost += ddirich_llf(error[2, :2], prior[:2])
    cost += ddirich_llf(error[3, :3], prior[:3])
    cost += ddirich_llf(p, prior)
    return -cost

# observations come from biopsies, with trailing zeros fixed, so only include
# the slots up to the first non-zero element; this is just the value we have
# assigned to 'true', so the number of active elements in biopsies is just the
# sum over all 'true' values.
dof = np.sum(true) - len(p0)
problem = DirectProblem(model, p0, labels=labels, dof=dof)
# p and error are probabilities in [0,1]
problem._bounds[0,:] = 0
problem._bounds[1,:] = 1
problem.setp(p0)
problem.derive_vars = post, post_labels
problem.visible_vars = list("error[2,1] error[2,2] error[3,1] error[3,2] error[3,3] error[4,1] error[4,2] error[4,3] error[4,4] p[1] p[2] p[3] p[4]".split())

#	mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
#error[2,1]	0.589	0.06414	0.001788	0.4618	0.5899	0.7106	1001	10000
#error[2,2]	0.411	0.06414	0.001788	0.2895	0.4101	0.5383	1001	10000
#error[3,1]	0.343	0.04458	7.204E-4	0.2601	0.3413	0.4351	1001	10000
#error[3,2]	0.03664	0.01773	3.001E-4	0.009127	0.03436	0.07722	1001	10000
#error[3,3]	0.6203	0.04678	7.569E-4	0.5246	0.6214	0.7077	1001	10000
#error[4,1]	0.09864	0.04228	6.379E-4	0.03315	0.09341	0.197	1001	10000
#error[4,2]	0.02275	0.02313	3.87E-4	5.184E-4	0.01534	0.08681	1001	10000
#error[4,3]	0.2052	0.05992	0.001004	0.1042	0.2004	0.3357	1001	10000
#error[4,4]	0.6734	0.07257	0.001289	0.5205	0.6774	0.8061	1001	10000
#p[1]	0.1521	0.04878	0.001575	0.04877	0.1534	0.2427	1001	10000
#p[2]	0.3113	0.05377	0.001498	0.2161	0.307	0.4264	1001	10000
#p[3]	0.3888	0.04342	6.389E-4	0.3049	0.388	0.4765	1001	10000
#p[4]	0.1479	0.02961	3.886E-4	0.09444	0.1459	0.2109	1001	10000
