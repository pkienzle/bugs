#model
#{
#	for (i in 1 : nChild) {
#		theta[i] ~ dnorm(0.0, 0.001)
#		for (j in 1 : nInd) {
## Cumulative probability of > grade k given theta
#			for (k in 1: ncat[j] - 1) {
#				logit(Q[i, j, k]) <- delta[j] * (theta[i] - gamma[j, k])
#			}
#		}
#
## Probability of observing grade k given theta
#		for (j in 1 : nInd) {
#			p[i, j, 1] <- 1 - Q[i, j, 1]
#			for (k in 2 : ncat[j] - 1) {
#				p[i, j, k] <- Q[i, j, k - 1] - Q[i, j, k]
#			}
#			p[i, j, ncat[j]] <- Q[i, j, ncat[j] - 1]
#			grade[i, j] ~ dcat(p[i, j, 1 : ncat[j]])
#		}
#	}
#}

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dcat_llf, dnorm_llf, ilogit

#  data: rt[Num], nt[Num], rc[Num], nc[Num], Num
vars = "nChild,nInd,gamma,delta,ncat,grade".split(',')
_, data = load('../Bonesdata.txt')
nChild, nInd, gamma, delta, ncat, grade = (data[p] for p in vars)
ncat = np.asarray(ncat, 'i')
# init: d, delta.new, tau, mu[Num], delta[Num]
pars = "theta,grade".split(',')
_, init = load('../Bonesinits1.txt')
p0, labels = define_pars(init, pars)

# Grade is being filled in with fitted values by setting NA in the data
# for grade where it is missing, and setting NA in the initial value for
# grade where it is present.  That is, the same array is used for both
# present and missing data.  Rather than implementing that here, we will
# simply keep an index of where grade is missing and reconstruct grade
# from the two sources in the nllf function.  See aligator.py for a
# generic implementation of missing fit parameters.
active_index = ~np.isnan(p0)
labels = [label for label, active in zip(labels, active_index) if active]
p0 = p0[active_index]
missing_grade_index = np.isnan(grade)

def bones(pars):
    theta = pars[:13]
    grade[missing_grade_index] = np.floor(pars[13:] + 0.5)

    p = np.empty((nChild, nInd, 5))  # max(ncat) is 5
    Q = np.empty((nChild, nInd, 4))
    Q = ilogit(delta[None, :, None] * (theta[:, None, None] - gamma[None, :, :]))
    p[:, :, 1:-1] = Q[:, :, :-1] - Q[:, :, 1:]
    p[:, :, 0] = 1 - Q[:, :, 0]
    p[:, :, ncat-1] = Q[:, :, ncat-2]

    cost = 0
    cost += np.sum(dnorm_llf(theta, 0.0, 0.001))
    for i in range(nChild):
        for j in range(nInd):
            if 1 <= grade[i, j] <= ncat[j]:
                cost += dcat_llf(grade[i, j], p[i, j, :ncat[j]])
            else:
                cost = -inf

    return -cost


dof = 1
problem = DirectProblem(bones, p0, labels=labels, dof=dof)
problem.setp(p0)
problem.visible_vars = ["theta[%d]"%i for i in range(1, nChild+1)]

#	mean	sd	MC_error	val2.5pc	median	val97.5pc	start	sample
#theta[1]	0.3257	0.2063	0.004121	-0.1156	0.3335	0.7077	1001	10000
#theta[2]	1.367	0.2511	0.004992	0.9137	1.36	1.88	1001	10000
#theta[3]	2.349	0.276	0.00564	1.83	2.338	2.9	1001	10000
#theta[4]	2.898	0.2985	0.005811	2.299	2.89	3.482	1001	10000
#theta[5]	5.539	0.4973	0.009796	4.61	5.527	6.551	1001	10000
#theta[6]	6.756	0.609	0.01244	5.619	6.736	7.988	1001	10000
#theta[7]	6.456	0.5903	0.0115	5.267	6.465	7.633	1001	10000
#theta[8]	8.921	0.684	0.01458	7.559	8.929	10.24	1001	10000
#theta[9]	8.982	0.6714	0.01381	7.618	9.003	10.21	1001	10000
#theta[10]	11.94	0.7015	0.01503	10.6	11.92	13.36	1001	10000
#theta[11]	11.57	0.9041	0.019	9.895	11.51	13.43	1001	10000
#theta[12]	15.81	0.5761	0.01128	14.72	15.82	16.98	1001	10000
#theta[13]	16.93	0.7341	0.01439	15.59	16.9	18.43	1001	10000
