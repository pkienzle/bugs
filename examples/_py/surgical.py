"""
Surgical: Institutional ranking

::

    model
    {
        for( i in 1 : N ) {
            p[i] ~ dbeta(1.0, 1.0)
            r[i] ~ dbin(p[i], n[i])
        }
    }
"""

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dbin_llf, dbeta_llf

#  data: N=12, n[N], r[N]
vars = "n,r,N".split(',')
_, data = load('../Surgicaldata.txt')
n, r, N = (data[p] for p in vars)
# init: p[N]
pars = "p".split(',')
_, init = load('../Surgicalinits.txt')
p0, labels = define_pars(init, pars)

def nllf(p):

    cost = 0
    cost += np.sum(dbeta_llf(p, 1.0, 1.0))
    cost += np.sum(dbin_llf(r, p, n))

    return -cost

dof = 100
problem = DirectProblem(nllf, p0, labels=labels, dof=dof)
problem._bounds[0, :] = 0.0
problem._bounds[1, :] = 1.0
problem.setp(p0)


openbugs_result = """
       mean      sd       MC_error   2.5pc    median    97.5pc   start  sample
p[1]   0.0206    0.01992  2.255E-4   5.941E-4 0.01435   0.07358  1001   10000
p[2]   0.1267    0.02724  2.716E-4   0.07787  0.1248    0.1832   1001   10000
p[3]   0.07434   0.02363  2.559E-4   0.03487  0.07193   0.1275   1001   10000
p[4]   0.05785   0.008047 8.135E-5   0.04316  0.05755   0.07471  1001   10000
p[5]   0.0422    0.01383  1.234E-4   0.01952  0.04073   0.07318  1001   10000
p[6]   0.07033   0.01812  1.909E-4   0.03931  0.06892   0.1104   1001   10000
p[7]   0.06632   0.02015  1.899E-4   0.03252  0.06421   0.1104   1001   10000
p[8]   0.1477    0.02434  2.437E-4   0.1034   0.1464    0.1993   1001   10000
p[9]   0.07175   0.01779  2.016E-4   0.04124  0.07009   0.1104   1001   10000
p[10]  0.09067   0.02841  2.962E-4   0.04379  0.08805   0.1531   1001   10000
p[11]  0.1162    0.01991  1.78E-4    0.0798   0.1154    0.1577   1001   10000
p[12]  0.06895   0.01328  1.331E-4   0.04508  0.06824   0.0974   1001   10000
"""
