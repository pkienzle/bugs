"""
Asia: expert system

::

    model
    {
        smoking ~ dcat(p.smoking[1:2])
        tuberculosis ~ dcat(p.tuberculosis[asia,1:2])
        lung.cancer ~ dcat(p.lung.cancer[smoking,1:2])
        bronchitis ~ dcat(p.bronchitis[smoking,1:2])
        either <- max(tuberculosis,lung.cancer)
        xray ~ dcat(p.xray[either,1:2])
        dyspnoea ~ dcat(p.dyspnoea[either,bronchitis,1:2])
    }
"""

raise NotImplementedError("Model fails to reproduce the OpenBUGS result")

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dcat_llf, round

# data: asia=2, dyspnoea=2,
# p.smoking[2] p.tuberculosis[2,2] p.bronchitis[2,2],
# p.lung.cancer[2,2], p.xray[2,2], p.dyspnoea[2,2,2]
_, data = load('../Asiadata.txt')
# init: smoking, tuberculosis, lung.cancer, bronchitis, xray
pars = "smoking,tuberculosis,lung.cancer,bronchitis,xray".split(',')
_, init = load('../Asiainits.txt')
p0, labels = define_pars(init, pars)

def asia(pars):
    smoking, tuberculosis, lung_cancer, bronchitis, xray \
        = np.asarray(np.round(pars), 'i')
    cost = 0
    cost += dcat_llf(smoking, data['p.smoking'])
    cost += dcat_llf(lung_cancer, data['p.lung.cancer'][smoking-1])
    cost += dcat_llf(bronchitis, data['p.bronchitis'][smoking-1])
    either = max(tuberculosis, lung_cancer)
    cost += dcat_llf(xray, data['p.xray'][either-1])
    cost += dcat_llf(data['dyspnoea'], data['p.dyspnoea'][either-1, bronchitis-1])
    return -cost

def post(pars):
    smoking, tuberculosis, lung_cancer, bronchitis, xray = pars
    either = np.maximum(tuberculosis, lung_cancer)
    return [either]
post_vars = ["either"]


problem = DirectProblem(asia, p0, labels=labels, dof=1)
problem.setp(p0)
for i, p in enumerate(problem.p):
    problem._bounds[:, i] = [0.500000001, 2.499999999]
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "bronchitis", "either", "lung.cancer", "smoking", "tuberculosis", "xray",
    ]

import warnings
warnings.warn("The Asia model contains integer parameters, which have not yet been shown to work in Bumps")
#raise NotImplementedError("No support for integer variables")
integer_vars = [
    "bronchitis", "either", "lung.cancer", "smoking", "tuberculosis", "xray",
]
# uncomment the following to treat integer variables as integers.  Need to
# verify that the mean value is updated appropriately.  Make sure it is using
# rounding since our bounds are [0.5, 2.5].
problem.integer_vars = integer_vars


openbugs_result = """
               mean    sd     median  2.5pc 97.5pc
bronchitis     1.815   0.3881    2     1     2
either         1.181   0.3847    1     1     2
lung.cancer    1.1     0.2995    1     1     2
smoking        1.626   0.4839    2     1     2
tuberculosis   1.086   0.2798    1     1     2
xray           1.218   0.4129    1     1     2
"""
