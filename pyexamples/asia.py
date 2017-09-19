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
from __future__ import print_function, division

from bumps.names import *
from bugs.parse import load, define_pars
from bugs.model import dcat_llf, round

# data: asia=2, dyspnoea=2,
# p.smoking[2] p.tuberculosis[2,2] p.bronchitis[2,2],
# p.lung.cancer[2,2], p.xray[2,2], p.dyspnoea[2,2,2]
_, data = load('../examples/Asiadata.txt')
# init: smoking, tuberculosis, lung.cancer, bronchitis, xray
pars = "smoking,tuberculosis,lung.cancer,bronchitis,xray".split(',')
_, init = load('../examples/Asiainits.txt')
p0, labels = define_pars(init, pars)

def nllf(pars):
    smoking, tuberculosis, lung_cancer, bronchitis, xray \
        = np.asarray(np.floor(pars), 'i')
    cost = 0
    cost += dcat_llf(smoking, data['p.smoking'])
    cost += dcat_llf(tuberculosis, data['p.tuberculosis'][data['asia']-1])
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


problem = DirectProblem(nllf, p0, labels=labels, dof=1)
problem.setp(p0)
for i, p in enumerate(problem.p):
    problem._bounds[:, i] = [1.00000000, 2.999999999]
problem.derive_vars = post, post_vars
problem.visible_vars = [
    "bronchitis", "either", "lung.cancer", "smoking", "tuberculosis", "xray",
    ]
problem.integer_vars = [
    "bronchitis", "either", "lung.cancer", "smoking", "tuberculosis", "xray",
    ]

def brute_force():
    llf = []
    for smoking in (1, 2):
        for tuberculosis in (1, 2):
            for lung_cancer in (1, 2):
                for bronchitis in (1, 2):
                    for xray in (1, 2):
                        p = np.asarray([smoking, tuberculosis, lung_cancer, bronchitis, xray])
                        llf.append(-nllf(p))
    p = np.exp(np.asarray(llf))
    p = (p/np.sum(p)).reshape((2, 2, 2, 2, 2))
    p_bronchitis = np.sum(p[:, :, :, 1, :])
    p_lung_cancer = np.sum(p[:, :, 1, :, :])
    p_smoking = np.sum(p[1, :, :, :, :])
    p_tuberculosis = np.sum(p[:, 1, :, :, :])
    p_xray = np.sum(p[:, :, :, :, 1])
    p_either = p_tuberculosis + p_lung_cancer - np.sum(p[:, 1, 1, :, :])
    rates = np.array([p_bronchitis, p_either, p_lung_cancer, p_smoking, p_tuberculosis, p_xray])
    means = 1 + rates  # rates*2 + (1-rates)*1 = 1 + rates
    print("""
     bronchitis: %g
         either: %g
    lung_cancer: %g
        smoking: %g
   tuberculosis: %g
           xray: %g
    """%tuple(means.tolist()))

openbugs_result = """
               mean    sd     median  2.5pc 97.5pc
bronchitis     1.815   0.3881    2     1     2
either         1.181   0.3847    1     1     2
lung.cancer    1.1     0.2995    1     1     2
smoking        1.626   0.4839    2     1     2
tuberculosis   1.086   0.2798    1     1     2
xray           1.218   0.4129    1     1     2
"""

if __name__ == "__main__":
    brute_force()
