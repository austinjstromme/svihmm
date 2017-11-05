"""
This file does some basic testing of our VB implementation.
"""
# external packages
import numpy as np

# internals
from svihmm.distributions.Multinoulli import Multinoulli as mn
from svihmm.distributions.Dirichlet import Dirichlet
from svihmm.distributions.Gaussian import Gaussian as norm
from svihmm.distributions.NDGaussian import NDGaussian as mnorm
from svihmm.distributions.NormalInverseWishart import NormalInverseWishart as niw
from svihmm.distributions.NormalInverseChiSquared import NormalInverseChiSquared as nics
from svihmm.models.HMM import HMM
from svihmm.models.States import States
from svihmm.models.VBHMM import VBHMM


def run():
  eps = 0.2
  res = 0
  num_tests = 5

  print("Running VB tests...")

  #... Gaussian increasingness test...
  temp = int(elbo_increase_Gaussian())
  res += temp
  print("  Gaussian increase test: " + str(temp) + "/1")

  #...multi Gaussian increasingness test...
  temp = int(elbo_increase_multi_Gaussian())
  res += temp
  print("  multi Gaussian increase test: " + str(temp) + "/1")

  #...correct init case...
  temp = int(compare_correct(eps))
  res += temp
  print("  correct init case: " + str(temp) + "/1")

  #...simple case...
  temp = int(compare_simple(3*eps))
  res += temp
  print("  simple case: " + str(temp) + "/1")

  #...increasingness test...
  temp = int(elbo_increase())
  res += temp
  print("  increase test: " + str(temp) + "/1")

  print("VB tests completed: " + str(res) + "/" + str(num_tests))

  return (res == num_tests)

def elbo_increase():
  """
  Run VB on a VBHMM; ensure that elbo is always increasing
  """
  M_true = biased_coins_HMM(0.9, 0.1, 0.05, 0.8)
  num_steps = 200  # controls length of observation sequences
  N = 5  # number of observation sequences
  cnt = 10

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  VBlearner = make_VBHMM(x)

  incr = True
  last = -np.inf

  for j in range(0, cnt):
    VBlearner.VB_step()
    elbo = VBlearner.elbo()
    incr = (elbo > last) and incr
    last = elbo

  return incr

def elbo_increase_Gaussian():
  """
  Run VB on a VBHMM; ensure that elbo is always increasing
  """
  M_true = make_Gaussian_HMM(0.9, 0.1, -5., 5., 1., 1.)
  num_steps = 50  # controls length of observation sequences
  N = 1  # number of observation sequences
  cnt = 20 # number of EM steps

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  VBlearner = make_Gaussian_VBHMM(x)

  incr = True
  last = -np.inf
  eps = 0.01

  for j in range(0, cnt):
    VBlearner.VB_step()
    elbo = VBlearner.elbo()
    incr = ((elbo + eps) > last) and incr
    last = elbo

  return incr

def elbo_increase_multi_Gaussian():
  """
  Run VB on a VBHMM; ensure that elbo is always increasing
  """
  mu_0 = np.array([0., 0.])
  mu_1 = np.array([10., 10.])
  sigma_0 = np.eye(2)
  M_true = make_multi_Gaussian_HMM(0.9, 0.1, mu_0, mu_1, sigma_0, sigma_0)
  num_steps = 5  # controls length of observation sequences
  N = 1  # number of observation sequences
  cnt = 3# number of EM steps

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  VBlearner = make_multi_Gaussian_VBHMM(x)

  incr = True
  last = -np.inf
  eps = 0.01

  for j in range(0, cnt):
    VBlearner.VB_step()
    elbo = VBlearner.elbo()
    incr = ((elbo + eps) > last) and incr
    last = elbo

  return incr

def compare_simple(eps):
  """
  Run 20 steps of VB on a VBHMM with bad priors; ensure that the generated
  HMM has transition matrix and emissions parameters within eps of the
  (highly clustered) original
  """
  
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 200  # controls length of observation sequences
  N = 5  # number of observation sequences
  cnt = 20

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  VBlearner = make_VBHMM(x)

  for j in range(0, cnt):
    VBlearner.VB_step()

  diff = VBlearner.gen_M().compare(M_true)

  return (diff < eps)
  

def compare_correct(eps):
  """
  Run 5 steps of VB on a VBHMM with correct priors; ensure after 5 steps
  that the generated HMM has transition matrix and emissions parameters
  within eps of the original HMM
  """
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 200  # controls length of observation sequences
  N = 5  # number of observation sequences
  cnt = 5

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  VBlearner = make_correct_VBHMM(M_true, x)

  for j in range(0, cnt):
    VBlearner.VB_step()

  diff = VBlearner.gen_M().compare(M_true)

  return (diff < eps)

def biased_coins_HMM(p, q, t_1, t_2):
  """
  This method returns a 2 state HMM, where state 1 is a p-biased coin,
  state 2 is a q-biased coin, the probability of staying in state
  1 is t_1, and the probability of staying in state 2 is t_2.
  The initial distribution pi is uniform.
  """
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  eigval, eigvec = np.linalg.eig(A)
  maxind = max((l, i) for i, l in enumerate(eigval))[1]
  #pi = eigvec[:,maxind]
  #if np.min(pi) < 0:
  #  pi = -pi
  #  pi = (1./np.sum(pi))*pi
  pi = np.array([0.5, 0.5])
  D = [mn([p, 1. - p]), mn([q, 1. - q])]
  
  return HMM(K, A, pi, D)

def make_correct_VBHMM(M_true, x):
  """
  Factory method to create "correct" VBHMM true HMM M_true
  """
  K = 2
  u_A = Dirichlet([2., 2.])
  u_pi = Dirichlet(np.exp(M_true.pi)*20.)
  u_D = []
  for k in range(0, K):
    u_D.append(Dirichlet(M_true.D[k].params*20.))
  D = [mn([0.5, 0.5]), mn([0.5, 0.5])]
  res = VBHMM(K, u_A, u_pi, u_D, D, x)
  for k in range(0, K):
    res.w_A[k].set_natural(20.*np.exp(M_true.A[k]))
  return res

def make_VBHMM(x):
  """
  Factory method to create a VBHMM
  """
  K = 2
  p = 0.5
  q = 0.5
  # specify hyperparams
  u_A = Dirichlet(np.array([3., 3.]))
  u_pi = Dirichlet(np.array([2., 2.]))
  u_D = [Dirichlet(np.array([3., 2.]))] + [Dirichlet(np.array([2., 3.]))]
  D = [mn([p, 1. - p]), mn([q, 1. - q])]
  return VBHMM(K, u_A, u_pi, u_D, D, x)

def make_Gaussian_HMM(t_1, t_2, m_1, m_2, s_1, s_2):
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([0.5, 0.5])
  D = [norm([m_1, s_1]), norm([m_2, s_2])]
  
  return HMM(K, A, pi, D)

def make_multi_Gaussian_HMM(t_1, t_2, m_1, m_2, s_1, s_2):
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([0.5, 0.5])
  D = [mnorm([m_1, s_1]), mnorm([m_2, s_2])]
  
  return HMM(K, A, pi, D)

def make_Gaussian_VBHMM(x):
  """
  Factory method to create a VBHMM
  """
  K = 2
  p = 0.5
  q = 0.5
  # specify hyperparams
  u_A = Dirichlet(np.array([3., 3.]))
  u_pi = Dirichlet(np.array([2., 2.]))
  # hyperparams for the emissions:
  # centered around 0, sigma = 1
  suff_stats_one = [0., 2., 2., 2.]

  # centered around 5, sigma = 1
  suff_stats_two = [10., 2., 2., 2.]
  u_D = [nics(suff_stats_one), nics(suff_stats_two)]
  D = [norm([0., 1.]), norm([0., 1.])]
  return VBHMM(K, u_A, u_pi, u_D, D, x)

def make_multi_Gaussian_VBHMM(x):
  """
  Factory method to create a VBHMM
  """
  K = 2
  # specify hyperparams
  u_A = Dirichlet(np.array([3., 3.]))
  u_pi = Dirichlet(np.array([2., 2.]))
  # hyperparams for the emissions:
  # centered around (-5, 5) sigma = I
  suff_stats_one = [np.array([-15., -15.]), np.eye(2), 3., 3.]

  # centered around 5, sigma = I
  suff_stats_two = [np.array([15., 15.]), np.eye(2), 3., 3.]
  u_D = [niw(suff_stats_one), niw(suff_stats_two)]
  mu = np.array([0., 0.])
  sigma = np.eye(2)
  D = [mnorm([mu, sigma]), mnorm([mu, sigma])]
  return VBHMM(K, u_A, u_pi, u_D, D, x)

run()
