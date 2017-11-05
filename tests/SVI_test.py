"""
This file does some basic testing of the VBHMM class.
"""
# external packages
import numpy as np

# internals
from svihmm.distributions.Multinoulli import Multinoulli as mn
from svihmm.distributions.Dirichlet import Dirichlet
from svihmm.models.HMM import HMM
from svihmm.models.States import States
from svihmm.models.VBHMM import VBHMM

def run():
  eps = 0.25
  res = 0
  num_tests = 2

  print("Running SVI tests...")

  temp = int(compare_correct(eps))
  res += temp
  print("  correct init case: " + str(temp) + "/1")


  #...simple case...
  temp = int(compare_simple(6*eps))
  res += temp
  print("  simple case: " + str(temp) + "/1")

  print("SVI tests passed: " + str(res) + "/" + str(num_tests))

  return (res == num_tests)
  


def compare_simple(eps):
  """
  Run 20 steps of SVI on a VBHMM with bad priors; ensure that the generated
  HMM has transition matrix and emissions parameters within eps of the
  (highly clustered) original
  """
  
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 200  # controls length of observation sequences
  N = 1  # number of observation sequences
  cnt = 100

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  SVIlearner = make_VBHMM(x)

  buf = 5
  L = 20
  rho = 0.01

  for j in range(0, cnt):
    SVIlearner.SVI_step(buf, L, rho)

  diff = SVIlearner.gen_M().compare(M_true, local=True)

  if (diff >= eps):
    print("  diff = " + str(diff))

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
  pi = eigvec[:,maxind]
  if np.min(pi) < 0:
    pi = -pi
    pi = (1./np.sum(pi))*pi
  D = [mn([p, 1. - p]), mn([q, 1. - q])]
  
  return HMM(K, A, pi, D)

def compare_correct(eps):
  """
  Run some steps of SVI on a VBHMM with correct priors; ensure the
  generated HMM has transition matrix and emissions parameters
  within eps of the original HMM
  """
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 200  # controls length of observation sequences
  N = 1  # number of observation sequences
  cnt = 20

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  # generate observation sequences
  SVIlearner = make_correct_VBHMM(M_true, x)

  buf = 5
  L = 20
  rho = 0.01

  for j in range(0, cnt):
    SVIlearner.SVI_step(buf, L, rho)

  found_M = SVIlearner.gen_M()

  diff = SVIlearner.gen_M().compare(M_true, local=True)

  if (diff >= eps):
    print("  diff = " + str(diff))

  return (diff < eps)

def make_correct_VBHMM(M_true, x):
  """
  Factory method to create "correct" VBHMM according to M_true
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
  Factory method to create a fairly generic VBHMM
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

