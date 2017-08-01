"""
This file does some basic testing of our EM implementation.
"""
# external imports
import numpy as np

# internals
from BayesHMM.distributions.Multinoulli import Multinoulli as mn
from Gaussian import Gaussian as norm
from Dirichlet import Dirichlet
from HMM import HMM
from States import States

def run():
  eps = 0.2
  res = 0
  num_tests = 3

  print("Running HMM tests...")

  #...correct init case...
  temp = int(compare_correct(2*eps))
  res += temp
  print("  correct init case: " + str(temp) + "/1")

  #...decreasingness test...
  temp = int(NLL_decrease())
  res += temp
  print("  decrease test: " + str(temp) + "/1")

  #... Gaussian decreasingness test...
  temp = int(NLL_decrease_Gaussian())
  res += temp
  print("  Gaussian HMM decrease test: " + str(temp) + "/1")

  print("HMM tests completed: " + str(res) + "/" + str(num_tests))

  return (res == num_tests)

def NLL_decrease():
  """
  Run EM on a HMM; ensure that NLL is always decreasing
  """
  M_true = biased_coins_HMM(0.9, 0.1, 0.05, 0.8)
  num_steps = 200  # controls length of observation sequences
  N = 5  # number of observation sequences
  cnt = 10 # number of EM steps

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  EMlearner = biased_coins_HMM(0.6, 0.4, 0.7, 0.1)
  EMstates = States(EMlearner, x)

  decr = True
  last = np.inf

  for j in range(0, cnt):
    EMlearner.EM_step(EMstates)
    NLL = -EMstates.LL()
    decr = (last > NLL) 
    last = NLL

  return decr

def NLL_decrease_Gaussian():
  """
  Run EM on a HMM; ensure that NLL is always decreasing
  """
  M_true = make_Gaussian_HMM(0.9, 0.1, 0., 10., 1., 1.)
  num_steps = 100  # controls length of observation sequences
  N = 5  # number of observation sequences
  cnt = 10 # number of EM steps

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  EMlearner = make_Gaussian_HMM(0.6, 0.4, 0. , 5., 1., 1.)
  EMstates = States(EMlearner, x)

  decr = True
  last = np.inf

  for j in range(0, cnt):
    EMlearner.EM_step(EMstates)
    NLL = -EMstates.LL()
    decr = (last > NLL) 
    last = NLL

  return decr

def make_Gaussian_HMM(t_1, t_2, m_1, m_2, s_1, s_2):
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([0.5, 0.5])
  D = [norm([m_1, s_1]), norm([m_2, s_2])]
  
  return HMM(K, A, pi, D)


def compare_simple(eps):
  """
  Run 20 steps of EM on a HMM with bad priors; ensure that the generated
  HMM has transition matrix and emissions parameters within eps of the
  (highly clustered) original
  """
  
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 50  # controls length of observation sequences
  N = 10  # number of observation sequences
  cnt = 50

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  EMlearner = biased_coins_HMM(0.6, 0.4, 0.7, 0.1)
  EMstates = States(EMlearner, x)

  for j in range(0, cnt):
    EMlearner.EM_step(EMstates)

  diff = EMlearner.compare(M_true)

  return (diff < eps)
  

def compare_correct(eps):
  """
  Run 5 steps of EM on a HMM with correct priors; ensure after 5 steps
  that it has transition matrix and emissions parameters within eps of
  the original HMM.
  """
  M_true = biased_coins_HMM(0.9, 0.1, .99, .99)
  num_steps = 200  # controls length of observation sequences
  N = 2 # number of observation sequences
  cnt = 5 

  # generate observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  EMlearner = biased_coins_HMM(0.9, 0.1, .99, .99)
  EMstates = States(EMlearner, x)

  for j in range(0, cnt):
    EMlearner.EM_step(EMstates)

  diff = EMlearner.compare(M_true)

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
  #eigval, eigvec = np.linalg.eig(A)
  #maxind = max((l, i) for i, l in enumerate(eigval))[1]
  #pi = eigvec[:,maxind]
  #if np.min(pi) < 0:
  #  pi = -pi
  #  pi = (1./np.sum(pi))*pi
  pi = np.array([0.5, 0.5])
  D = [mn([p, 1. - p]), mn([q, 1. - q])]
  
  return HMM(K, A, pi, D)


run()
