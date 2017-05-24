import numpy as np
from VBHMM import VBHMM
from Dirichlet import Dirichlet
from HMM import HMM
from Multinoulli import Multinoulli
import matplotlib.pyplot as plt
import pylab
from States import States

def biased_coins_HMM(p, q, t_1, t_2):
  """
  This method returns a 2 state HMM, where state 1 is a p-biased coin,
  state 2 is a q-biased coin, the probability of staying in state
  1 is t_1, and the probability of staying in state 2 is t_2.
  The initial distribution pi is uniform.
  """
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([.5, .5])
  D = [Multinoulli([p, 1. - p]), Multinoulli([q, 1. - q])]
  return HMM(K, A, pi, D)

def make_VBHMM():
  """
  Factory method to create a VBHMM
  """
  K = 2
  p = 0.5
  q = 0.5
  #specify hyperparams
  u_A = Dirichlet(np.array([3., 3.]))
  u_pi = Dirichlet(np.array([2., 4.]))
  u_D = [Dirichlet(np.array([2., 2.]))] + [Dirichlet(np.array([3., 3.]))]
  D = [Multinoulli([p, 1. - p]), Multinoulli([q, 1. - q])]
  return VBHMM(K, u_A, u_pi, u_D, D)

def sim_params_example():
  p = 0.9 
  q = 0.1
  t_1 = 0.05
  t_2 = 0.8
  M_true = biased_coins_HMM(p, q, t_1, t_2)
  num_steps = 200 # controls length of observation sequences
  N = 10 # number of observation sequences
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]
  print("initializing learner and states...")
  M_learner = biased_coins_HMM(p, q, t_1, t_2)
  states = States(M_learner, x)
  #print(" at first, M_learner = " + str(M_learner))

  print("training learner...")
  em_steps = 50
  train_likelihood = []
  for j in range(0, em_steps):
    states.e_step()
    M_learner.M_step(states)
  print("M = " + str(M_learner))
  print("true M = " + str(M_true))

def plot_LL():
  p = 0.9 
  q = 0.1
  t_1 = 0.05
  t_2 = 0.8
  M_true = biased_coins_HMM(p, q, t_1, t_2)
  num_steps = 100 # controls length of observation sequences
  N = 10 # number of observation sequences
 
  # generate observation seqeunces
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  # now initialize the model we will be learning to uniform 
  # everything:
  M_learner = biased_coins_HMM(.4, .6, .5, .5)
  states = States(M_learner, x)
  steps = 50
  NLL = []
  for j in range(0, steps):
    M_learner.M_step(states)
    states.e_step()
    NLL.append(-states.LL())

  plt.plot(NLL)
  plt.title("NLL vs EM steps, N = " + str(N) + ", T = " + str(num_steps)
            + " synthetic")
  plt.savefig("NLL.png")
  #plt.show()

def plot_SVI_L():
  p = 0.9 
  q = 0.1
  t_1 = 0.05
  t_2 = 0.8
  M_true = biased_coins_HMM(p, q, t_1, t_2)
  num_steps = 10000 # controls length of observation sequences
  N = 1 # number of observation sequences
 
  # generate observation seqeunces
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  # now initialize the model we will be learning to uniform 
  # everything:
  SVI_learner = make_VBHMM()
  steps = 15
  L = [10, 50, 100, 200, 400]
  buf = 30
  rho = 0.5
  SVI_elbo = []
  for l in L:
    SVI_states = States(SVI_learner.gen_M(), x)
    for j in range(0, steps):
      SVI_learner.SVI_step(SVI_states, buf, l, rho)
    SVI_elbo.append(SVI_learner.elbo(SVI_states))

  plt.plot(L, SVI_elbo, label="SVI")
  plt.title("elbo vs subchain length, T = " + str(num_steps) + " synthetic")
  plt.savefig("SVIvsL.png")
  #plt.show()

def plot_SVI_elbo():
  p = 0.9 
  q = 0.1
  t_1 = 0.05
  t_2 = 0.8
  M_true = biased_coins_HMM(p, q, t_1, t_2)
  num_steps = 1000 # controls length of observation sequences
  N = 1 # number of observation sequences
 
  # generate observation seqeunces
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  # now initialize the model we will be learning to uniform 
  # everything:
  SVI_learner = make_VBHMM()
  VB_learner = make_VBHMM()
  SVI_states = States(SVI_learner.gen_M(), x)
  VB_states = States(VB_learner.gen_M(), x)
  steps = 15
  L = num_steps/10
  buf = num_steps/20
  rho = 0.5
  SVI_elbo = []
  VB_elbo = []
  for j in range(0, steps):
    rho -=0.005
    SVI_learner.SVI_step(SVI_states, buf, L, rho)
    SVI_elbo.append(SVI_learner.elbo(SVI_states))
    VB_learner.VB_step(VB_states)
    VB_elbo.append(VB_learner.elbo(VB_states))

  plt.plot(SVI_elbo, color="r", label="SVI")
  plt.plot(VB_elbo, color="b", label="VB")
  pylab.legend(loc='center right')
  plt.title("elbo vs steps, T = " + str(num_steps) + " synthetic")
  plt.savefig("SVIvsVB.png")

def plot_elbo():
  p = 0.9 
  q = 0.1
  t_1 = 0.05
  t_2 = 0.8
  M_true = biased_coins_HMM(p, q, t_1, t_2)
  num_steps = 100 # controls length of observation sequences
  N = 10 # number of observation sequences
 
  # generate observation seqeunces
  x = [M_true.gen_obs(num_steps) for j in xrange(0, N)]

  # now initialize the model we will be learning to uniform 
  # everything:
  M_learner = make_VBHMM()
  states = States(M_learner.gen_M(), x)
  steps = 50
  elbo = []
  for j in range(0, steps):
    M_learner.VB_step(states)
    elbo.append(M_learner.elbo(states))

  plt.plot(elbo)
  plt.title("elbo vs VB steps, VBHMM, N = " + str(N) + ", T = " + str(num_steps) + " synthetic")
  plt.savefig("VB.png")
#print("init to true params example: 10 observation sequences of length 100")
#sim_params_example()

plot_SVI_L()



