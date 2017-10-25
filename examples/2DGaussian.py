"""
We learn a HMM with 2 hidden states each with 2D Gaussian emissions.
"""
#TODO: make this easier to understand, also make the demo plot the difference
# of buffer size

# import numpy and svihmm
import numpy as np
import context  # for svihmm

# import models
from svihmm.models.States import States # message passing object
from svihmm.models.HMM import HMM
from svihmm.models.VBHMM import VBHMM

# import distributions
from svihmm.distributions.Dirichlet import Dirichlet
from svihmm.distributions.NDGaussian import NDGaussian as mnorm
from svihmm.distributions.NormalInverseWishart import NormalInverseWishart as niw

# import plotting
import matplotlib.pyplot as plt

def make_VBHMM():
  """
  Factory method to create a fairly generic VBHMM.
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
  return VBHMM(K, u_A, u_pi, u_D, D)
  
def make_HMM():
  t_1 = 0.9
  t_2 = 0.1
  m_1 = np.array([0., 0.])
  m_2 = np.array([10., 10.])
  s_1 = np.eye(2)
  s_2 = np.eye(2)
  K = 2
  A = np.array([[t_1, 1. - t_1], [1. - t_2, t_2]])
  pi = np.array([0.5, 0.5])
  D = [mnorm([m_1, s_1]), mnorm([m_2, s_2])]
  
  return HMM(K, A, pi, D)

def main():
  # create the HMM which will make our synthetic data
  HMM_true = make_HMM()
  # number of observations to generate
  obs_sequence_length = 100000
  # generate the data
  data = [HMM_true.gen_obs(obs_sequence_length)]

  # make our learner and initialize its local belief probabilities
  learner = make_VBHMM()
  states = States(learner.gen_M(), data)

  # run 10 steps of SVI with buf = 5, L = 20, learning rate rho = 0.01
  for j in range(10):
    learner.SVI_step(states, 5, 20, 0.01)

  elbos = []
  # now run 10 more steps, but save the elbo (otherwise the plot is unhelpful)
  for j in range(10):
    learner.SVI_step(states, 5, 20, 0.01)
    elbos.append(learner.elbo(states))

  plt.plot(elbos)
  plt.xlabel("SVI step - 10")
  plt.ylabel("ELBO value")

  plt.title("SVI demo, buf = 5, L = 20, rho = 0.01")

  plt.show()

if __name__ == "__main__":
  main()
