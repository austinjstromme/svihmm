import numpy as np
import fb_compute as fb
from Multinoulli import Multinoulli
import math
import scipy as sc
import LogMatrixUtil as lm

class HMM(object):
  """
  A HMM is an abstraction of a Hidden Markov Model. In this abstraction
  each HMM has several parameters associated to it, including the number
  of hidden states K, the transition matrix A from hidden states to hidden
  states, an initial distribution pi, and a list of distributions D. Note
  that all probabilities are in the log domain.
  """

  def __init__(self, K, A, pi, D):
    """
    Initializes a Hidden Markov Model (HMM) with K hidden states.

    K: a positive integer givin the number of hidden states.
    A: a KxK np.array of transition probabilities between states. NOT
      in log domain.
    pi: a 1xK np.array of start probabilities. NOT in log domain.
    D: a list of K Exponential Distributions (see Exponential.py), one for
      each state
    """
    #TODO: add checks that the arguments are of the right form
    self.K = K
    self.A = np.log(A)
    self.pi = np.log(pi)
    self.D = D

  def get_params(self):
    """
    Returns the parameters K, A, pi, and D,
    in a list: [K, A, pi, D]; converts probabilities
    back from log domain.
    """
    return [self.K, np.exp(self.A), np.exp(self.pi), self.D]

  def compare(self, M, local=False):
    """
    Returns a real number representing the similarity between this
    and another HMM M; symmetric operation. If local=True we don't
    compare the pi vectors. Heuristic for verifying structure learning.
    """
    res = 0.
    if not local:
      res = np.linalg.norm(np.exp(self.pi) - np.exp(M.pi))
    res += np.linalg.norm(np.exp(self.A) - np.exp(M.A))
    for k in range(0, self.K):
      res += np.linalg.norm(self.D[k].params - M.D[k].params)
    return res

  def M_step(self, S):
    """
    Does a single M step.

    Args:
      S: a States object which points to this HMM.

    Effects:
      self: updates this HMM's parameter values
    """
    self.update_start(S)
    self.update_trans(S)
    for k in range(0, self.K):
      self.D[k].update_params(S.data, S.gamma, k)

  def EM_step(self, S):
    """
    Does a single step of EM.

    Args:
      S: states object to do EM with; must point to this HMM

    Effects:
      self: updates parameters according to M step
      S: gamma and xi are update according to E step
    """
    # e-step
    S.e_step()
    
    # update A and pi
    self.update_trans(S)
    self.update_start(S)
    # update emissions
    for k in range(0, self.K):
      log_suff_stat = np.log(self.D[k].get_expected_suff(S, k))
      self.D[k].set_natural(log_suff_stat)
    
  def update_trans(self, S):
    """
    Updates the transition matrix A.
    """
    A_expect = S.get_trans()

    #normalize and put in
    for j in range(0, self.K):
      self.A[j] = lm.norm(A_expect[j])[0]

  def update_start(self, S):
    """
    Updates the start probability vector pi.
    """
    self.pi = S.get_start() - np.log(len(S.gamma))

  def EM(self, S, N):
    """
    Does N steps of EM with states object S.
    """
    for i in range(0, N):
      self.EM_step(S)

  def gen_obs(self, n):
    """
    Generate a list of n observations.
    """
    m = Multinoulli(np.exp(self.pi))
    curr = m.gen_sample()
    obs = [self.D[curr].gen_sample()]
    for i in range(1, n):
      m = Multinoulli(np.exp(self.A[curr]))
      curr = m.gen_sample()
      obs.append(self.D[curr].gen_sample())
    return obs

  def __str__(self):
    res = "HMM with K = " + str(self.K) + "\n"
    res += "  A = " + str(np.exp(self.A)) + "\n"
    res += "  pi = " + str(np.exp(self.pi)) + "\n"
    res += "  D = "
    for k in range(0, self.K):
      res += str(self.D[k]) + "\n"
    return res

