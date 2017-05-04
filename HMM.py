import numpy as np
import fb_compute as fb
from Multinoulli import Multinoulli
import math
import scipy as sc

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
    D: a list of K Distributions (see Distribution.py), one for
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

  def compare(self, M):
    """
    Returns a real number representing the similarity between this
    and another HMM M; symmetric operation.
    """
    #TODO: implement

  def M_step(self, S):
    """
    Does a single M step.

    Args:
      S: a States object which points to this HMM.

    Effects:
      self: updates this HMM's parameter values
    """
    self.update_start(S.gamma)
    self.update_trans(S.xi)
    for k in range(0, self.K):
      self.D[k].update_params(S.data, S.gamma, k)

  def EM_step(self, x):
    """
    Does a single step of EM.

    Args:
      x: a list of N observation sequences; x[i] is a list of length
        T_i of observations
    """
    N = len(x)
    gamma_data = []
    xi_data = []
    # start by doing forwards backwards on each observation sequence
    for j in range(0, N):
      [gamma, xi, Z_vals] = fb.fb_main(x[j], self.A, self.D, self.pi)
      gamma_data.append(gamma)
      xi_data.append(xi)

    # update A and pi
    self.update_trans(xi_data)
    self.update_start(gamma_data)
    # update the distribution at each hidden state
    for k in range(0, self.K):
      self.D[k].update_params(x, gamma_data, k)

  def update_trans(self, xi_data):
    """
    Updates the transition matrix A.
    """
    N = len(xi_data)

    A_expect = np.zeros((self.K, self.K))
    for j in range(0, self.K):
      for k in range(0, self.K):
        A_expect[j][k] = sc.misc.logsumexp([[xi_data[n][t][j][k]
          for t in range(0, len(xi_data[n]))] for n in range(0, N)])

    #normalize and put in
    for j in range(0, self.K):
      self.A[j] = lm.norm(A_expect[j])[0]

  def trans_count(self, xi):
    """
    Computes the expected transition matrix, i.e. returns
      a KxK np.array such that (j,k) is the expected number of
      transitions from state j to state k.
    """
    #return np.array([[sum(xi[:][j][k]) for k in xrange(0,K)] for j
    #                    for j in xrange(0, K)])
    return np.sum(xi, axis=0)

  def update_start(self, gamma):
    """
    Updates the start probability vector pi.
    """
    pi_expect = np.zeros(self.K)
    N = len(gamma_data)
    for k in range(0, self.K)
      sum_k = [gamma[t][0][k] for t in range(0, N)]
      pi_expect[k] = sc.misc.logsumexp(sum_k) - np.log(N)
    self.pi = pi_expect

  def EM(self, x, N):
    """
    Does N steps of EM with observations x.
    """
    for i in range(0, N):
      self.EM_step(x)

  def gen_obs(self, n):
    """
    Generate a list of n observations.
    """
    m = Multinoulli(self.pi)
    curr = m.gen_sample()
    obs = [self.D[curr].gen_sample()]
    for i in range(1, n):
      m = Multinoulli(self.A[curr])
      curr = m.gen_sample()
      obs.append(self.D[curr].gen_sample())
    return obs

  def __str__(self):
    res = "HMM with K = " + str(self.K) + "\n"
    res += "  A = " + str(self.A) + "\n"
    res += "  pi = " + str(self.pi) + "\n"
    res += "  D = "
    for k in range(0, self.K):
      res += str(self.D[k]) + "\n"
    return res

