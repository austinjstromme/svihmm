import numpy as np
import fb_compute as fb

class HMM(object):
"""
A HMM is an abstraction of a Hidden Markov Model. In this abstraction
each HMM has several parameters associated to it, including the number
of hidden states K, the transition matrix A from hidden states to hidden
states, an initial distribution pi, and a list of distributions D.
"""

  def __init__(self, K, A, pi, D):
    """
    Initializes a Hidden Markov Model (HMM) with K hidden states.

    K: an positive integer givin the number of hidden states.
    A: a KxK np.array of transition probabilities between states.
    pi: a 1xK np.array of start probabilities.
    D: a list of K Distributions (see Distribution.py), one for
      each state
    """
    #TODO: add checks that the arguments are of the right form
    self.K = K
    self.A = A
    self.pi = pi
    self.D = D

  def get_params(self):
    """
    Returns the parameters K, A, pi, and D,
    in a list: [K, A, pi, D].
    """
    return [self.K, self.A, self.pi, self.D]

  def compare(self, M):
    """
    Returns a real number representing the similarity between this
    and another HMM M; symmetric operation.
    """
    #TODO: implement

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
      [gamma, xi] = fb.fb_main(x, self.A, self.D, self.pi)
      gamma_data.append(gamma)
      xi_data.append(xi)

    # update A and pi
    self.update_trans(xi_data)
    self.update_start(gamma_data)
    # update the distribution at each hidden state
    for k in range(0, self.K):
      self.D[k].update_params(x, gamma_data)

  def update_trans(self, xi_data):
  """
  Updates the transition matrix A.
  """
  N = len(xi_data)
  A_expect = sum([np.sum(xi_data[i], axis=0) for i in range(0, N)])

  for j in range(0, self.K):
    row_sum = sum(A_expect[j])
    for k in range(0, self.K):
      self.A[j][k] = A_expect[j][k]/row_sum

  def trans_count(self, xi):
  """
  Computes the expected transition matrix, i.e. returns
    a KxK np.array such that (j,k) is the expected number of
    transitions from state j to state k.
  """
    #return np.array([[sum(xi[:][j][k]) for k in xrange(0,K)] for j
    #                    for j in xrange(0, K)])
    return np.sum(xi, axis=0)

  def update_start(gamma_data)
  """
  Updates the start probability vector pi.
  """
  pi_expect = np.zeros(self.K)
  N = len(gamma_data)
  for i in range(0, N):
    pi_expect += self.start_count(gamma_data[i])*(1.0/N)
  self.pi = pi_expect

  def start_count(self, gamma)
  """
  Computes the expected initial probability vector.
  """
    return np.array([gamma[0][k] for k in xrange(0, self.K)])

  def EM(self, x, N):
    """
    Does N steps of EM with observations x.
    """
    for i in range(0, N):
      EM_step(x)

  def generate_obs(self, n):
    """
    Generate a list of n observations.
    """
    #TODO: implement
    pass
