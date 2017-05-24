from Exponential import Exponential
import math
from numpy import random as nprand
from scipy.misc import logsumexp
import LogMatrixUtil as lm
import numpy as np
from scipy.special import gamma
from scipy.special import digamma

class Dirichlet(Exponential):
  """
  This class represents a Dirichlet distribution.

  Fields:
    params: the natural parameters of a Dirichlet distribution in
      canonical form
  """

  def __init__(self, params):
    """
    Initializes a Dirchlet distribution.

    Args:
      params: vector alpha such that this becomes a Dir(alpha) distribution
    """
    # store natural 
    self.params = np.array(params) - 1.0

  def gen_sample(self):
    """
    Generates a single sample.
    """
    #TODO: implement
    raise NotImplementedError()

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w

    Args:
      w: np.array of length L of new natural parameters
    """
    self.params = w


  def get_natural(self):
    """
    Returns the natural parameters of this Dirichlet.
    """
    return self.params

  def natural_update(self, delta_w, eta):
    """
    Updates the natural parameters of this Dirichlet using delta_w. 

    Args:
      delta_w: a np.array of size L
      eta: weight of the update

    Effects:
      self: the natural parameters w become (1 - eta)*w + eta*delta_w
    """
    #TODO: move this method into Exponential family class
    w = self.get_natural()
    self.set_natural((1. - eta)*w + eta*delta_w)

  def KL_div(self, other):
    """
    Computes the KL divergence between self and other; i.e.
      KL(other || self)
    """
    alo = other.get_natural() + 1.
    als = self.get_natural() + 1.
    alo_sum = np.sum(alo)


    res = np.log(gamma(alo_sum)/gamma(np.sum(als)))
    temp = 0.
    for j in range(0, self.params.shape[0]):
      temp += np.log(gamma(alo[j])/gamma(als[j]))
      temp -= ((alo[j] - als[j])*(digamma(alo[j]) - digamma(alo_sum)))
    res -= temp
    if math.isnan(res):
      res = 0.
    if res == np.inf:
      res = 0.
    return res
















