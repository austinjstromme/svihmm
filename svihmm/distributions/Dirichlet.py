# external imports
import math
import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma 
from scipy.special import digamma as dg
from numpy import random as nprand

# internals
import context
from Exponential import Exponential
import utils.LogMatrixUtil as lm

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

    Returns:
      x: a sample from this.
    """
    # TODO: implement this
    raise NotImplementedError()

  def get_natural(self):
    """
    Returns the natural parameters of this Dirichlet.

    Returns:
      w: np.array of length L, natural parameters for this.
    """
    return self.params

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w

    Args:
      w: np.array of length L of new natural parameters
    """
    self.params = w

  def KL_div(self, other):
    """
    Computes the KL divergence between self and other; i.e.
      KL(other || self).

    Returns:
      x: KL(other || self).
    """
    alo = other.get_natural() + 1.
    als = self.get_natural() + 1.
    alo_sum = np.sum(alo)

    res = np.log(gamma(alo_sum)/gamma(np.sum(als)))
    temp = 0.
    for j in range(0, self.params.shape[0]):
      temp += np.log(gamma(alo[j])/gamma(als[j]))
      temp -= ((alo[j] - als[j])*(dg(alo[j]) - dg(alo_sum)))
    res -= temp
    if math.isnan(res) or (res == np.inf):
      res = 0.
    return res

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    #TODO: implement this
    raise NotImplementedError()

