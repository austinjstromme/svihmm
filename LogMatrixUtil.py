"""
This file contains utility functions for convenient manipulation of
matrices in log domain. That is, if A, B are np.arrays and A', B' are
the results of np.log(A), np.log(B), then we would like to be able to
easily compute (A * B)' or (A \odot B)'. These operations are both
implemented below.
"""
import numpy as np
from scipy.misc import logsumexp

def mult(a, b):
  """
  Returns log(exp(a) \odot \exp(b)). Assumes a, b are np.arrays
    with compatible dimensions.
  """
  return a + b;

def dot(a, b):
  """
  Returns log(exp(a) \dot \exp(b)). Assumes a, b are np.arrays
    with compatible dimensions.

  See Stack Overflow Post "Numerically stable way to multiply log prob matrices"
  """
  max_a = np.max(a)
  max_b = np.max(b)
  c = np.dot(np.exp(a - max_a), np.exp(b - max_b))
  c = np.log(c)
  c += max_a + max_b
  return c

def norm(a):
  """
  Returns [b, Z] such that b = a - log(Z) and Z = exp(logsumexp(a))
  """
  Z = np.exp(logsumexp(a))
  b = a - np.log(Z)
  return [b, Z]

def outer(a, b):
  """
  Returns log( exp(a) * exp(b)^T) where a and b are column vectors.
  """
  res = np.zeros((a.shape[0], b.shape[0]))
  for i in range(0, a.shape[0]):
    res[i] = b + a[i]
  return res
