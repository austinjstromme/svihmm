from Distribution import Distribution

class Exponential(Distribution):

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.
    Prior defaults to sensible value if not specified. See
    specific distributions for more information.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def gen_sample(self):
    """
    Generates a sample from this distribution.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def get_natural(self):
    """
    Returns the vector of natural parameters.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def set_natural(self, w):
    """
    Sets the vector of natural parameters.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assume this dist is the one corresponding to
    the jth hidden state.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    raise NotImplementedError("Exponential.py is an interface.")
     


