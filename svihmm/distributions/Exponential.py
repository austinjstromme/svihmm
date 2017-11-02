from .Distribution import Distribution

class Exponential(Distribution):

  def __init__(self, params, prior=None):
    """
    Initializes the distribution with given parameters and prior.
    Prior defaults to sensible value if not specified. See
    specific distributions for more information.

    Args:
      params: the parameters to initialize this distribution to.
      prior: the distribution to use as the prior for this.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def gen_sample(self):
    """
    Generates a sample from this distribution.

    Returns:
      x: a sample from this.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def get_natural(self):
    """
    Returns the natural parameters of this distribution.

    Returns:
      w: np.array of length L, natural parameters for this.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def set_natural(self, w):
    """
    Updates the parameters so the natural parameters become w.

    Args:
      w: np.array of length L of new natural parameters
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def gen_log_expected(self):
    """
    Generates the log expected distribution according to the
    current prior.

    Returns:
      p: a distribution such that p(x) = exp(E[ln(q(x))]) where the expectation
      is over the distribution on q via the prior.

    NOTE: the returned distribution may only implement Distribution.py.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def get_expected_suff(self, S, j):
    """
    Returns the vector of the expected sufficient statistics from
    a given states object; assume this dist is the one corresponding to
    the jth hidden state.

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.

    Returns:
      w: a np.array of length L where is the number of parameters of the prior.
    """
    raise NotImplementedError("Exponential.py is an interface.")

  def maximize_likelihood(self, S, j):
    """
    Updates the parameters of this distribution to maximize the likelihood
    of it being the jth hidden state's emitter.

    Args:
      S: States object.
      j: the hidden state this distribution corresponds to.
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
