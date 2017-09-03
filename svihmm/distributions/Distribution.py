
class Distribution(object):
  """
  A distribution simply encapsulates a mass function.
  """

  def __init__(self, mass):
    """
    Initializes a distribution with mass method mass.
    """
    self.mass = mass

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    return self.mass(x)
