
class Distribution(object):

  def __init__(self):
    raise NotImplementedError("Distribution.py is an interface.")

  def gen_sample(self):
    raise NotImplementedError("Distribution.py is an interface.")

  def mass(self, x):
    """
    Computes the probability of an observation x.

    Args:
      x: a single observation

    Returns:
      p(x)
    """
    raise NotImplementedError("Distribution.py is an interface.")
