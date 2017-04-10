
class mn_emitter:
  """
  This is a class for multinoulli emitters.
  
  Fields:
    p: 1xs np.array of emission probabilities
        where s is the number of classes
  """
  
  def __init__(self, p):
    """
    Initializes an mn_emitter object.
    """
    self.p = p


  def get_params(self):
    """
    Returns p.
    """
    return self.p

  def prob_obs(self, x):
    """
    returns the probability of observation x
    """
    return self.p[x]
