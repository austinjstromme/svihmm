import numpy as np
import EM_alg as em
from multinoulli_emitter import mn_emitter as mne


def simulate_simple_HMM(p, t_biased, t_fair, num_steps):
  """
  This method returns a list of num_steps observations
  from a simulated 2 state HMM, where state 1 is a fair coin
  and state 2 is a biased coin with p(Heads | biased) = p and
  the probability of moving from fair to baised is t_biased
  and the probability of moving from biased to fair is t_fair
  """
  state = np.random.binomial(1,0.5)
  obs = [gen_obs(state, p)]
  for t in range(1, num_steps):
    state = gen_trans(state, t_biased, t_fair)
    obs.append(gen_obs(state, p))
  return obs

def gen_obs(state, p):
  if state == 0:
    return np.random.binomial(1, 0.5)
  else:
    return np.random.binomial(1, p)

def gen_trans(state, t_biased, t_fair):
  if state == 0:
    return np.random.binomial(1, t_biased)
  else:
    return 1 - np.random.binomial(1, t_fair)


p = 0.9 
t_biased = 0.5
t_fair = 0.1
num_steps = 500
N = 1
x = [simulate_simple_HMM(p, t_biased, t_fair, num_steps) 
      for i in xrange(0,N)]

print("first simulation gives " + str(x[0]))

#initialize everybody with simple guesses
A = np.array([[0.4,0.6],[0.3,0.7]])
B = [mne(np.array([0.5,0.5])) for j in xrange(0,2)]
pi = np.array([0.5 for i in xrange(0,2)])

num_iter_em = 100

for j in range(0, num_iter_em):
  em.EM_main(x, A, B, pi)

print("A = " + str(A))
for j in range(0,2):
  print("B[" + str(j) + "] = " + str(B[j].get_params()))
print("pi = " + str(pi))
