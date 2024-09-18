import numpy as np

from .AbstractEnvironment import Environment

class AdversarialPricingFullEnvironment(Environment):

    def __init__(self, conversion_probability, theta_seq, cost):
        self.conversion_probability = conversion_probability
        self.theta_seq = theta_seq
        self.cost = cost
        self.t = 0

    def round(self, p_t, n_t):
        d_t = np.array([])
        for p in p_t:
            d = np.random.binomial(n_t, self.conversion_probability(p, self.theta_seq[self.t]))
            d_t = np.append(d_t, d)
        r_t = (p_t - self.cost)*d_t
        self.t += 1
        return d_t, r_t
    
    def reset(self):
        self.t = 0
    