import numpy as np

from .AbstractEnvironment import Environment

class AdversarialPricingBanditEnvironment(Environment):

    def __init__(self, conversion_probability, theta_seq, cost):
        self.conversion_probability = conversion_probability
        self.theta_seq = theta_seq
        self.cost = cost
        self.t = 0

    def round(self, p_t, n_t):
        d_t = np.random.binomial(n_t, self.conversion_probability(p_t, self.theta_seq[self.t]))
        self.t += 1
        r_t = (p_t - self.cost)*d_t
        return d_t, r_t
    