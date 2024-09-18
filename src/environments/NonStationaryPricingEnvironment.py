from .AbstractEnvironment import Environment
import numpy as np

class NonStationaryPricingEnvironment(Environment):
    def __init__(self, cost, conversion_probabilities, T_interval, seed):
        np.random.seed(seed)
        self.cost = cost
        self.conversion_probability = conversion_probabilities
        self.T_interval = T_interval
        self.t = 0
        self.interval = 0


    def round(self, p_t, n_t):
        d_t = np.random.binomial(n_t, self.conversion_probability[self.interval](p_t))
        r_t = (p_t - self.cost)*d_t
        self.t += 1
        # update interval -> different demand curve
        if self.t % self.T_interval == 0:
            self.interval += 1
        return d_t, r_t
