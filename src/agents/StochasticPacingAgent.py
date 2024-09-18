import numpy as np

from .AbstractBiddingAgent import BiddingAgent

class StochasticPacingAgent(BiddingAgent):
    def __init__(self, valuation, budget, T, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta # learning rate
        self.T = T # number of auctions
        self.rho = self.budget/self.T # average budget per auction
        self.lmbd = 1 # pacing multiplier
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation/(self.lmbd+1)
    
    def update(self, f_t, c_t, m_t = None):
        # projection on [0, 1/rho]-> values smaller than 0 become 0 and higher than 1/rho become 1/rho
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        self.budget -= c_t
    
