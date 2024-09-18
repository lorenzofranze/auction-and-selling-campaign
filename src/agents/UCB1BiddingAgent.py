import numpy as np
import scipy.optimize as opt

from .AbstractBiddingAgent import BiddingAgent

class UCB1BiddingAgent(BiddingAgent):
    def __init__(self, budget, bids, T, range=1):
        self.budget = budget
        self.bids = bids
        self.K = len(bids)
        self.T = T
        self.rho = self.budget/self.T
        self.range = range
        self.b_t = None
        self.f_avg = np.zeros(self.K)
        self.c_avg = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)
        self.t = 0
    
    def bid(self):
        if self.budget < 1:
            return 0 # budget depleted
        if self.t < self.K:
            self.b_t = self.t 
        else:
            # compute utility UCB and cost LCB for every arm
            f_ucb = self.f_avg + self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)*(1 - self.t/self.T)**5
            c_lcb = self.c_avg - self.range*np.sqrt(2*np.log(self.T)/self.N_pulls)*0.01*(1 - self.t/self.T)**5
            # we reduce the optimism in the cost because otherwise we run out of budget early

            prog = opt.linprog(c=-f_ucb, A_ub=[c_lcb], b_ub=[self.rho], A_eq=[np.ones(self.K)], b_eq=[1], bounds=(0,1))
            # prog = opt.linprog(c=-self.f_avg, A_ub=[self.c_avg], b_ub=[self.rho], A_eq=[np.ones(self.K)], b_eq=[1], bounds=(0,1), method="highs")
            gamma = prog.x # solution of the linear program
            # for numerical reasons the probabilities might not sum up to 1
            sum_gamma = np.sum(gamma)
            gamma = gamma/sum_gamma # this ensures that the sum of the probabilities is 1
            self.b_t = np.random.choice(range(self.K), p=gamma) # sample index of the bid to play
        return self.bids[self.b_t]
    
    def update(self, f_t, c_t, m_t = None):
        self.N_pulls[self.b_t] += 1
        self.f_avg[self.b_t] += (f_t - self.f_avg[self.b_t])/self.N_pulls[self.b_t]
        self.c_avg[self.b_t] += (c_t - self.c_avg[self.b_t])/self.N_pulls[self.b_t]
        self.budget -= c_t
        self.t += 1