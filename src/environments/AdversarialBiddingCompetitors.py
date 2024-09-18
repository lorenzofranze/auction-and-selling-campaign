from .AbstractEnvironment import Environment

class AdversarialBiddingCompetitors(Environment):
    
    def __init__(self, bids_sequence, n_competitors, T):
        # bids_sequence is a matrix where rows correspond to competitors and columns to rounds
        self.bids_sequence = bids_sequence
        self.n_competitors = n_competitors
        self.T = T
        self.t = 0

    def round(self):
        bids = self.bids_sequence[:, self.t]
        self.t += 1
        return bids