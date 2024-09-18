from .AbstractEnvironment import Environment

class StochasticBiddingCompetitors(Environment):
    def __init__(self, distribution, n_competitors):
        # distribution is a lambda function
        self.distribution = distribution
        self.n_competitors = n_competitors

    def round(self):
        bids = self.distribution(self.n_competitors)
        return bids
    
