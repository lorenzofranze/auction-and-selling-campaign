import numpy as np

from .AbstractAuction import Auction

# second price auctions are truthful (bid == value) 
class SecondPriceAuction(Auction):
    def __init__(self, ctrs):
        # ctrs = click-through rates (slot prominence * ad quality)
        self.ctrs = ctrs
        # assumed to be known by the auctioneer (can estimate them)
        self.n_adv = len(self.ctrs) # number of advertisers
    
    def get_winners(self, bids):
        # we compute the expected utility of each ad for the advertiser
        adv_values = self.ctrs*bids
        # sort the values in ascending order
        adv_ranking = np.argsort(adv_values)
        # the winner is chosen both based on its bid and the ad's ctr
        winner = adv_ranking[-1]
        return winner, adv_values
    
    def get_payments_per_click(self, winners, values, bids):
        adv_ranking = np.argsort(values)
        second = adv_ranking[-2]
        # apply formula for payment of second price auctions
        payment = values[second]/self.ctrs[winners]
        return payment.round(2)