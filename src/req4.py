import os

import argparse
import numpy as np
import random

import agents as ag
import environments as envi
import auctions as au
from utils import *

import warnings
"""
We have 3 interpretations of the requirement:
[scenario_1]. There are 3 groups of bidders, each of n_bidders // 3 size. 
    a. primal dual truthful bidders
    b. primal dual non-truthful bidders
    c. UCB bidders
[scenario_2]. There are 3 bidders of types a, b, c and n-3 stochastic bidders
[scenario_3]. There are 3 bidders of types a, b, c and n-3 adversarial bidders
"""
class Requirement:
    def __init__(self, args, n_iters):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        if self.ctrs is None:
            self.ctrs = np.random.uniform(0.4, 0.9, self.num_participants)
        else:
            assert len(self.ctrs) == self.num_participants, "Number of CTRs must match number of bidders"

        self.T_bidding = self.num_auctions

    def main(self):
        # report = PDFReport("prova.pdf", 4)

        num_participants = self.num_participants
        if num_participants % 3 != 0:
            warnings.warn(f"Number of competitors must be divisible by 3, decreasing num_competitors to reach divisibility by 3: reaching number {num_participants - num_participants % 3}")
            num_participants -= num_participants % 3
            self.ctrs = self.ctrs[:num_participants]

        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)
        #custom eta for truthful bidders
        if self.eta is None:
            eta = 1/np.sqrt(self.T_bidding) 
        else:
            eta = self.eta

        #also available bids changes depending on what valuation the bidder has
        available_bids = np.zeros(shape = (num_participants, K))
        
        idx_trut = range(0, num_participants // 3) #truthful bidders
        idx_non_trut = range(num_participants // 3, 2 * num_participants // 3) #non-truthful bidders
        idx_ucb = range(2 * num_participants // 3, num_participants) #ucb bidders


        ''' LOGGING METRICS '''
        # Initialize lists to store total utilities, payments, and wins for each type of bidder
        total_utilities_distribution = {0: [], 1: [], 2: []}
        total_payments_distribution = {0: [], 1: [], 2: []}
        total_wins_distribution = {0: [], 1: [], 2: []}
        regret_per_trial_bidding_nont = []
        regret_per_trial_bidding_t = []
        regret_per_trial_bidding_ucb = []

        for seed in range(self.n_iters):
            np.random.seed(self.seed + seed)
            if self.valuation is None:
                valuations = np.random.uniform(0.7, 0.8, num_participants)
            else:
                valuations = np.ones(num_participants) * self.valuation #same valuation for all bidders

            if self.ctrs is None:
                self.ctrs = np.random.uniform(0.4, 0.9, self.num_participants)

            bidders = []
            for i in idx_trut:
                bidders.append(ag.StochasticPacingAgent(valuations[i], self.budget, self.T_bidding, eta))
            for j in idx_non_trut:
                bidders.append(ag.AdversarialPacingAgent(available_bids[j], valuations[j], self.budget, self.T_bidding, eta))
            for z in idx_ucb:
                bidders.append(ag.UCB1BiddingAgent(self.budget, bids=available_bids[z], T = self.T_bidding, range=1))

            auction = au.FirstPriceAuction(self.ctrs)

            min_bid = 0
            max_bids = valuations - 0.05#max bis are the valuation of the bidders minus an epsilon

            for i in range(num_participants):
                available_bids[i] = np.linspace(min_bid, max_bids[i], K)

            ''' LOGGING '''
            all_bids = np.zeros((num_participants, self.T_bidding))
            m_ts = np.zeros((num_participants, self.T_bidding))
            my_utilities = np.zeros(shape = (num_participants, self.T_bidding))
            total_wins_types = np.zeros(3)
            total_utility_types = np.zeros(3)
            total_spent_types = np.zeros(3)            

            for t in range(self.T_bidding):

                all_bids_t = np.zeros(num_participants)
                for i, bidder in enumerate(bidders):
                    if i < num_participants // 3: #truthful bidders: I get the bid to the closest of the available bids otherwise impossible to construct regret 
                        bid = bidder.bid()
                        all_bids_t[i] = available_bids[i][np.abs(available_bids[i] - bid).argmin()]
                    else:
                        all_bids_t[i] = bidder.bid()
                all_bids[:, t] = all_bids_t

                sorted_bids_idx = np.argsort(all_bids_t)
                m_t_1 = all_bids_t[sorted_bids_idx[-1]]
                m_t_2 = all_bids_t[sorted_bids_idx[-2]] #this allows me to get the best opponent bid if a bidder j is the winner



                # get winner and payments
                winner, _ = auction.round(all_bids_t)


                for i, agent in enumerate(bidders):
                    #m_t is actually max bid of the opponents
                    if all_bids_t[i] == m_t_1:
                        m_t = copy.deepcopy(m_t_2)
                    else:
                        m_t = copy.deepcopy(m_t_1)
                    

                    has_won = (winner == i)
                    f_t = (valuations[i] - all_bids_t[i]) * has_won
                    c_t = all_bids_t[i] * has_won #you pay the bid only if you win
                    agent.update(f_t, c_t, m_t)
                    
                    total_wins_types[i//(num_participants//3)] += has_won
                    total_utility_types[i//(num_participants//3)] += f_t
                    total_spent_types[i//(num_participants//3)] += c_t

                    m_ts[i, t] = m_t
                    my_utilities[i, t] = f_t

            ''' LOGGING METRICS '''
            # Store total utilities, payments, and wins for this iteration
            for i in range(3):
                total_utilities_distribution[i].append(total_utility_types[i]) 
                total_payments_distribution[i].append(total_spent_types[i]) 
                total_wins_distribution[i].append(total_wins_types[i] / (self.T_bidding))
                            

            print("\n\nFinal results: \n")
            print(f"Total wins for truthful bidders: {total_wins_types[0]}, Total utility: {total_utility_types[0]}, Total spent on average: {total_spent_types[0]/(num_participants//3)}")
            print(f"Total wins for non-truthful bidders: {total_wins_types[1]}, Total utility: {total_utility_types[1]}, Total spent on average: {total_spent_types[1]/(num_participants//3)}")
            print(f"Total wins for UCB bidders: {total_wins_types[2]}, Total utility: {total_utility_types[2]}, Total spent on average: {total_spent_types[2]/(num_participants//3)}")         


            ''' ADVERSARIAL CLAIRVOYANT '''
            clairvoyant_utilities = np.zeros((num_participants, self.T_bidding))
            for i in range(num_participants):
                _, clairvoyant_utilities[i], _ = get_clairvoyant_non_truthful_adversarial(self.budget, valuations[i], self.T_bidding, available_bids[i], all_bids, auction_agent=auction, idx_agent=i)
            
            #now average the utilities for each of the 3 types of bidders
            clairvoyant_utilities_types = np.zeros((3, self.T_bidding))
            my_utilities_types = np.zeros((3, self.T_bidding))
            for i in range(self.T_bidding):
                #for each time t I shoulde have the average utility of the 3 types of bidders
                clairvoyant_utilities_types[0, i] = np.mean(clairvoyant_utilities[idx_trut, i])
                clairvoyant_utilities_types[1, i] = np.mean(clairvoyant_utilities[idx_non_trut, i])
                clairvoyant_utilities_types[2, i] = np.mean(clairvoyant_utilities[idx_ucb, i])
                #same for my_utilities
                my_utilities_types[0, i] = np.mean(my_utilities[idx_trut, i])
                my_utilities_types[1, i] = np.mean(my_utilities[idx_non_trut, i])
                my_utilities_types[2, i] = np.mean(my_utilities[idx_ucb, i])
            
            #now I can compute the regret for each type of bidder at each iteration
            regret_per_trial_bidding_t.append(np.cumsum(clairvoyant_utilities_types[0] - my_utilities_types[0]))
            regret_per_trial_bidding_nont.append(np.cumsum(clairvoyant_utilities_types[1] - my_utilities_types[1]))
            regret_per_trial_bidding_ucb.append(np.cumsum(clairvoyant_utilities_types[2] - my_utilities_types[2]))




        # Define a dictionary to map bidder types to their utilities
        bidder_types = {
            "Truthful Bidders": 0,
            "Non-Truthful Bidders": 1,
            "UCB Bidders": 2
        }

        # PLOTTING REGRET FOR ALL BIDDERS
        plt.figure()
        for regret, label in zip([regret_per_trial_bidding_t, regret_per_trial_bidding_nont, regret_per_trial_bidding_ucb], bidder_types.keys()):
            regret = np.array(regret)
            average_regret_bidding = np.mean(regret, axis=0)
            regret_sd_bidding = np.std(regret, axis=0)
            plt.plot(np.arange(self.T_bidding), average_regret_bidding, label=label)
            plt.fill_between(np.arange(self.T_bidding), average_regret_bidding - regret_sd_bidding/np.sqrt(self.n_iters), average_regret_bidding + regret_sd_bidding/np.sqrt(self.n_iters), alpha=0.3)
        plt.title("Regret over time")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend()
        # plt.savefig("regret_all_bidders.png")
        plt.show()

        # PLOTTING UTILITIES FOR ALL BIDDERS
        plt.figure()
        for label, index in bidder_types.items():
            plt.plot(np.arange(self.T_bidding), my_utilities_types[index], label=label)
        plt.title("Utilities over time")
        plt.xlabel("Time")
        plt.ylabel("Utility")
        plt.legend()
        # plt.savefig("utilities_all_bidders.png")
        plt.show()

        # PLOTTING CLAIRVOYANT UTILITIES FOR ALL BIDDERS
        plt.figure()
        for label, index in bidder_types.items():
            plt.plot(np.arange(self.T_bidding), clairvoyant_utilities_types[index], label=label)
        plt.title("Clairvoyant utilities over time")
        plt.xlabel("Time")
        plt.ylabel("Utility")
        plt.legend()
        # plt.savefig("clairvoyant_utilities_all_bidders.png")
        plt.show()

        # PLOTTING DISTRIBUTIONS FOR TOTAL UTILITIES
        plt.figure()
        for label, index in bidder_types.items():
            plt.hist(total_utilities_distribution[index], bins=20, alpha=0.5, label=label, density=True)
        plt.title("Distribution of Total Utilities")
        plt.xlabel("Total Utility")
        plt.ylabel("Density")
        plt.legend()
        # plt.savefig("distribution_total_utilities.png")
        plt.show()

        # PLOTTING DISTRIBUTIONS FOR TOTAL PAYMENTS
        plt.figure()
        for label, index in bidder_types.items():
            plt.hist(total_payments_distribution[index], bins=20, alpha=0.5, label=label, density=True)
        plt.title("Distribution of Total Payments")
        plt.xlabel("Total Payment")
        plt.ylabel("Density")
        plt.legend()
        # plt.savefig("distribution_total_payments.png")
        plt.show()

        # PLOTTING DISTRIBUTIONS FOR PERCENTAGE OF WINS
        plt.figure()
        for label, index in bidder_types.items():
            plt.hist(total_wins_distribution[index], bins=20, alpha=0.5, label=label, density=True)
        plt.title("Distribution of Percentage of Wins")
        plt.xlabel("Percentage of Wins")
        plt.ylabel("Density")
        plt.legend()
        # plt.savefig("distribution_percentage_wins.png")        
        plt.show()



    def adversarial(self):
        num_participants = self.num_participants
        num_competitors = num_participants - 3

        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)
        #custom eta for truthful bidders
        if self.eta is None:
            eta = 1/np.sqrt(self.T_bidding) 
        else:
            eta = self.eta

        if self.valuation is None:
            self.valuation = 0.8 #TO BE CHANGED


        min_bid = 0
        max_bid = self.valuation - 0.05
        available_bids = np.linspace(min_bid, max_bid, K)
        total_auctions = self.T_bidding

        '''LOGGING METRICS'''
        # Initialize lists to store the utilities, payments, and wins for each bidder type across iterations
        utilities_distribution_t = []
        utilities_distribution_nont = []
        utilities_distribution_ucb = []

        payments_distribution_t = []
        payments_distribution_nont = []
        payments_distribution_ucb = []

        wins_distribution_t = []
        wins_distribution_nont = []
        wins_distribution_ucb = []        
                
        utilities_per_iteration_t = []
        utilities_per_iteration_nont = []
        utilities_per_iteration_ucb = []
        clairvoyant_utilities_per_iteration = np.zeros((3, self.T_bidding))

        regret_per_trial_bidding_nont = []
        regret_per_trial_bidding_t = []
        regret_per_trial_bidding_ucb = []

        #ITERATIONS
        for seed in range(self.n_iters):
            np.random.seed(seed)

            self.ctrs = np.random.uniform(0.4, 0.9, num_competitors+3)
            if self.my_ctrs is not None:
                self.ctrs[0] = self.my_ctrs[0]
                self.ctrs[1] = self.my_ctrs[1]
                self.ctrs[2] = self.my_ctrs[2]
            
            

            other_bids = np.random.uniform(0.2, 0.80, size=(num_competitors, total_auctions))# matrix of bids for each competitor in each auction

            #the 3 agents: truthful, non-truthful, ucb
            agent_truthful = ag.StochasticPacingAgent(self.valuation, self.budget, total_auctions, eta)
            agent_non_truthful = ag.AdversarialPacingAgent(available_bids, self.valuation, self.budget, total_auctions, eta)
            agent_ucb = ag.UCB1BiddingAgent(self.budget, bids=available_bids, T = total_auctions, range=1)

            envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, total_auctions)
            auction = au.FirstPriceAuction(self.ctrs)

            ''' LOGGING BIDDING '''
            total_wins_types = np.zeros(3)
            total_utility_types = np.zeros(3)
            total_spent_types = np.zeros(3) 
            all_bids = np.ndarray((num_participants, total_auctions))
            m_ts = np.zeros((3, self.T_bidding))
            my_utilities = np.zeros(shape = (3, self.T_bidding))
            my_bids = np.zeros(shape = (3, self.T_bidding))
            my_payments = np.zeros(shape = (3, self.T_bidding))


     

            for t in range(total_auctions):
                # agent chooses bid
                bid_t = np.zeros(3)
                bid_t_t = agent_truthful.bid()
                bid_t_nont = agent_non_truthful.bid()
                bid_t_ucb = agent_ucb.bid()   
                bid_t[0] = available_bids[np.abs(available_bids - bid_t_t).argmin()]
                # bid_t[0] = bid_t_t
                bid_t[1] = bid_t_nont
                bid_t[2] = bid_t_ucb
                # get bids from other competitors
                other_bids_t = envir.round()
                # m_t = other_bids_t.max()

                # get winner and payments
                bids = np.append(bid_t, other_bids_t)
                all_bids[:, t] = bids
                sorted_bids_idx = np.argsort(bids)
                m_t_1 = bids[sorted_bids_idx[-1]]
                m_t_2 = bids[sorted_bids_idx[-2]] #this allows me to get the best opponent bid if a bidder j is the winner

                winner, _ = auction.round(bids)
                
                if winner == 0:
                    m_t = copy.deepcopy(m_t_2)
                    total_wins_types[0] += 1
                    f_t = (self.valuation - bid_t[0])
                    c_t = bid_t[0]

                    #logging
                    m_ts[0, t] = m_t
                    my_utilities[0, t] = f_t
                    my_bids[0, t] = bid_t[0]
                    my_payments[0, t] = c_t
                    total_utility_types[0] += f_t
                    total_spent_types[0] += c_t
                    total_wins_types[0] += 1

                elif winner == 1:
                    m_t = copy.deepcopy(m_t_2)
                    total_wins_types[1] += 1
                    f_t = (self.valuation - bid_t[1])
                    c_t = bid_t[1]

                    #logging
                    m_ts[1, t] = m_t
                    my_utilities[1, t] = f_t
                    my_bids[1, t] = bid_t[1]
                    my_payments[1, t] = c_t
                    total_utility_types[1] += f_t
                    total_spent_types[1] += c_t
                    total_wins_types[1] += 1
                
                elif winner == 2:
                    m_t = copy.deepcopy(m_t_2)
                    total_wins_types[2] += 1
                    f_t = (self.valuation - bid_t[2])
                    c_t = bid_t[2]

                    #logging
                    m_ts[2, t] = m_t
                    my_utilities[2, t] = f_t
                    my_bids[2, t] = bid_t[2]
                    my_payments[2, t] = c_t
                    total_utility_types[2] += f_t
                    total_spent_types[2] += c_t
                    total_wins_types[2] += 1
                
                else:
                    m_t = copy.deepcopy(m_t_1)
                    f_t = 0
                    c_t = 0

                    #logging
                    m_ts[2, t] = m_t
                    my_utilities[2, t] = f_t
                    my_bids[2, t] = bid_t[2]
                    my_payments[2, t] = c_t
                
                agent_ucb.update(f_t, c_t, m_t)
                agent_non_truthful.update(f_t, c_t, m_t)
                agent_truthful.update(f_t, c_t, m_t)


                # print(f"Auction {t+1}: Bid: {bid_t:.2f}, Opponent bid {m_t:.2f}, Utility: {f_t:.2f}, Payment: {c_t:.2f}, Winner: {winner}")

            ''' LOGGING METRICS '''
            utilities_per_iteration_t.append(my_utilities[0])
            utilities_per_iteration_nont.append(my_utilities[1])
            utilities_per_iteration_ucb.append(my_utilities[2])

            # Accumulate utilities, payments, and wins for each bidder type
            utilities_distribution_t.append(total_utility_types[0])
            utilities_distribution_nont.append(total_utility_types[1])
            utilities_distribution_ucb.append(total_utility_types[2])

            payments_distribution_t.append(total_spent_types[0])
            payments_distribution_nont.append(total_spent_types[1])
            payments_distribution_ucb.append(total_spent_types[2])

            wins_distribution_t.append(total_wins_types[0])
            wins_distribution_nont.append(total_wins_types[1])
            wins_distribution_ucb.append(total_wins_types[2])            
            

            print("\n\nFinal results: \n")
            print("Iters: ", seed)
            print(f"Total wins for truthful bidders: {total_wins_types[0]}, Total utility: {total_utility_types[0]}, Total spent on average: {total_spent_types[0]}")
            print(f"Total wins for non-truthful bidders: {total_wins_types[1]}, Total utility: {total_utility_types[1]}, Total spent on average: {total_spent_types[1]}")
            print(f"Total wins for UCB bidders: {total_wins_types[2]}, Total utility: {total_utility_types[2]}, Total spent: {total_spent_types[2]}")

            ''' ADVERSARIAL CLAIRVOYANT '''
            clairvoyant_utilities = np.zeros((3, self.T_bidding))
            for i in range(3):
                except_other_agents = [j for j in range(3) if j != i] #we don't consider in the regret the contribution of the other 2 main agents 
                _, clairvoyant_utilities[i], _ = get_clairvoyant_non_truthful_adversarial(self.budget, self.valuation, self.T_bidding, available_bids, all_bids, auction_agent=auction, idx_agent=i, exclude_bidders=except_other_agents)

                clairvoyant_utilities_per_iteration[i] += clairvoyant_utilities[i]
            #regret for each type of bidder
            regret_per_trial_bidding_t.append(np.cumsum(clairvoyant_utilities[0] - my_utilities[0]))
            regret_per_trial_bidding_nont.append(np.cumsum(clairvoyant_utilities[1] - my_utilities[1]))
            regret_per_trial_bidding_ucb.append(np.cumsum(clairvoyant_utilities[2] - my_utilities[2]))


        mean_utilities_t = np.mean(utilities_per_iteration_t, axis=0)
        mean_utilities_nont = np.mean(utilities_per_iteration_nont, axis=0)
        mean_utilities_ucb = np.mean(utilities_per_iteration_ucb, axis=0)
        my_utilities = [mean_utilities_t, mean_utilities_nont, mean_utilities_ucb]

        clairvoyant_utilities_per_iteration /= self.n_iters
        # Define a dictionary to map bidder types to their utilities
        bidder_types = {
            "Truthful Bidders": 0,
            "Non-Truthful Bidders": 1,
            "UCB Bidders": 2
        }

        ''' PLOTTING REGRET FOR ALL BIDDERS '''
        plt.figure()
        for regret, label in zip([regret_per_trial_bidding_t, regret_per_trial_bidding_nont, regret_per_trial_bidding_ucb], bidder_types.keys()):
            regret = np.array(regret)
            average_regret_bidding = np.mean(regret, axis=0)
            regret_sd_bidding = np.std(regret, axis=0)
            plt.plot(np.arange(self.T_bidding), average_regret_bidding, label=label)
            plt.fill_between(np.arange(self.T_bidding), average_regret_bidding - regret_sd_bidding/np.sqrt(self.n_iters), average_regret_bidding + regret_sd_bidding/np.sqrt(self.n_iters), alpha=0.3)
        plt.title("Regret over time")
        plt.xlabel("Time")
        plt.ylabel("Regret")
        plt.legend()
        #create folder if it does not exist
        if not os.path.exists("req4_adv"):
            os.makedirs("req4_adv")        
        # plt.savefig("req4_adv/regret_all_bidders.png")
        plt.show()

        ''' PLOTTING UTILITIES FOR ALL BIDDERS '''
        plt.figure()
        for label, index in bidder_types.items():
            plt.plot(np.arange(self.T_bidding), my_utilities[index], label=label)
        plt.title("Utilities over time")
        plt.xlabel("Time")
        plt.ylabel("Utility")
        plt.legend()
        # plt.savefig("req4_adv/utilities_all_bidders.png")
        plt.show()

        ''' PLOTTING CLAIRVOYANT UTILITIES FOR ALL BIDDERS '''  
        plt.figure()
        for label, index in bidder_types.items():
            plt.plot(np.arange(self.T_bidding), clairvoyant_utilities_per_iteration[index], label=label)
        plt.title("Clairvoyant utilities over time")
        plt.xlabel("Time")
        plt.ylabel("Utility")
        plt.legend()
        # plt.savefig("req4_adv/clairvoyant_utilities_all_bidders.png")
        plt.show()

        '''PLOTTING DISTRIBUTIONS FOR UTILITIES'''
        plt.figure()
        plt.hist(utilities_distribution_t, bins=20, alpha=0.5, label="Truthful Bidders", density=True)
        plt.hist(utilities_distribution_nont, bins=20, alpha=0.5, label="Non-Truthful Bidders", density=True)
        plt.hist(utilities_distribution_ucb, bins=20, alpha=0.5, label="UCB Bidders", density=True)
        plt.title("Distribution of Utilities")
        plt.xlabel("Utility")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

        '''PLOTTING DISTRIBUTIONS FOR PAYMENTS'''
        plt.figure()
        plt.hist(payments_distribution_t, bins=20, alpha=0.5, label="Truthful Bidders", density=True)
        plt.hist(payments_distribution_nont, bins=20, alpha=0.5, label="Non-Truthful Bidders", density=True)
        plt.hist(payments_distribution_ucb, bins=20, alpha=0.5, label="UCB Bidders", density=True)
        plt.title("Distribution of Payments")
        plt.xlabel("Payment")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

        '''PLOTTING DISTRIBUTIONS FOR WINS'''
        plt.figure()
        plt.hist(wins_distribution_t, bins=20, alpha=0.5, label="Truthful Bidders", density=True)
        plt.hist(wins_distribution_nont, bins=20, alpha=0.5, label="Non-Truthful Bidders", density=True)
        plt.hist(wins_distribution_ucb, bins=20, alpha=0.5, label="UCB Bidders", density=True)
        plt.title("Distribution of Wins")
        plt.xlabel("Wins")
        plt.ylabel("Density")
        plt.legend()
        plt.show()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--valuation", dest="valuation", type=float, default=None)
    parser.add_argument("--num_auctions", dest="num_auctions", type=int, default = 1000)
    parser.add_argument("--budget", dest="budget", type=float, default=100)
    parser.add_argument("--my_ctrs", dest="my_ctrs", type=parse_list, default=None)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=20)
    parser.add_argument("--num_participants", dest="num_participants", type=int, default=9)
    parser.add_argument("--ctrs", dest = "ctrs", type=parse_list, default = None)
    parser.add_argument("--eta", dest="eta", type=float, default=None) #learning rate for truthful bidders (default is 1/sqrt(T), one might decrease it to improve competition)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--scenario", dest="scenario", type=str, choices=['solo', 'adversarial'], default='solo')

    args = parser.parse_args()    

    req = Requirement(args, 100)

    if args.scenario == 'solo':
        req.main()
    elif args.scenario == 'adversarial':
        req.adversarial()
    else:
        print("Invalid scenario")
        exit(1)