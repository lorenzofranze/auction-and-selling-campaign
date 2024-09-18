import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

import agents as ag
import environments as envi
import auctions as au
from utils import *

class Requirement1:
    def __init__(self, args, n_iters):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        #pricing members
        self.T_pricing = self.num_days

        #bidding members
        self.auctions_per_day = [self.auctions_per_day for _ in range(self.num_days)] 
        self.auctions_per_day = [int(n * np.random.uniform(0.8, 1.2)) for n in self.auctions_per_day] #add noise to the number of auctions

        if self.ctrs is None:
            self.ctrs = np.random.uniform(0.2, 0.7, self.num_competitors+1)
            assert len(self.ctrs) == self.num_competitors+1, "Number of CTRs must match number of bidders"

        self.T_bidding = np.sum(self.auctions_per_day)

    def main(self):
        
        ''' PRICING SETUP '''
        item_cost = self.item_cost
        min_price = 0
        max_price = 1

        # a round of pricing for each day
        # discretization step from theory
        eps_pricing = self.T_pricing**(-1/3)
        K_pricing = int(1/eps_pricing + 1)
        if K_pricing % 2 == 0:
            K_pricing += 1 # this ensures K is odd

        discr_prices = np.linspace(min_price, max_price, K_pricing)
        
        # parametric conversion probability
        theta = self.theta
        conversion_probability = lambda p: (1 - p/max_price) ** theta
        # such that the probability of conversion is 1 at price = 0 and 0 at price = max_price
        pricing_envir = envi.StochasticPricingEnvironment(conversion_probability, item_cost)

        ''' BIDDING SETUP '''
        num_competitors = self.num_competitors
        my_budget = self.budget

        min_bid = 0
        max_bid = 1

        # discretization step from theory
        eps_bidding = self.T_bidding ** (-1 / 3)
        K_bidding = int(1/eps_bidding + 1)

        available_bids = np.linspace(min_bid, max_bid, K_bidding)
        # learning rate from theory
        eta = 1 / np.sqrt(self.T_bidding)

        my_ctr = self.ctr
        self.ctrs[0] = my_ctr
        my_valuation = self.valuation

        other_bids = lambda n: np.random.beta(14.4, 9.6, n) * max_bid
        # choose alpha and beta based on the mean and standard deviation of the competitors bids
        
        bidding_envir = envi.StochasticBiddingCompetitors(other_bids, num_competitors)
        auction = au.SecondPriceAuction(self.ctrs)

        regret_per_trial_pricing = []
        regret_per_trial_bidding = []
        for seed in range(self.n_iters):
            np.random.seed(seed)

            # instantiate agents
            pricing_agent = ag.GPUCBAgent(self.T_pricing, K_pricing)

            if self.bidder_type == 'UCB':
                bidding_agent = ag.UCB1BiddingAgent(my_budget, available_bids, self.T_bidding)
            elif self.bidder_type == 'pacing':
                bidding_agent = ag.StochasticPacingAgent(my_valuation, my_budget, self.T_bidding, eta)
            else:
                print("Invalid bidder type")
                exit(1)

            total_sales = 0
            total_profit = 0
            total_wins = 0
            total_utility = 0
            total_spent = 0
            total_clicks = 0

            ''' LOGGING BIDDING '''
            my_utilities = np.array([])
            my_bids = np.array([])
            my_payments = np.array([])
            m_ts = np.array([])

            ''' LOGGING PRICING '''
            my_rewards = np.array([])

            for t in range(self.num_days):
                ### Pricing phase: choose the price at the start of the day
                price_t = pricing_agent.pull_arm()
                # rescale price from [0,1] to [min_price, max_price]
                price_t = denormalize_zero_one(price_t, min_price, max_price)

                day_wins = 0
                n_clicks = 0
                ### Bidding phase: each auction is a user connecting to the site where the ad slot is displayed
                for auction_index in range(self.auctions_per_day[t]):
                    
                    bid_t = bidding_agent.bid()
                    other_bids_t = bidding_envir.round()
                    m_t = other_bids_t.max()
                    bids = np.append(bid_t, other_bids_t)

                    winner, payment_per_click = auction.round(bids)

                    my_win = 0
                    if winner == 0: # auction won
                        my_win = 1
                        day_wins += 1

                        user_clicked = np.random.binomial(1, my_ctr)
                        n_clicks += user_clicked
                        # each click on the ad will result in a pricing round

                    # utility and cost for the bidding agent are computed             
                    f_t = (my_valuation - payment_per_click) * my_win
                    c_t = payment_per_click * my_win
                    bidding_agent.update(f_t, c_t)

                    total_utility += f_t
                    total_spent += c_t

                    ''' LOGGING BIDDING '''
                    my_utilities = np.append(my_utilities, f_t)
                    my_bids = np.append(my_bids, bid_t)
                    my_payments = np.append(my_payments, c_t)
                    m_ts = np.append(m_ts, m_t)

                ### Pricing phase: updating the price at the end of the day
                # get bandit feedback from environment
                
                # n_clicks = 50
                d_t, r_t = pricing_envir.round(price_t, n_clicks)
                max_reward = (max_price - item_cost) * n_clicks
                min_reward = (min_price - item_cost) * n_clicks

                # update agent with profit normalized to [0,1]
                # pricing_agent.update(normalize_zero_one(r_t, min_reward, max_reward))
                pricing_agent.update(r_t/n_clicks if n_clicks > 0 else 0)
                
                # update sales and profit on the played price
                day_sales = d_t
                day_profit = r_t

                ''' LOGGING PRICING '''
                my_rewards = np.append(my_rewards, r_t)

                total_wins += day_wins
                total_sales += day_sales
                total_profit += day_profit
                total_clicks += n_clicks

                # print(f"Day {t+1}: Price: {price_t}, Day wins: {day_wins}, N.clicks: {n_clicks}, Day Sales: {day_sales}, Day Profit: {day_profit}")

            print(f"Total wins: {total_wins}, Total utility: {total_utility}, Total spent: {total_spent}, Total sales: {total_sales}, Total profit: {total_profit}")

            # at the end of each trial we compute the clairvoyants

            ''' PRICING CLAIRVOYANT '''
            expected_profit_curve = total_clicks/self.num_days * conversion_probability(discr_prices) * (discr_prices-item_cost)
            best_price_index = np.argmax(expected_profit_curve)

            expected_clairvoyant_rewards = np.repeat(np.ceil(expected_profit_curve[best_price_index]), self.num_days)

            regret_per_trial_pricing.append(np.cumsum(expected_clairvoyant_rewards - my_rewards))

            ''' BIDDING CLAIRVOYANT '''
            clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments = get_clairvoyant_truthful_stochastic(my_budget, my_valuation, m_ts, self.T_bidding)

            regret_per_trial_bidding.append(np.cumsum(clairvoyant_utilities - my_utilities))
        
        ''' PLOT REGRET PRICING '''
        regret_per_trial_pricing = np.array(regret_per_trial_pricing)
        average_regret_pricing = regret_per_trial_pricing.mean(axis=0)
        regret_sd_pricing = regret_per_trial_pricing.std(axis=0)

        plt.plot(np.arange(self.num_days), average_regret_pricing, label='Average Regret Pricing')
        plt.title('Cumulative regret of pricing')
        plt.fill_between(np.arange(self.num_days),
                        average_regret_pricing-regret_sd_pricing/np.sqrt(self.n_iters),
                        average_regret_pricing+regret_sd_pricing/np.sqrt(self.n_iters),
                        alpha=0.3,
                        label='Uncertainty')
        #plt.plot((0,T-1), (average_regret_pricing[0], average_regret_pricing[-1]), 'ro', linestyle="--")
        plt.xlabel('$t$')
        plt.legend()
        plt.show()

        ''' PLOT REGRET BIDDING '''
        regret_per_trial_bidding = np.array(regret_per_trial_bidding)
        average_regret_bidding = regret_per_trial_bidding.mean(axis=0)
        regret_sd_bidding = regret_per_trial_bidding.std(axis=0)

        plt.plot(np.arange(self.T_bidding), average_regret_bidding, label='Average Regret Bidding')
        plt.title('Cumulative regret of bidding')
        plt.fill_between(np.arange(self.T_bidding),
                        average_regret_bidding-regret_sd_bidding/np.sqrt(self.n_iters),
                        average_regret_bidding+regret_sd_bidding/np.sqrt(self.n_iters),
                        alpha=0.3,
                        label='Uncertainty')
        #plt.plot((0,T-1), (average_regret_bidding[0], average_regret_bidding[-1]), 'ro', linestyle="--")
        plt.xlabel('$t$')
        plt.legend()
        plt.show()

    
    ''' ONLY BIDDING '''
    def bidding(self):

        num_competitors = self.num_competitors
        my_budget = self.budget

        min_bid = 0
        max_bid = 1

        # in this case we are just considering bidding so no need to separate for the different days
        # discretization step from theory
        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)

        available_bids = np.linspace(min_bid, max_bid, K)
        # learning rate from theory
        eta = 1 / np.sqrt(self.T_bidding)

        my_ctr = self.ctr
        self.ctrs[0] = my_ctr
        my_valuation = self.valuation
        
        other_bids = lambda n: np.random.beta(14.4, 9.6, n) * max_bid
        # choose alpha and beta based on the mean and standard deviation of the competitors bids
        
        envir = envi.StochasticBiddingCompetitors(other_bids, num_competitors)
        auction = au.SecondPriceAuction(self.ctrs)

        regret_per_trial = []
        bids_per_trial = []
        for seed in range(self.n_iters):
            np.random.seed(seed)

            if self.bidder_type == 'UCB':
                agent = ag.UCB1BiddingAgent(my_budget, available_bids, self.T_bidding)
            elif self.bidder_type == 'pacing':
                agent = ag.StochasticPacingAgent(my_valuation, my_budget, self.T_bidding, eta)
            else:
                print("Invalid bidder type")
                exit(1)

            total_wins = 0
            total_utility = 0
            total_spent = 0

            my_utilities = np.array([])
            my_bids = np.array([])
            my_payments = np.array([])
            m_ts = np.array([])

            for t in range(self.T_bidding):
                # agent chooses bid
                bid_t = agent.bid()
                # get bids from other competitors
                other_bids_t = envir.round()
                m_t = other_bids_t.max()
                bids = np.append(bid_t, other_bids_t)

                winner, payment_per_click = auction.round(bids)

                my_win = (winner == 0)

                f_t = (my_valuation - payment_per_click) * my_win
                c_t = payment_per_click * my_win
                # update agent
                agent.update(f_t, c_t)

                ''' LOGGING '''
                my_utilities = np.append(my_utilities, f_t)
                my_bids = np.append(my_bids, bid_t)
                my_payments = np.append(my_payments, c_t)
                m_ts = np.append(m_ts, m_t)

                total_wins += my_win
                total_utility += f_t
                total_spent += c_t

                # print(f"Auction: {t+1}, Bid: {bid_t}, Opponent bid: {m_t}, Utility: {f_t}, Payment: {c_t}, Winner: {winner}")

            print(f"Total wins: {total_wins}, Total utility: {total_utility}, Total spent: {total_spent}")
            ''' LOGGING BIDS '''
            bids_per_trial += my_bids.tolist()

            ''' CLAIRVOYANT '''
            clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments = get_clairvoyant_truthful_stochastic(my_budget, my_valuation, m_ts, self.T_bidding)

            cumulative_regret = np.cumsum(clairvoyant_utilities - my_utilities)
            regret_per_trial.append(cumulative_regret)

        ''' PLOT REGRET '''
        regret_per_trial = np.array(regret_per_trial)
        average_regret = regret_per_trial.mean(axis=0)
        regret_sd = regret_per_trial.std(axis=0)

        plt.plot(np.arange(self.T_bidding), average_regret, label='Average Regret')
        plt.title('Cumulative regret of bidder')
        plt.fill_between(np.arange(self.T_bidding),
                        average_regret-regret_sd/np.sqrt(self.n_iters),
                        average_regret+regret_sd/np.sqrt(self.n_iters),
                        alpha=0.3,
                        label='Uncertainty')
        #plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
        plt.xlabel('$t$')
        plt.legend()
        plt.show()

        plt.hist(bids_per_trial)
        plt.title('Bids distribution')
        plt.show()

        # plot_clayrvoyant_truthful(my_budget, clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments)

        ''' AGENT '''
        plot_agent_bidding(my_budget, my_bids, my_utilities, my_payments)

        ''' REGRET '''
        # plot_regret(my_utilities, clairvoyant_utilities)

    ''' ONLY PRICING '''
    def pricing(self):
        
        item_cost = self.item_cost
        min_price = 0 # anything lower than this would be a loss
        max_price = 1 # price at which the conversion probability is 0
        n_customers = 100

        K = 100

        discr_prices = np.linspace(min_price, max_price, K)

        theta = self.theta
        conversion_probability = lambda p: (1 - p/max_price)**theta
        # such that the probability of conversion is 1 at price = 0 and 0 at price = max_price

        reward_function = lambda price, n_sales: (price - item_cost) * n_sales

        # the maximum possible profit is selling to all customers at the maximum price for which the conversion probability is > 0
        max_reward = (max_price - item_cost) * n_customers
        min_reward = (min_price - item_cost) * n_customers
        
        ''' CLAIRVOYANT '''
        expected_profit_curve = n_customers * conversion_probability(discr_prices) * (discr_prices-item_cost)
        best_price_index = np.argmax(expected_profit_curve)

        expected_clairvoyant_rewards = np.repeat(expected_profit_curve[best_price_index], self.T_pricing)

        regret_per_trial = []
        for seed in range(self.n_iters):
            np.random.seed(seed)

            agent = ag.GPUCBAgent(T = self.T_pricing, discretization = K)
            envir = envi.StochasticPricingEnvironment(conversion_probability, item_cost)

            my_prices = np.array([])
            my_sales = np.array([])
            my_rewards = np.array([])
            total_sales = 0
            total_profit = 0

            for t in range(self.T_pricing): 
                # GP agent chooses price
                price_t = agent.pull_arm()
                # rescale price from [0,1] to [min_price, max_price]
                price_t = denormalize_zero_one(price_t, min_price, max_price)

                # get demand and reward from pricing environment
                d_t, r_t = envir.round(price_t, n_customers)
                # reward = total profit

                # update agent with profit normalized to [0,1]
                # agent.update(normalize_zero_one(r_t, min_reward, max_reward))
                agent.update(r_t/n_customers)

                ''' LOGGING '''
                my_prices = np.append(my_prices, price_t)
                my_sales = np.append(my_sales, d_t)
                my_rewards = np.append(my_rewards, r_t)

                total_sales += d_t
                total_profit += r_t

                # print(f"Day {t+1}: Price: {price_t}, Demand: {d_t}, Reward: {r_t}")

            print(f"Total Sales: {total_sales}, Total Profit: {total_profit}")

            cumulative_regret = np.cumsum(expected_clairvoyant_rewards - my_rewards)
            regret_per_trial.append(cumulative_regret)

        regret_per_trial = np.array(regret_per_trial)
        average_regret = regret_per_trial.mean(axis=0)
        regret_sd = regret_per_trial.std(axis=0)

        plt.plot(np.arange(self.T_pricing), average_regret, label='Average Regret')
        plt.title('Cumulative regret of GPUCB')
        plt.fill_between(np.arange(self.T_pricing),
                        average_regret-regret_sd/np.sqrt(self.n_iters),
                        average_regret+regret_sd/np.sqrt(self.n_iters),
                        alpha=0.3,
                        label='Uncertainty')
        #plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
        plt.xlabel('$t$')
        plt.legend()
        plt.show()

        plot_agent_pricing(my_prices, my_sales, my_rewards)

        # plot_regret(my_rewards, expected_clairvoyant_rewards)

        plot_demand_curve(discr_prices, conversion_probability, n_customers)

        plot_profit_curve(discr_prices, conversion_probability, n_customers, item_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=365)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default=10)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=5)
    parser.add_argument("--num_competitors", dest="num_competitors", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='main')
    parser.add_argument("--bidder_type", dest="bidder_type", type=str, choices=['UCB', 'pacing'], default='pacing')
    parser.add_argument("--budget", dest="budget", type=int, default=100)
    parser.add_argument("--valuation", dest="valuation", type=float, default=0.8)
    parser.add_argument("--ctr", dest="ctr", type=float, default=0.5)
    parser.add_argument("--theta", dest="theta", type=float, default=1)

    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)
    parser.add_argument("--item_cost", dest="item_cost", type = int, default = 0.2)

    args = parser.parse_args()    

    set_seed(args.seed)

    req = Requirement1(args, 100)

    if args.run_type == 'main':
        req.main()
    elif args.run_type == 'bidding':
        req.bidding()
    elif args.run_type == 'pricing':
        req.pricing()
    else:
        print("Invalid run type")
        exit(1)
    