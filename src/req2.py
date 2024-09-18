import argparse
import numpy as np
import random

import agents as ag
import environments as envi
import auctions as au
from utils import *
#set seed in numpy

class Requirement:
    def __init__(self, args, n_iters):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        #pricing members
        self.T_pricing = self.num_days



        #bidding members
        self.auctions_per_day = [self.auctions_per_day for _ in range(self.num_days)] #since it is 1 slot auction, 1 bid equals 1 user 
        self.auctions_per_day = [int(i + np.random.uniform(-5, 5)) for i in self.auctions_per_day] #add noise 

        self.competitors_per_day = [100 for _ in range(self.num_days)]

        self.ctrs_from_args = False
        if self.ctrs is not None:
            self.ctrs_from_args = True
            assert len(self.ctrs) == self.num_competitors+1, "Number of CTRs must match number of bidders"
        else:
            self.ctrs = np.random.uniform(0.2, 0.8, self.num_competitors+1)
        if self.my_ctr is not None:
            self.ctrs[0] = self.my_ctr

        self.T_bidding = np.sum(self.auctions_per_day)

    def main(self):
        # report = PDFReport("prova.pdf", self.requirement)
        ''' PRICING SETUP '''
        item_cost = 0.1
        min_price = 0
        max_price = 1

        # a round of pricing for each day
        T_pricing = self.num_days
        eps_pricing = T_pricing ** (-1 / 3)
        K_pricing = int(1/eps_pricing + 1)
        if K_pricing % 2 == 0:
            K_pricing += 1 # this ensures K is odd
        discr_prices = np.linspace(min_price, max_price, K_pricing)
        # parametric conversion probability        
        conversion_probability = lambda p, theta: (1 - p) ** theta
        eta_pricing = np.sqrt(np.log(K_pricing) / T_pricing)

        ''' BIDDING SETUP '''
        num_competitors = self.num_competitors
        budget = self.budget
        # discretization step from theory
        T_bidding = self.T_bidding
        eps_bidding = T_bidding ** (-1 / 3)
        K_bidding = int(1/eps_bidding + 1)
        min_bid = 0
        max_bid = self.my_valuation - 0.001
        available_bids = np.linspace(min_bid, max_bid, K_bidding)
        eta_bidding = 1 / np.sqrt(T_bidding)
        my_valuation = self.my_valuation

        regret_per_trial_pricing = []
        regret_per_trial_bidding = []
        for seed in range(self.n_iters):
            np.random.seed(seed)
            
            ''' PRICING AGENT AND ENVIRONMENT SETUP'''
            pricing_agent = ag.HedgeAgent(K_pricing, eta_pricing)
            
            theta_seq = generate_adv_sequence(T_pricing, 0.5, 2)
            pricing_envir = envi.AdversarialPricingFullEnvironment(conversion_probability, theta_seq, item_cost)

            ''' BIDDING AGENT AND ENVIRONMENT SETUP'''
            if not self.ctrs_from_args:
                self.ctrs = np.random.uniform(0.4, 0.9, num_competitors+1)
            if self.my_ctr is not None:
                self.ctrs[0] = self.my_ctr

            my_ctr = self.ctrs[0]

            other_bids = np.random.uniform(0.4, 0.7, size=(num_competitors, T_bidding))
            bidding_envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, T_bidding)
            bidding_agent = ag.AdversarialPacingAgent(available_bids, my_valuation, budget, T_bidding, eta_bidding)
            auction = au.FirstPriceAuction(self.ctrs)

            ''' LOGGING PRICING '''
            my_prices = np.array([])
            my_sales = np.array([])
            my_rewards = np.array([])
            num_buyers = np.array([])
            total_sales = 0
            total_profit = 0

            ''' LOGGING BIDDING '''
            total_wins = 0
            total_utility = 0
            total_spent = 0
            all_bids = np.ndarray((num_competitors+1, self.T_bidding))
            m_ts = np.array([])
            my_utilities = np.array([])
            my_bids = np.array([])
            my_payments = np.array([])
            total_clicks = 0

            tot_auctions_counter = 0
            for t in range(self.num_days):
                ### Pricing phase: setting the price
                arm_t = pricing_agent.pull_arm()
                price_t = discr_prices[arm_t]

                day_wins = 0
                n_clicks = 0
                ### Bidding phase: each auction is a user connecting to the site
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

                                 
                    f_t = (my_valuation - payment_per_click) * my_win
                    c_t = payment_per_click * my_win
                    bidding_agent.update(f_t, c_t, m_t)

                    total_utility += f_t
                    total_spent += c_t

                    ''' LOGGING BIDDING '''
                    all_bids[:, tot_auctions_counter] = bids
                    my_utilities = np.append(my_utilities, f_t)
                    my_bids = np.append(my_bids, bid_t)
                    my_payments = np.append(my_payments, c_t)
                    m_ts = np.append(m_ts, m_t)
                    tot_auctions_counter += 1

                ### Pricing phase: updating the price
                # get full feedback from environment
                d_t, r_t = pricing_envir.round(discr_prices, n_clicks)
                # compute losses with normalized reward
                losses_t = 1 - r_t/n_clicks if n_clicks > 0 else np.ones(K_pricing)
                # update pricing agent with full feedback
                pricing_agent.update(losses_t)

                # update sales and profit on the played price
                day_sales = d_t[arm_t]
                day_profit = r_t[arm_t]

                ''' LOGGING PRICING '''
                total_sales += day_sales
                total_profit += day_profit

                my_prices = np.append(my_prices, price_t)
                my_sales = np.append(my_sales, day_sales)
                my_rewards = np.append(my_rewards, day_profit)  
                num_buyers = np.append(num_buyers, n_clicks)              

                ''' LOGGING BIDDING '''
                total_wins += day_wins
                total_clicks += n_clicks

                # print(f"Day {t+1}: Price: {price_t}, Day wins: {day_wins}, N.clicks: {n_clicks}, Day Sales: {day_sales}, Day Profit: {day_profit}")

            print(f"Total wins: {total_wins:.2f}, Total utility: {total_utility:.2f}, Total spent: {total_spent:.2f}, Total sales: {total_sales:.2f}, Total profit: {total_profit:.2f}")

            
            ''' PRICING CLAIRVOYANT '''
            expected_clairvoyant_rewards, _ = get_clairvoyant_pricing_adversarial(my_prices, my_rewards, discr_prices, T_pricing, pricing_envir, num_buyers)

            regret_per_trial_pricing.append(np.cumsum(expected_clairvoyant_rewards - my_rewards))

            ''' BIDDING CLAIRVOYANT '''
            clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments = get_clairvoyant_non_truthful_adversarial(budget, my_valuation, self.T_bidding, available_bids, all_bids, auction_agent=auction, idx_agent=0)

            regret_per_trial_bidding.append(np.cumsum(clairvoyant_utilities - my_utilities))
        
        '''PLOT REGRET PRICING'''
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
        plt.savefig('pricing_regret.png')

        ''' PLOT REGRET BIDDING '''
        regret_per_trial_bidding = np.array(regret_per_trial_bidding)
        average_regret_bidding = regret_per_trial_bidding.mean(axis=0)
        regret_sd_bidding = regret_per_trial_bidding.std(axis=0)

        plt.figure()
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
        plt.savefig('bidding_regret.png')

    def bidding(self):
        num_competitors = self.num_competitors
        budget = self.budget 
        


        eps = self.T_bidding**(-1/3)
        K = int(1/eps + 1)

        min_bid = 0
        max_bid = self.my_valuation - 0.001
        available_bids = np.linspace(min_bid, max_bid, K)

        # in this case we are just considering bidding so no need to separate for the different days.
        n_auctions = sum(self.auctions_per_day)

        # learning rate from theory
        eta = 1/np.sqrt(n_auctions)
        
        
        #In this case we are just considering bidding so no need to separete for the different days.
        total_auctions = sum(self.auctions_per_day)

        regret_per_trial_bidding = []
        for seed in range(self.n_iters):
            np.random.seed(seed)
            random.seed(seed)

            if not self.ctrs_from_args:
                self.ctrs = np.random.uniform(0.2, 0.8, num_competitors+1)
            if self.my_ctr is not None:
                self.ctrs[0] = self.my_ctr
            my_ctr = self.ctrs[0]
            other_bids = np.random.uniform(0.2, 0.8, size=(num_competitors, total_auctions))
            # matrix of bids for each competitor in each auction

            agent = ag.AdversarialPacingAgent(available_bids, self.my_valuation, budget, total_auctions, eta)
            envir = envi.AdversarialBiddingCompetitors(other_bids, num_competitors, total_auctions)
            auction = au.FirstPriceAuction(self.ctrs)

            ''' LOGGING BIDDING '''
            total_wins = 0
            total_utility = 0
            total_spent = 0
            all_bids = np.ndarray((num_competitors+1, total_auctions))
            m_ts = np.array([])
            my_utilities = np.array([])
            my_bids = np.array([])
            my_payments = np.array([])
            total_clicks = 0


            for t in range(total_auctions):
                # agent chooses bid
                bid_t = agent.bid()
                # get bids from other competitors
                other_bids_t = envir.round()
                m_t = other_bids_t.max()

                # get winner and payments
                bids = np.append(bid_t, other_bids_t)
                winner, payments_per_click = auction.round(bids)
                my_win = (winner == 0)

                f_t = (self.my_valuation - bid_t) * my_win
                c_t = bid_t * my_win
                # update agent with full feedback (m_t)
                agent.update(f_t, c_t, m_t)


                ''' LOGGING BIDDING '''
                all_bids[:, t] = bids
                my_utilities = np.append(my_utilities, f_t)
                my_bids = np.append(my_bids, bid_t)
                my_payments = np.append(my_payments, c_t)
                m_ts = np.append(m_ts, m_t)

                total_wins += my_win
                total_utility += f_t
                total_spent += c_t

                # print(f"Auction {t+1}: Bid: {bid_t:.2f}, Opponent bid {m_t:.2f}, Utility: {f_t:.2f}, Payment: {c_t:.2f}, Winner: {winner}")
            
            print(f"Total wins: {total_wins:.2f}, Total utility: {total_utility:.2f}, Total spent: {total_spent:.2f}")


            ''' BIDDING CLAIRVOYANT '''
            clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments = get_clairvoyant_non_truthful_adversarial(budget, self.my_valuation, self.T_bidding, available_bids, all_bids, auction_agent=auction, idx_agent=0)

            regret_per_trial_bidding.append(np.cumsum(clairvoyant_utilities - my_utilities))

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
        plt.savefig('just_bidding_regret.png')
        

        #now plot the bids in time
        plt.figure()        
        plt.plot(np.arange(self.T_bidding), my_bids, label='My bids')
        plt.plot(np.arange(self.T_bidding), m_ts, label='Opponent bids')
        plt.title('Bids in time')
        plt.xlabel('$t$')
        plt.legend()
        plt.savefig('bids_in_time.png')


        plot_agent_bidding(self.budget, my_bids, my_utilities, my_payments)

    def pricing(self):
        
        num_buyers = self.num_buyers
        item_cost = 0.1
        min_price = item_cost
        max_price = 1

        eps = self.T_pricing**(-1/3)
        K = int(1/eps + 1)
        
        discr_prices = np.linspace(min_price, max_price, K)

        learning_rate = np.sqrt(np.log(K) / self.T_pricing)
        eta = np.sqrt(np.log(K) / self.T_pricing)
        T = self.T_pricing

        regret_per_trial = []
        for seed in range(self.n_iters):
            np.random.seed(seed)
            random.seed(seed)
            #initialize agent
            agent = ag.HedgeAgent(K, learning_rate)

            conversion_probability = lambda p, theta: (1 - p)**theta

            theta_seq = generate_adv_sequence(self.T_pricing, 0.5, 2)
            envir = envi.AdversarialPricingFullEnvironment(conversion_probability, theta_seq, item_cost)

            ''' LOGGING PRICING '''
            my_prices = np.array([])
            my_sales = np.array([])
            my_rewards = np.array([])
            total_sales = 0
            total_profit = 0

            total_sales = 0
            total_profit = 0
            for t in range(self.T_pricing):
                #pull arm
                arm_t = agent.pull_arm()
                #get price
                price_t = discr_prices[arm_t]
                
                losses_t = np.array([])
                #full-feedback: need feedback on all prices
                d_t, r_t = envir.round(discr_prices, num_buyers)
                # compute losses with normalized reward
                losses_t = 1 - r_t/num_buyers

                # update done only for the played price
                total_sales += d_t[arm_t]
                total_profit += r_t[arm_t]

                #update agent with full feedback
                agent.update(losses_t)
                day_sales = d_t[arm_t]
                day_profit = r_t[arm_t]

                ''' LOGGING PRICING '''
                total_sales += day_sales
                total_profit += day_profit

                my_prices = np.append(my_prices, price_t)
                my_sales = np.append(my_sales, day_sales)
                my_rewards = np.append(my_rewards, day_profit)                     

                # print(f"Day {t+1}: Price: {price_t:.2f}, Losses: {np.round(losses_t, 2)}, Theta: {np.round(theta_seq[t], 2)}, Demand: {np.round(d_t, 2)}, Profit: {np.round(r_t, 2)}")
            
            print(f"Iteration: {seed}, Total Sales: {total_sales:.2f}, Total Profit: {total_profit:.2f}")

            ''' PRICING CLAIRVOYANT '''
            expected_clairvoyant_rewards, _ = get_clairvoyant_pricing_adversarial(my_prices, my_rewards, discr_prices, T, envir, num_buyers)
            regret_per_trial.append(np.cumsum(expected_clairvoyant_rewards - my_rewards))
            
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
        plt.savefig('just_pricing_regret.png')

        # plot_demand_curve(discr_prices, conversion_probability, num_buyers)

        # plot_profit_curve(discr_prices, conversion_probability, num_buyers, item_cost)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_days", dest="num_days", type=int, default=365)
    parser.add_argument("--auctions_per_day", dest="auctions_per_day", type=int, default = 10)
    parser.add_argument("--budget", dest="budget", type=float, default=100)
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=100)
    parser.add_argument("--num_competitors", dest="num_competitors", type=int, default=10)
    parser.add_argument("--ctrs", dest = "ctrs", type=list, default = None)
    parser.add_argument("--my_ctr", dest = "my_ctr", type=float, default = None)
    parser.add_argument("--my_valuation", dest = "my_valuation", type=float, default = 0.8)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    parser.add_argument("--run_type", dest="run_type", type=str, choices=['main', 'bidding', 'pricing'], default='main')

    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)

    args = parser.parse_args()    

    req = Requirement(args, 100)

    if args.run_type == 'main':
        req.main()
    elif args.run_type == 'bidding':
        req.bidding()
    elif args.run_type == 'pricing':
        req.pricing()
    else:
        print("Invalid run type")
        exit(1)
    
