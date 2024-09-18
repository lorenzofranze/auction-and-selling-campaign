import argparse
import numpy as np
import math

from agents import CUSUM_GP_UCBAgent, UCB1Agent, SWUCBAgent, CUSUMUCBAgent, ThompsonSamplingCUSUM

#from agents.CUSUMUCBAgent import CUSUMUCBAgent
#from agents.ThompsonSamplingCUSUM import ThompsonSamplingCUSUM

from environments import NonStationaryPricingEnvironment
from utils import *
import matplotlib.pyplot as plt


class Requirement3():

    def __init__(self, args):
        self.args = args
        #extract all args in members
        for key, value in vars(args).items():
            setattr(self, key, value)

        #set algorithm to run
        if self.learning_type == "all":
            self.learning_type = ["ucb", "ucb_sw", 'gp_ucb_cusum' , 'cusum_ucb_ts']

        #set seed in numpy
        np.random.seed(self.seed)

        #pricing parameters
        self.T_pricing = 50000
        self.T_interval = 10000
        self.intervals = math.ceil(self.T_pricing/self.T_interval)
        #defining demand curves using max operator to avoid negative values
        self.demand_functions = [
            lambda price: np.maximum(0, 1 - price / 30 ),
            lambda price: np.maximum(0, 1 - price / 60 ),
            lambda price: np.maximum(0, np.exp(- ((price-10) **2  )/25)) ,
            lambda price: np.maximum(0, (1 / (2 * np.sqrt(0.05 * price - 0.3))) - 0.5),
            lambda price: np.maximum(0, 1 - (2.4 * ((price - 10) / 30) 
                               - 2.8 * ((price- 10) / 30) ** 2 
                               + 1.4 * ((price- 10) / 30) ** 3))
        ]
        #defining pricing cost parameters
        self.cost = 10
        self.max_price = 40
        self.min_price = self.cost

        #using discretization prescribed by theory but for each interval
        epsilon = self.T_interval**(-0.33)
        self.K = int(1/epsilon)

        print("number of possible prices K:", self.K)

        self.prices = np.linspace(self.min_price, self.max_price, self.K)

    def show_demand_curves(self, demand_curves, profit_curves, best_prices_indices):

        best_prices = [self.prices[i] for i in best_prices_indices]
        fig, axs = plt.subplots(1, 2)
        for i in range(self.intervals):
            axs[0].plot(self.prices, demand_curves[i], label=f'interval {i}')
            axs[0].scatter(best_prices[i], demand_curves[i][best_prices_indices[i]], color='red')
            axs[1].plot(self.prices, profit_curves[i], label=f'interval {i}')
            axs[1].scatter(best_prices[i], profit_curves[i][best_prices_indices[i]], color='red')
        axs[0].set_title('Demand curves')
        axs[0].set_xlabel('Price')
        axs[0].set_ylabel('Demand')
        axs[0].legend()
        axs[1].set_title('Profit curves')
        axs[1].set_xlabel('Price')
        axs[1].set_ylabel('Profit')
        axs[1].legend()
        #display figures but go on with the code
        plt.show()

    def main(self):
        #####
        #show demand curves and profit curves
        #####
        demand_curves = [self.num_buyers * self.demand_functions[i](self.prices) for i in range(self.intervals)]
        profit_curves = [demand_curve * (self.prices - self.cost) for demand_curve in demand_curves]
        best_prices_indices = [np.argmax(profit_curve) for profit_curve in profit_curves]
        
        #show demand curves and profit curves
        self.show_demand_curves(demand_curves, profit_curves, best_prices_indices)
    

        #####
        #clairvoyant agent
        #####
        best_prices_each_interval = [self.prices[i] for i in best_prices_indices]
        clairvoyant_rewards_each_interval = np.array([profit_curves[i][best_prices_indices[i]] for i in range(self.intervals)])
        #repeat for each day
        best_prices = np.repeat(best_prices_each_interval, self.T_interval)
        clairvoyant_rewards = np.repeat(clairvoyant_rewards_each_interval, self.T_interval)
        #clairvoyant_rewards = normalize_zero_one(clairvoyant_rewards, self.min_price, self.max_price)

        maximum_profit = (max(self.prices) - self.cost) * self.num_buyers

        print("best prices for each interval:" , best_prices_each_interval)
        print("best profit for each interval:", clairvoyant_rewards_each_interval)
        print("maximum profit:", maximum_profit)

        colors = ['b', 'g', 'r', 'c', 'm']

        U_T = self.intervals-1
        h = 2*np.log(self.T_pricing/U_T) # sensitivity of detection, threshold for cumulative deviation
        M = int(np.log(self.T_pricing/U_T)) # robustness of change detection

        #####        naive UCB1 agent      #####

        if "ucb" in self.learning_type:

            print("---- UCB1 agent ----")
            #initialize agent
            regret_per_trial = []

            for ite in range(self.n_iters):
                print(f'iteration {ite}', end="\r")
                env = NonStationaryPricingEnvironment(cost = self.cost, 
                                                conversion_probabilities = self.demand_functions, 
                                                T_interval = self.T_interval, 
                                                seed = self.seed)
                #agent = UCB1Agent(self.K, self.T_pricing, range = maximum_profit)
                agent = UCB1Agent(self.K, self.T_pricing, range = maximum_profit)


                agent_rewards = np.array([])

                for t in range(self.T_pricing):
                    a_t = agent.pull_arm()
                    p_t = self.prices[a_t]
                    _ , r_t = env.round(p_t = p_t, n_t=self.num_buyers)
                    
                    agent.update(r_t)

                    agent_rewards = np.append(agent_rewards, r_t)

                cumulative_regret = np.cumsum(clairvoyant_rewards-agent_rewards)
                regret_per_trial.append(cumulative_regret)

            regret_per_trial = np.array(regret_per_trial)

            average_regret = regret_per_trial.mean(axis=0)

            #normalize average regret
            average_regret = average_regret / maximum_profit

            regret_sd = regret_per_trial.std(axis=0)


            plt.plot(np.arange(self.T_pricing), average_regret, label='Average Regret')
            plt.title('cumulative regret of UCB1')
            plt.fill_between(np.arange(self.T_pricing),
                            average_regret-regret_sd/np.sqrt(self.n_iters),
                            average_regret+regret_sd/np.sqrt(self.n_iters),
                            alpha=0.3,
                            label='Uncertainty')
            plt.xlabel('$t$')
            plt.legend()
            plt.show()

            plt.figure()
            plt.barh(self.prices, agent.N_pulls)
            for interval in range(self.intervals):
                plt.axhline(y=best_prices_each_interval[interval], color=colors[interval], linestyle='--', label=f'best price interval {interval}')

            plt.ylabel('actions')
            plt.xlabel('numer of pulls')
            plt.legend()
            plt.title('Number of pulls for each action')
            plt.show()

            print("UCB1 agent total regret: ", average_regret[-1])


        #####        SW-UCB agent  - sliding window    #####

        if "ucb_sw" in self.learning_type:
            
            print("---- Sliding Window UCB agent ----")

            U_T = self.intervals-1 # maximum number of abrupt changes
            W = int(2*np.sqrt(self.T_pricing*np.log(self.T_pricing)/U_T)) # assuming U_T is known
            print("window size: ", W)

            regret_per_trial = []

            for ite in range(self.n_iters):
                print(f'iteration {ite}', end="\r")
                env = NonStationaryPricingEnvironment(cost = self.cost, 
                                                conversion_probabilities = self.demand_functions, 
                                                T_interval = self.T_interval, 
                                                seed = self.seed)
                agent = SWUCBAgent(self.K, self.T_pricing, W, range = maximum_profit)

                agent_rewards = np.array([])

                for t in range(self.T_pricing):
                    a_t = agent.pull_arm()
                    p_t = self.prices[a_t]
                    _ , r_t = env.round(p_t = p_t, n_t=self.num_buyers)
                    agent.update(r_t)

                    agent_rewards = np.append(agent_rewards, r_t)

                cumulative_regret = np.cumsum(clairvoyant_rewards-agent_rewards)
                regret_per_trial.append(cumulative_regret)

            regret_per_trial = np.array(regret_per_trial)

            average_regret = regret_per_trial.mean(axis=0)
            #normalize average regret
            average_regret = average_regret / maximum_profit
            regret_sd = regret_per_trial.std(axis=0)


            plt.plot(np.arange(self.T_pricing), average_regret, label='Average Regret')
            plt.title('cumulative regret of UCB sliding window')
            plt.fill_between(np.arange(self.T_pricing),
                            average_regret-regret_sd/np.sqrt(self.n_iters),
                            average_regret+regret_sd/np.sqrt(self.n_iters),
                            alpha=0.3,
                            label='Uncertainty')
            plt.xlabel('$t$')
            plt.legend()
            plt.show()

            plt.figure()
            plt.barh(self.prices, agent.N_pulls)
            for interval in range(self.intervals):
                plt.axhline(y=best_prices_each_interval[interval], color=colors[interval], linestyle='--', label=f'best price interval {interval}')

            plt.ylabel('actions')
            plt.xlabel('numer of pulls')
            plt.title('Number of pulls for each action')
            plt.legend()
            plt.show()

            print("UCB - sliding window agent total regret: ", average_regret[-1])


                #####        naive UCB1 agent      #####

        if "gp_ucb_cusum" in self.learning_type:

            print("---- GP-UCB CUSUM agent ----")

            #initialize agent
            regret_per_trial = []

            for ite in range(1):
                env = NonStationaryPricingEnvironment(cost = self.cost, 
                                                conversion_probabilities = self.demand_functions, 
                                                T_interval = self.T_interval, 
                                                seed = self.seed)
                #agent = UCB1Agent(self.K, self.T_pricing, range = maximum_profit)
                agent = CUSUM_GP_UCBAgent(self.T_pricing, M = 2*M, h = 240, epsilon=130, K = self.K,  scale = maximum_profit)


                agent_rewards = np.array([])

                for t in range(self.T_pricing):
                    
                    p_t = agent.pull_arm()
                    p_t = denormalize_zero_one(p_t, self.min_price, self.max_price)
                    _ , r_t = env.round(p_t = p_t, n_t=self.num_buyers)

                    if(t%2000==0):
                        print(f'time: {t} - price: {p_t}')
                    
                    agent.update(r_t)

                    agent_rewards = np.append(agent_rewards, r_t)

                cumulative_regret = np.cumsum(clairvoyant_rewards-agent_rewards)
                regret_per_trial.append(cumulative_regret)

            print(agent.reset_times)
            print(agent.n_resets)

            regret_per_trial = np.array(regret_per_trial)

            average_regret = regret_per_trial.mean(axis=0)

            #normalize average regret
            average_regret = average_regret / maximum_profit

            regret_sd = regret_per_trial.std(axis=0)


            plt.plot(np.arange(self.T_pricing), average_regret, label='Average Regret')
            plt.title('cumulative regret of GP-UCB-CUSUM')
            plt.fill_between(np.arange(self.T_pricing),
                            average_regret-regret_sd/np.sqrt(self.n_iters),
                            average_regret+regret_sd/np.sqrt(self.n_iters),
                            alpha=0.3,
                            label='Uncertainty')
            plt.xlabel('$t$')
            plt.legend()
            plt.show()

            plt.figure()
            plt.barh(self.prices, agent.N_pulls)
            for interval in range(self.intervals):
                plt.axhline(y=best_prices_each_interval[interval], color=colors[interval], linestyle='--', label=f'best price interval {interval}')

            plt.ylabel('actions')
            plt.xlabel('numer of pulls')
            plt.legend()
            plt.title('Number of pulls for each action')
            plt.show()

            print("GP-UCB-CUSUM agent total regret: ", average_regret[-1])

        
        
        # CUSUM agents run both UCB and Thompson Sampling

        if 'cusum_ucb_ts' in self.learning_type: 

            #####        CUMSUM UCB agent      #####

            print("---- CUSUM UCB agent ----")

            U_T = self.intervals-1
            h = 2*np.log(self.T_pricing/U_T) # sensitivity of detection, threshold for cumulative deviation
            alpha = np.sqrt(U_T*np.log(self.T_pricing/U_T)/self.T_pricing) # probability of extra exploration
            M = int(np.log(self.T_pricing/U_T)) # robustness of change detection

            regret_per_trial = []

            for ite in range(self.n_iters):
                print(f'iteration {ite}', end="\r")
                env = NonStationaryPricingEnvironment(cost = self.cost, 
                                                conversion_probabilities = self.demand_functions, 
                                                T_interval = self.T_interval, 
                                                seed = self.seed)
                agent = CUSUMUCBAgent(self.K, self.T_pricing, M, h,alpha, range=maximum_profit)

                agent_rewards = np.array([])

                for t in range(self.T_pricing):
                    a_t = agent.pull_arm()
                    p_t = self.prices[a_t]
                    _ , r_t = env.round(p_t = p_t, n_t=self.num_buyers)
                    
                    agent.update(r_t)

                    agent_rewards = np.append(agent_rewards, r_t)

                cumulative_regret = np.cumsum(clairvoyant_rewards-agent_rewards)
                regret_per_trial.append(cumulative_regret)

            regret_per_trial = np.array(regret_per_trial)

            average_regret = regret_per_trial.mean(axis=0)
            #normalize average regret
            average_regret = average_regret / maximum_profit
            regret_sd = regret_per_trial.std(axis=0)


            plt.plot(np.arange(self.T_pricing), average_regret, label='Average Regret')
            plt.title('cumulative regret of UCB with CUSUM')
            plt.fill_between(np.arange(self.T_pricing),
                            average_regret-regret_sd/np.sqrt(self.n_iters),
                            average_regret+regret_sd/np.sqrt(self.n_iters),
                            alpha=0.3,
                            label='Uncertainty')
            plt.xlabel('$t$')
            plt.legend()
            plt.show()

            plt.figure()
            plt.barh(self.prices, agent.N_pulls)
            for interval in range(self.intervals):
                plt.axhline(y=best_prices_each_interval[interval], color=colors[interval], linestyle='--', label=f'best price interval {interval}')

            plt.ylabel('actions')
            plt.xlabel('numer of pulls')
            plt.title('Number of pulls for each action')
            plt.legend()
            plt.show()

            print("UCB - CUSUM agent total regret: ", average_regret[-1])

            #####      Thompson Sampling CUSUM agent      #####

            print("---- Thompson Sampling CUSUM agent ----")

            U_T = self.intervals-1
            h = 2*np.log(self.T_pricing/U_T) # sensitivity of detection, threshold for cumulative deviation
            M = int(np.log(self.T_pricing/U_T)) # robustness of change detection

            regret_per_trial_ts = []  # for thomposon samplig in order to distinguish from UCB before

            #executed just 2 times since it is slower
            for ite in range(2):
                print(f'iteration {ite}', end="\r")
                env = NonStationaryPricingEnvironment(cost = self.cost, 
                                                conversion_probabilities = self.demand_functions, 
                                                T_interval = self.T_interval, 
                                                seed = self.seed)
                agent = ThompsonSamplingCUSUM(self.K, self.T_pricing, M, h, range=maximum_profit)

                agent_rewards = np.array([])

                for t in range(self.T_pricing):
                    if(t%2000==0):
                        print(f'iteration: {ite} - time: {t} - price: {p_t}' , end="\r")
                    a_t = agent.pull_arm()
                    p_t = self.prices[a_t]
                    _ , r_t = env.round(p_t = p_t, n_t=self.num_buyers)
                    
                    agent.update(r_t)

                    agent_rewards = np.append(agent_rewards, r_t)

                cumulative_regret = np.cumsum(clairvoyant_rewards-agent_rewards)
                regret_per_trial_ts.append(cumulative_regret)

            regret_per_trial_ts = np.array(regret_per_trial_ts)

            average_regret_ts = regret_per_trial_ts.mean(axis=0)
            #normalize average regret
            average_regret_ts = average_regret_ts / maximum_profit


            plt.plot(np.arange(self.T_pricing), average_regret, label='UCB CUSUM') # for UCB CUSUM
            plt.plot(np.arange(self.T_pricing), average_regret_ts, label='Thomposon Sampling CUSUM') # for UCB CUSUM
            plt.title('cumulative regret of UCB with CUSUM vs Thompson Sampling with CUSUM')
            plt.xlabel('$t$')
            plt.legend()
            plt.show()


            plt.figure()
            plt.barh(self.prices, agent.N_pulls)
            for interval in range(self.intervals):
                plt.axhline(y=best_prices_each_interval[interval], color=colors[interval], linestyle='--', label=f'best price interval {interval}')

            plt.ylabel('actions')
            plt.xlabel('numer of pulls')
            plt.title('Number of pulls for each action')
            plt.legend()
            plt.show()

            plt.barh(self.prices, agent.n_resets)
            plt.title('number of resets per arm of CUSUM-TS')
            plt.ylabel('actions')
            plt.xlabel('number of resets')



            print("TS - CUSUM agent total regret: ", average_regret_ts[-1])




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", dest="n_iters", type = int, default=10)
    parser.add_argument("--seed", dest="seed", type=int, default=1)
    #for pricing only
    parser.add_argument("--num_buyers", dest="num_buyers", type = int, default = 100)
    parser.add_argument("--learning_type", dest="learning_type", type = str, choices=['ucb', 'ucb_sw', 'gp_ucb_cusum' , 'cusum_ucb_ts', "all"], default = "all")

    args = parser.parse_args()    

    req = Requirement3(args)

    req.main()
    