import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import copy
""" from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle """

def parse_list(argument):
    return [float(item) for item in argument.split(',')]


def set_seed(seed):
    np.random.seed(seed)
    return

def generate_adv_sequence(len, min, max):
    return_array = np.zeros(len)
    for i in range(len):
        return_array[i] = np.random.uniform(min, max)
    return return_array

def normalize_zero_one(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)

def denormalize_zero_one(x, min_x, max_x):
    return min_x + (max_x - min_x) * x

def get_clairvoyant_truthful_stochastic(budget, my_valuation, m_t, n_auctions):
    # the clairvoyant knows the max bid at each round 
    ## I compute my sequence of utilities at every round
    utility = (my_valuation-m_t)*(my_valuation>=m_t)
    # recall that operations with ndarray produce ndarray

    ## Now I have to find the sequence of m_t summing up to budget B and having the maximum sum of utility
    ## In second price auctions, I can find the sequence **greedily**:
    sorted_round_utility = np.flip(np.argsort(utility)) # sorted rounds, from most profitable to less profitable
    clairvoyant_utilities = np.zeros(n_auctions)
    clairvoyant_bids= np.zeros(n_auctions)
    clairvoyant_payments = np.zeros(n_auctions)
    c = 0 # total money spent
    i = 0 # index over the auctions
    while c <= budget-1 and i < n_auctions and utility[sorted_round_utility[i]] > 0:
        clairvoyant_bids[sorted_round_utility[i]] = 1 # bid 1 in the remaining most profitable auction
        # recall that since this is a second-price auction what I pay doesn't depend on my bid (but determines if I win)
        # notice that since the competitors' bids are fixed < 1 the clairvoyant can just bid 1 to the auctions he wants to win and 0 to the rest
        clairvoyant_utilities[sorted_round_utility[i]] = utility[sorted_round_utility[i]]
        clairvoyant_payments[sorted_round_utility[i]] = m_t[sorted_round_utility[i]]
        c += m_t[sorted_round_utility[i]]
        i+=1
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments

def get_clairvoyant_non_truthful_adversarial(budget, my_valuation, n_auctions, discr_bids, all_bids, auction_agent = None, idx_agent = 0, exclude_bidders = None):
    # the clairvoyant knows the max bid at each round
    
    clairvoyant_utilities = np.zeros(n_auctions)
    clairvoyant_bids= np.zeros(n_auctions)
    clairvoyant_payments = np.zeros(n_auctions)

    max_utility = -np.inf
    best_bid_idx = None
    auction_agent = copy.deepcopy(auction_agent)
    #for the adversarial case in req4
    if exclude_bidders is not None:
        assert idx_agent not in exclude_bidders, "idx_agent cannot be in exclude_bidders"
        #compute new position of idx_agent in auction_agent.ctrs after excluding exclude_bidders
        exclude_bidders = np.array(exclude_bidders)
        exclude_bidders = np.sort(exclude_bidders)
        before_ag_idx = exclude_bidders[exclude_bidders < idx_agent]
        idx_agent = idx_agent - len(before_ag_idx)


        #now delete elements with index exclude_bidders from auction_agent.ctrs
        auction_agent.ctrs = np.delete(auction_agent.ctrs, exclude_bidders)
        auction_agent.n_adv = len(auction_agent.ctrs)
        #now the same for all_bids
        all_bids = np.delete(all_bids, exclude_bidders, axis=0)

   



    for bid_idx, bid in enumerate(discr_bids):
        c = 0 # total money spent
        bid_utility = 0
        temp_utilities = np.zeros(n_auctions)
        temp_bids= np.zeros(n_auctions)
        temp_payments = np.zeros(n_auctions)            
        for auction_idx in range(n_auctions):
            if c <= budget-1:
                all_bids[idx_agent, auction_idx] = bid
                winner, _ = auction_agent.get_winners(all_bids[:, auction_idx])
                bid_utility += (my_valuation-bid)*(winner == idx_agent)
                c += bid*(winner == idx_agent)

                temp_bids[auction_idx] = bid
                temp_utilities[auction_idx] = (my_valuation-bid)*(winner == idx_agent)
                temp_payments[auction_idx] = bid*(winner == idx_agent)
            else:
                temp_bids[auction_idx] = 0
                temp_payments[auction_idx] = 0
                temp_utilities[auction_idx] = 0
                break
        if bid_utility > max_utility: 
            max_utility = copy.deepcopy(bid_utility)
            best_bid_idx = bid_idx
            clairvoyant_utilities = copy.deepcopy(temp_utilities)
            clairvoyant_bids = np.copy(temp_bids)
            clairvoyant_payments = np.copy(temp_payments)
        
            
        best_bid = discr_bids[best_bid_idx]
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments

def get_clairvoyant_pricing_adversarial(my_prices, my_rewards, discr_prices, T_pricing, adv_pricing_agent, num_buyers):
    max_reward = -np.inf
    best_price = None
    best_price_idx = None

    for price_idx, price in enumerate(discr_prices):
        indices = np.where(my_prices == price)
        total_reward = np.sum(my_rewards[indices])
        if total_reward > max_reward:
            max_reward = total_reward
            best_price = price
            best_price_idx = price_idx
    
    #now get the rewards for the best price with each theta_t
    rewards_clairvoyant = np.zeros(T_pricing)
    adv_pricing_agent.reset() #resets counter to t = 0

    #if num_buyers is an integer, I assume that the number of buyers is the same at each round
    if isinstance(num_buyers, int):
        for t in range(T_pricing):
            _, r_t = adv_pricing_agent.round(np.array([best_price]), num_buyers)
            rewards_clairvoyant[t] = r_t
    elif isinstance(num_buyers, np.ndarray):
        for t in range(T_pricing):
            _, r_t = adv_pricing_agent.round(np.array([best_price]), num_buyers[t])
            rewards_clairvoyant[t] = r_t
    else:
        raise ValueError('num_buyers must be an integer or a numpy array')

    
    return rewards_clairvoyant, best_price




def plot_clayrvoyant_truthful(budget, clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments):
    plt.title('Clairvoyant Chosen Bids')
    plt.plot(clairvoyant_bids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.show()

    plt.title('Clairvoyant Cumulative Payment')
    plt.plot(np.cumsum(clairvoyant_payments))
    plt.axhline(budget, color='red', linestyle='--', label='Budget')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum m_t~ 1_{b_t > m_t}$')
    plt.show()

    plt.title('Clairvoyant Cumulative Utility')
    plt.plot(np.cumsum(clairvoyant_utilities))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum u_t$')
    plt.show()

def plot_agent_bidding(budget, agent_bids, agent_utilities, agent_payments):
    plt.title('Agent Bids')
    plt.plot(agent_bids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Agent Chosen Bids')
    plt.show()

    plt.title('Agent Cumulative Payment')
    plt.plot(np.cumsum(agent_payments))
    plt.axhline(budget, color='red', linestyle='--', label='Budget')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum m_t~ 1_{b_t > m_t}$')
    plt.show()

    plt.title('Agent Cumulative Utility')
    plt.plot(np.cumsum(agent_utilities))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum u_t$')
    plt.show()

def plot_agent_pricing(agent_prices, agent_sales, agent_rewards):
    plt.title('Agent Prices')
    plt.plot(agent_prices)
    plt.xlabel('$t$')
    plt.ylabel('$p_t$')
    plt.title('Chosen Prices')
    plt.show()

    '''plt.title('Agent Cumulative Sales')
    plt.plot(np.cumsum(agent_sales))
    plt.xlabel('$t$')
    plt.ylabel('$\sum s_t$')
    plt.show()

    plt.title('Agent Cumulative Revenue')
    plt.plot(np.cumsum(agent_rewards))
    plt.xlabel('$t$')
    plt.ylabel('$\sum r_t$')
    plt.show()'''

def plot_demand_curve(prices, conversion_probability, n_customers):
    expected_demand_curve = n_customers * conversion_probability(prices)

    # numpy allows us to pass an array of parameters instead of a single parameter to a distribution
    estimated_demand_curve = np.random.binomial(n_customers, conversion_probability(prices))
    # an array of random variables is sampled, each using one of the parameters in the array

    plt.figure()
    plt.plot(prices, expected_demand_curve, label='Expected Demand Curve')
    plt.plot(prices, estimated_demand_curve, label='Estimated Demand Curve')
    plt.xlabel('Item Price')
    plt.legend()
    plt.show();

def plot_profit_curve(prices, conversion_probability, n_customers, cost):

    expected_profit_curve = n_customers * conversion_probability(prices) * (prices-cost)

    estimated_profit_curve = np.random.binomial(n_customers, conversion_probability(prices)) * (prices-cost)

    best_price_index = np.argmax(expected_profit_curve)
    best_price = prices[best_price_index]

    plt.figure()
    plt.plot(prices, expected_profit_curve, label='Expected Profit Curve')
    plt.plot(prices, estimated_profit_curve, label='Estimated Profit Curve')
    plt.scatter(best_price, expected_profit_curve[best_price_index], color='red', s=50)
    plt.xlabel('Item Price')
    plt.legend()
    plt.show();


def plot_regret(agent_utilities, clairvoyant_utilities):
    regret = np.cumsum(clairvoyant_utilities) - np.cumsum(agent_utilities)
    plt.title('Agent Regret')
    plt.plot(regret)
    plt.xlabel('$t$')
    plt.ylabel('Regret')
    plt.show()

#REPORT AND PLOTS
""" class PDFReport:
    def __init__(self, filename, requirement = 1):
        self.requirement = requirement
        assert self.requirement in [1, 2, 3, 4], "Requirement must be 1 or 2"
        self.doc = SimpleDocTemplate(filename, pagesize=A4)
        self.elements = []
        self.width, self.height = A4
        self.styles = getSampleStyleSheet()
        self.story = []

    def header(self, title = None, params=None):
        # Add Title
        if title is None:
            title = f"Requirement {self.requirement} {params['run_type']}"
        title_style = self.styles['Title']
        self.story.append(Paragraph(title, title_style))
        self.story.append(Spacer(1, 0.25 * inch))

        # Add Parameters
        if params:
            param_data = []
            keys = list(params.keys())
            for i in range(0, len(keys), 3):
                row = [
                    f"{keys[i]}: {params[keys[i]]}",
                    f"{keys[i+1]}: {params[keys[i+1]]}" if i + 1 < len(keys) else "",
                    f"{keys[i+2]}: {params[keys[i+2]]}" if i + 2 < len(keys) else ""
                ]
                param_data.append(row)

            param_table = Table(param_data, colWidths=[self.width/3.5, self.width/3.5, self.width/3.5])
            param_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            self.story.append(param_table)
            self.story.append(Spacer(1, 0.25 * inch))

    def add_text(self, text):
        text_style = self.styles['BodyText']
        self.story.append(Paragraph(text, text_style))
        self.story.append(Spacer(1, 0.25 * inch))

    def add_image(self, image_path, width=4*inch, height=3*inch):
        img = Image(image_path, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.25 * inch))

    def add_double_image(self, image_path1, image_path2, width=3.5*inch, height=2.5*inch):
        img1 = Image(image_path1, width=width, height=height)
        img2 = Image(image_path2, width=width, height=height)
        
        table_data = [[img1, img2]]
        image_table = Table(table_data, colWidths=[width, width])
        self.story.append(image_table)
        self.story.append(Spacer(1, 0.25 * inch))

    def save(self):
        self.doc.build(self.story) """