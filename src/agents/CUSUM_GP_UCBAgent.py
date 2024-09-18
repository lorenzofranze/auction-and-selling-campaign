import numpy as np

from .AbstractAgent import Agent

class CUSUM_GP_UCBAgent(Agent):
    def __init__(self, T, M, h, epsilon, K, scale=1):
        self.T = T
        self.K = K
        # GPUCB agent assumes prices are in [0,1] (which is why we need rescaling outside the class)
        self.arms = np.linspace(0, 1, K)
        self.gp = RBFGaussianProcess(scale=scale, reg=0.1)
        self.a_t = None
        self.mu_t = np.zeros(K)
        # all formulas are in the linked paper
        self.sigma_t = np.zeros(K)
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.N_pulls = np.zeros(K)
        self.t = 0
        #for change detection
        self.scale = scale
        self.M = M
        self.h = h
        self.epsilon = epsilon
        self.counters = np.repeat(M, K)
        self.reset_times = np.zeros(K)
        self.all_rewards = [[] for _ in np.arange(K)]
        self.n_resets = np.zeros(K)

        #for same value detection
        self.last_value = None
        self.n_same_value = 0
    
    def pull_arm(self):

        if (self.counters > 0).any():
            for a in np.arange(self.K):
                if self.counters[a] > 0:
                    self.counters[a] -= 1
                    self.a_t = a
                    break
        else:
            if self.n_same_value >= 40:
                return self.last_value
            
            self.mu_t, self.sigma_t = self.gp.predict(self.arms)
            ucbs = self.mu_t + self.beta(self.t) * self.sigma_t  # beta encourages exploration
            self.a_t = np.argmax(ucbs)

        #same value detection
         
        if self.last_value!=None and abs(self.last_value - self.arms[self.a_t]) < 1e-8 :
            self.n_same_value += 1
        else:
            self.n_same_value = 0
            self.last_value = self.arms[self.a_t]
        
        self.last_value = self.arms[self.a_t] 

        return self.arms[self.a_t]
    
    def update(self, r_t):


        if self.counters[self.a_t] == 0:
            if self.change_detection():
                for a in range(self.K):
                    self.n_resets[a] +=1
                    self.counters[a] = self.M
                    self.all_rewards[a] = []
                    self.reset_times[a] = self.t
                self.gp.k_xx_inv = None #reset GP
                self.last_value = None
                self.n_same_value = 0
                
            else:
                if self.n_same_value < 40:
                    self.gp = self.gp.fit(self.arms[self.a_t], r_t)
                else:
                    self.N_pulls[self.a_t] += 1
                    self.all_rewards[self.a_t].append(r_t)


        self.t += 1
    
    def change_detection(self):
        ''' CUSUM CD sub-routine. This function returns 1 if there's evidence that the last pulled arm has its average reward changed '''
        u_0 = np.mean(self.all_rewards[self.a_t][:self.M])
        sp, sm = (np.array(self.all_rewards[self.a_t][self.M:])- u_0 - self.epsilon , u_0 - np.array(self.all_rewards[self.a_t][self.M:]) - self.epsilon )
        gp, gm = 0, 0
        for sp_, sm_ in zip(sp, sm):
            gp, gm = max([0, gp + sp_]), max([0, gm + sm_])
            
            if max([gp, gm]) >= self.h:
                return True
        return False



class RBFGaussianProcess:
    def __init__(self, scale=1, reg=1e-2):
        self.scale = scale 
        self.reg = reg
        self.k_xx_inv = None

    def rbf_kernel_incr_inv(self, B, C, D):
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = - self.k_xx_inv @ B @ temp
        block3 = - temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        res = np.concatenate((res1, res2), axis=0)
        return res

    def rbf_kernel(self, a, b):
        a_ = a.reshape(-1, 1)
        b_ = b.reshape(-1, 1)
        output = -1 * np.ones((a_.shape[0], b_.shape[0]))
        for i in range(a_.shape[0]):
            output[i, :] = np.power(a_[i] - b_, 2).ravel()
        return np.exp(-self.scale * output)
    
    def fit(self, x=np.array([]), y=np.array([])):
        x,y = np.array(x),np.array(y)
        if self.k_xx_inv is None:
            self.y = y.reshape(-1,1)
            self.x = x.reshape(-1,1)
            k_xx = self.rbf_kernel(self.x, self.x) + self.reg * np.eye(self.x.shape[0])
            self.k_xx_inv = np.linalg.inv(k_xx)
        else:
            B = self.rbf_kernel(self.x, x)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
            self.k_xx_inv = self.rbf_kernel_incr_inv(B, B.T, np.array([1 + self.reg]))

        return self

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()
    