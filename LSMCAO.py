#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:17:08 2022

@author: sampasmann
"""

import numpy as np
from scipy.stats import qmc

class AmericanOptionsLSMC(object):
    """ Class for American options pricing using Longstaff-Schwartz (2001):
    "Valuing American Options by Simulation: A Simple Least-Squares Approach."
    Review of Financial Studies, Vol. 14, 113-147.
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term 
    
    Unitest(doctest): 
    >>> AmericanPUT = AmericanOptionsLSMC('put', 36., 40., 1., 50, 0.06, 0.06, 0.2, 10000  )
    Price:  4.473117701771221
    Delta:  -0.7112251324731934
    """

    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, simulations):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert div >= 0
            self.div = float(div)
            assert sigma > 0
            self.sigma = float(sigma)
            assert simulations > 0
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Options parameters')

        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

    @property
    def MCprice_matrix(self, seed = 123):
        """ Returns MC price matrix rows: time columns: price-path simulation """

        np.random.seed(seed)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0,:] = self.S0
        for t in range(1, self.M + 1):
            brownian = np.random.standard_normal( int(self.simulations / 2))
            brownian = np.concatenate((brownian, -brownian))
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
                                  + self.sigma * brownian * np.sqrt(self.time_unit)))
        
        """
        np.random.seed(seed)
        #samples = np.random.standard_normal(size=(self.M,int(self.simulations/2)))
        #samples = np.concatenate((samples,-samples),axis=1)
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0,:] = self.S0
        drift = (self.r - 0.5*self.sigma ** 2.) * self.time_unit
        engine = qmc.MultivariateNormalQMC(mean=np.zeros(self.M), 
                                           cov=np.eye((self.M)),
                                           engine=qmc.Sobol(self.M,scramble=True,seed=seed))
        samples = engine.random(int(0.5*self.simulations)).T
        samples = np.concatenate((samples,-samples),axis=1)
        Z = self.sigma*np.sqrt(self.time_unit)*samples
        for t in range(1, self.M + 1):
            MCprice_matrix[t, :] = (MCprice_matrix[t-1, :]*np.exp(drift + Z[t-1,:]))
        """
        return MCprice_matrix

    @property
    def MCpayoff(self):
        """Returns the inner-value of American Option"""
        
        if self.option_type == 'call':
            payoff = np.maximum(self.MCprice_matrix - self.strike,
                           np.zeros((self.M + 1, self.simulations),
                           dtype=np.float64))
        else:
            payoff = np.maximum(self.strike - self.MCprice_matrix,
                            np.zeros((self.M + 1, self.simulations),
                            dtype=np.float64))

        return payoff

    @property
    def value_vector(self):
        value_matrix = np.zeros_like(self.MCpayoff)
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        for t in range(self.M - 1, 0 , -1):
            regression = np.polyfit(self.MCprice_matrix[t, :], value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression, self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value,
                                          self.MCpayoff[t, :],
                                          value_matrix[t + 1, :] * self.discount)

        return value_matrix[1,:] * self.discount


    @property
    def price(self): return np.sum(self.value_vector) / float(self.simulations)
    
    @property
    def delta(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)
    
    @property
    def gamma(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0 + diff, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0 - diff, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.delta - myCall_2.delta) / float(2. * diff)
    
    @property
    def vega(self):
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma + diff, 
                                       self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0,
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma - diff, 
                                       self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)    
    
    @property
    def rho(self):        
        diff = self.r * 0.01
        if (self.r - diff) < 0:        
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T, self.M, 
                                       self.r + diff, self.div, self.sigma, 
                                       self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T, self.M, 
                                       self.r, self.div, self.sigma, 
                                       self.simulations)
            return (myCall_1.price - myCall_2.price) / float(diff)
        else:
            myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T, self.M, 
                                       self.r + diff, self.div, self.sigma, 
                                       self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T, self.M, 
                                       self.r - diff, self.div, self.sigma, 
                                       self.simulations)
            return (myCall_1.price - myCall_2.price) / float(2. * diff)
    
    @property
    def theta(self): 
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T + diff, self.M, 
                                       self.r, self.div, self.sigma, 
                                       self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.option_type, self.S0, 
                                       self.strike, self.T - diff, self.M, 
                                       self.r, self.div, self.sigma, 
                                       self.simulations)
        return (myCall_2.price - myCall_1.price) / float(2. * diff)

def prices():
    for S0 in (36., 38., 40., 42., 44.):  # initial stock price values
        for vol in (0.2, 0.4):  # volatility values
            for T in (1.0, 2.0):  # times-to-maturity
                AmericanPUT = AmericanOptionsLSMC('put', S0, 40., T, 50, 0.06, 0.06, vol, 100000)
                print("Initial price: %4.1f, Sigma: %4.2f, Expire: %2.1f --> Option Value %8.3f" % (S0, vol, T, AmericanPUT.price))
    

def original_time_test():
    from time import time
    t0 = time()
    optionValues = prices()  # calculate all values
    t1 = time(); d1 = t1 - t0
    return print("Duration in Seconds %6.3f" % d1)



"""  
if (__name__ == "__main__"):
    import time
    S0      = 36.0  # underlying stock price
    strike  = 40.0  # strike price
    T       = 1.0   # Time in years
    sigma   = 0.4   # variance
    M       = 50    # Number of exercise opportunities per year
    r       = 0.06  # constant risk free short rate
    div     = 0.06  # (dividend yield)
    N       = 10000 # Number of simulations per time step (T/M)
    #AmericanPUT = AmericanOptionsLSMC('put', S0, strike, T, M, r, div, sigma, N)
    #print( 'Delta: ', AmericanPUT.delta)
    start = time.time()
    time_test()
    stop = time.time()
    print("Time: ",stop-start)
"""