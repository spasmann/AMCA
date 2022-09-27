#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 09:13:55 2022

@author: sampasmann

modified from:
    https://github.com/jpcolino/IPython_notebooks/blob/master/Least%20Square%20Monte%20Carlo%20Implementation%20in%20a%20Python%20Class.ipynb
"""
import numpy as np
from numba import njit, jit, vectorize
from scipy.stats import qmc
from linear_regression import fit_poly, eval_polynomial

@njit
def AMCA(option_type, S0, strike, T, M, r, div, sigma, N):
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
    
    """
    try:
        assert isinstance(option_type, str)
        S0 = float(S0)
        strike = float(strike)
        assert T > 0
        T = float(T)
        assert M > 0
        M = int(M)
        assert r >= 0
        r = float(r)
        assert div >= 0
        div = float(div)
        assert sigma > 0
        sigma = float(sigma)
        assert simulations > 0
        simulations = int(simulations)
    except ValueError:
        raise ValueError('Error: invalid Options parameters')

    if (option_type != 'call') and (option_type != 'put'):
        raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
    if (S0 < 0 )or (strike < 0) or (T <= 0) or (r < 0) or (div < 0) or sigma < 0:
        raise ValueError('Error: Negative inputs not allowed')
        """

    dt = T / float(M)
    discount = np.exp(-r*dt)
    price = Price(option_type, MCprice_matrix, strike, M, N, dt, discount)
    
    return price 

@jit(nopython=True, fastmath=True)
def MCprice_matrix(M, N, S0, r, sigma, dt, seed = 123):
    """ Returns MC price matrix rows: time columns: price-path simulation """

    np.random.seed(seed)
    samples = np.random.standard_normal(size=(M,int(N/2)))
    samples = np.concatenate((samples,-samples),axis=1)
    MCprice_matrix = np.zeros((M+1, N), dtype=np.float64)
    MCprice_matrix[0,:] = S0
    drift = (r - 0.5*sigma ** 2.) * dt
    #engine = qmc.MultivariateNormalQMC(mean=np.zeros(M), 
    #                                   cov=np.eye((M)),
    #                                   engine=qmc.Sobol(M,scramble=True,seed=seed))
    #samples = engine.random(int(0.5*N)).T
    #samples = np.concatenate((samples,-samples),axis=1)
    Z = sigma*np.sqrt(dt)*samples
    for t in range(1, M + 1):
        MCprice_matrix[t, :] = (MCprice_matrix[t-1, :]*np.exp(drift + Z[t-1,:]))
    
    return MCprice_matrix

@njit
def MCpayoff(option_type, MCprice_matrix, strike, M, N):
    """Returns the inner-value of American Option"""
    if (option_type == 'call'):
        payoff = np.maximum(MCprice_matrix - strike,
                       np.zeros((M+1, N),
                       dtype=np.float64))
    elif (option_type == 'put'):
        payoff = np.maximum(strike - MCprice_matrix,
                        np.zeros((M+1, N),
                        dtype=np.float64))
    else:
        print("Invalid Option Type")
        Exception
        
    return payoff

@jit(nopython=True, fastmath=True)
def value_vector(option_type, strike, M, N, dt, discount):
    price_sim           = MCprice_matrix(M, N, S0, r, sigma, dt)
    payoff              = MCpayoff(option_type, price_sim, strike, M, N)
    value_matrix        = np.zeros_like(payoff)
    value_matrix[-1, :] = payoff[-1, :]
    for t in range(M-1, 0 , -1):
        regression         = fit_poly(price_sim[t, :], 
                                value_matrix[t + 1, :]*discount, 5)
        continuation_value = eval_polynomial(regression, price_sim[t, :], 5)
        value_matrix[t, :] = payoff[t, :]
        for i in range(N):
            if (value_matrix[t,i] < continuation_value[i]):
                value_matrix[t,i] = value_matrix[t+1,i]*discount
        
    return (value_matrix[1,:]*discount)

@njit
def Price(option_type, MCprice_matrix, strike, M, N, dt, discount): 
    return (np.sum(value_vector(option_type, strike, M, N, dt, discount))/float(N))


def prices():
    for S0 in (36., 38., 40., 42., 44.):  # initial stock price values
        for vol in (0.2, 0.4):  # volatility values
            for T in (1.0, 2.0):  # times-to-maturity
                #with ThreadPoolExecutor(6) as ex:
                AmericanPUT = AMCA('put', S0, 40., T, 50, 0.06, 0.06, vol, 100000)
                print("Initial price: %4.1f, Sigma: %4.2f, Expire: %2.1f --> Option Value %8.3f" % (S0, vol, T, AmericanPUT))
    

def time_test():
    from time import time
    t0 = time()
    optionValues = prices()  # calculate all values
    t1 = time(); d1 = t1 - t0
    return print("Duration in Seconds %6.3f" % d1)

if (__name__=="__main__"):

    S0      = 155.87  # underlying stock price
    strike  = 155.87  # strike price
    T       = 3.0   # Time in years
    sigma   = 0.018   # variance
    M       = 252    # Number of exercise opportunities per year
    r       = 0.86  # constant risk free short rate
    div     = 0.06  # (dividend yield)
    N       = 100000 # Number of simulations per time step (T/M)
    #time_test()
    #import time
    print("Price: ", AMCA('call', S0, strike, T, M, r, div, sigma, 100))
    print("Price: ",AMCA('call', S0, strike, T, M, r, div, sigma, N))
