# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

# risk budgeting optimization
def calculate_portfolio_var(w, Covar):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*Covar*w.T)[0,0]

def calculate_risk_contribution(w, Covar):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,Covar))
    # Marginal Risk Contribution
    MRC = Covar*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(w,pars):
    #log.info("risk_budget_objective")
    # calculate portfolio risk
    Covar = pars[0]# covariance table
    Target_Weights = pars[1] # risk target in percent of portfolio risk
    sigma_stdev =  np.sqrt(calculate_portfolio_var(w,Covar)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sigma_stdev, Target_Weights))
    asset_RC = calculate_risk_contribution(w, Covar)
    J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
    return J

### MAIN
port_perf = pd.read_csv('RiskParity.csv', index_col=0)
#print(port_perf.iloc[:, 0:3])

# Take a slice of first 3 columns to retrieve asset class returns
idx_perf = port_perf.iloc[:, 0:3]

number_of_assets = len(idx_perf.columns.values)

# Calculate covariance of asset returns, cast as numpy matrix
covar_matrix = np.matrix(idx_perf.cov())  
#print(idx_perf.cov())

bnds = [(0,1)] * number_of_assets # bounds for weights (number of bounds  = to number of assets)    
#print("bnds1={0}".format(bnds1))
# Constraints: Total weight must be 100% and long-only
cons = ({'type': 'eq', 'fun': total_weight_constraint}, 
        {'type': 'ineq', 'fun': long_only_constraint})

target_weights = 1.0*np.ones_like(idx_perf.iloc[0])/number_of_assets
print("Target weights: {0}".format(target_weight))

# We are looking for equal risk portfolios so we set risk to 1/N where N=number of assets

optimizationRslts = minimize(risk_budget_objective,
                             target_weights,
                             args=[covar_matrix, target_weights], 
                             method='SLSQP',
                             constraints=cons,
                             bounds=bnds1,
                             options={'ftol': 1e-12}) # , options={'disp': True}   

if optimizationRslts.success:
    allocation = optimizationRslts.x     # Get optimization weights
    print(allocation)


