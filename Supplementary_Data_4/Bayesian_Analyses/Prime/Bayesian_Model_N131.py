#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from scipy import optimize
import theano.tensor as tt

sigma_exp = np.sqrt(0.81**2 / 2) # RMSE of true Kd-derived DeltaG and IC50-derived DeltaG

kdraws=5000

# read all of the data
def read_all_inpfile(inpfile):
    #   0              1           2          3         4              5                6              7                8              9               10
    # TKI       mutation        IC50    DDG_exp DDG_prime       DDG_fep1        dDDG_fep1       DDG_fep2        dDDG_fep2       DDG_fep3        dDDG_fep3
    tki, mut, ic50, ddg_exp, ddg_prime, ddg_fep, dddg_fep = [], [], [], [], [], [], []
    with open(inpfile) as input:
        input.readline() # skip header
        for line in input:
            try:
                data = [ item.strip() for item in line.split() ]
                if len(data) < 11:
                    continue # data for mutation is incomplete...
                else:
                    if data[0]=='gefitinib' or data[0]=='erlotinib':
                        continue # N=129/131 analysis; comment-out this logic if you are interested in more 8^)
#                    if float(data[2])==10000:
#                        continue # ic50 beyond range of assay
                    tki.append(data[0]); mut.append(data[1]); ic50.append(float(data[2]))
                    ddg_exp.append(float(data[3])); ddg_prime.append(float(data[4]))
                    ddg_fep.append([float(data[i]) for i in [5,7,9]]);                     
                    dddg_fep.append([float(data[i]) for i in [6,8,10]]);
            except:
                pass
    return tki, mut, np.array(ic50), np.array(ddg_exp), np.array(ddg_prime), np.array(ddg_fep), np.array(dddg_fep)

# Compute and return CIs
def ci(xtrace, width=0.95):
    """
    Compute confidence interval
    """
    x = np.array(xtrace)
    x.sort()
    n = len(xtrace)
    low = int(n * (1-width)/2.0)
    high = int(n * (1 - (1-width)/2.0))    
    return x[low], x[high]

# Determine which classification states each Experimental and Calculation DDG resides in
def confumat(Expe,Pred,Cut):
    neg_cut=-float(Cut)
    pred_state, expe_state = [], []
    for x in Pred:
        if float(x) >  abs(Cut):
            pred_state.append('r')
        if float(x) <= abs(Cut):
            pred_state.append('s')
    for x in Expe:
        if float(x) >  abs(Cut):
            expe_state.append('R')
        if float(x) <= abs(Cut):
            expe_state.append('S')
    return expe_state, pred_state

# create a confusion matrix
def comp_classy(Expe,Pred,cut=1.36,kprint='mute'):
  kTrues, kFalses = [], []
  Cut = float(cut)
  expe_state, pred_state = confumat(Expe,Pred,Cut)
  Ss, Rs, Sr, Rr = 0,0,0,0
  for x,y in zip(expe_state, pred_state):
    if x == 'R':
        if y == 'r':
            Rr = Rr + 1
        if y == 's':
            Rs = Rs + 1
    if x == 'S':
        if y == 'r':
            Sr = Sr + 1
        if y == 's':
            Ss = Ss + 1
  Trues  = Ss + Rr
  Falses = len(Pred) - Trues
  SsSr   = Ss + Sr  #specificity = TN / (TN+FP)
  RrRs   = Rr + Rs  #sensitivity = TP / (TP+FN)
  if SsSr == 0:
    SsSr = 0.000000000000000000000001  #protect if div-by-zero
  if RrRs == 0:
    RrRs = 0.000000000000000000000001  #protect if div-by-zero
  sensitiv = float(Rr)/RrRs
  specific = float(Ss)/SsSr
  accuracy = float(Trues)/len(Pred)
  return accuracy, specific, sensitiv

# Get the data from the input file 
tki, mut, ic50, DDG_exp_obs, DDG_prime_obs, DDG_FEP_obs, dDDG_FEP_obs = read_all_inpfile('./FreeEnergyResults.dat')
nmutants = len(tki)
print(nmutants)

# Use average over three replicates
DDG_FEP_mean_obs = DDG_FEP_obs.mean(1)
dDDG_FEP_mean_obs = DDG_FEP_obs.std(1) / np.sqrt(3)

# Build the Bayesian model for both FEP+ and Prime, sharing priors where possible

joint_model = pm.Model()

DDG_FEP_obs

with joint_model:
    # Priors on nuisance parameters (that will be marginalized out)
    mu_mut = pm.Uniform('mu_mut', -6, +6) # kcal/mol
    sigma_mut = pm.HalfFlat('sigma_mut', testval=1) # kcal/mol
        
    DDG_true = pm.Normal('DDG_true', mu=mu_mut, sd=sigma_mut, shape=nmutants)    
    
    # Priors on unknown values of interest
    RMSE_FEP_true = pm.HalfFlat('RMSE_FEP', testval=1) # kcal/mol, uninformative prior for nonnegative values of the RMSE
    MUE_FEP_true = pm.Deterministic('MUE_FEP', RMSE_FEP_true*np.sqrt(2.0/np.pi)) # store MUE estimate alongside RMSE using analytical relationship
    RMSE_prime_true = pm.HalfFlat('RMSE_prime', testval=1) # kcal/mol, uninformative prior for nonnegative values of the RMSE
    MUE_prime_true = pm.Deterministic('MUE_prime', RMSE_prime_true*np.sqrt(2.0/np.pi)) # store MUE estimate alongside RMSE using analytical relationship
    
    # Unknown true computed values
    DDG_FEP_calc_true = pm.Normal('DDG_FEP_calc_true', mu=(DDG_true), sd=RMSE_FEP_true, shape=nmutants)    
    
    # Data likelihood for observed data
    DDG_exp = pm.Normal('DDG_exp', mu=DDG_true, sd=sigma_exp, shape=nmutants, observed=DDG_exp_obs)
    DDG_FEP = pm.Normal('DDG_FEP_%d', mu=DDG_FEP_calc_true, sd=dDDG_FEP_mean_obs, shape=nmutants, observed=DDG_FEP_mean_obs)
    DDG_prime_calc_true = pm.Normal('DDG_prime_calc_true', mu=(DDG_true), sd=RMSE_prime_true, shape=nmutants, observed=DDG_prime_obs)


# Sample from the posterior and plot the RMSE confidence interval
with joint_model:
    trace = pm.sample(draws=5000)

#
#    Quantitative metrics report (MUE, RMSE)
#
def show_statistics(trace):
    for statistic in ['RMSE_FEP', 'MUE_FEP', 'RMSE_prime', 'MUE_prime']:
        x_t = np.array(trace[statistic])
        x_mean = x_t.mean()
        x_low, x_high = ci(x_t)
        print('True %12s: %.3f [%.3f, %.3f] kcal/mol (95%% CI)' % (statistic, x_mean, x_low, x_high))

# Quantitative metrics report        
#show_statistics(trace)

#
#    Classification metrics report (Accuracy==accu, Specificity==spec, Sensitivity==sens)
#

# Compute classification accuracy, specificity, sensitivity using the true experimental and true computed values...
def get_confusion(trace, calc_true, kdraws):
    ACCU, SPEC, SENS = [], [], []
    for idraw in range(kdraws):   
        if calc_true == 'DDG_FEP_calc_true':
            ddg_exp_true = trace['DDG_true'][idraw]
            ddg_cal_true = trace[calc_true][idraw]
        if calc_true == 'DDG_prime_calc_true':
            ddg_exp_true = trace['DDG_true'][idraw]
            ddg_cal_true = DDG_prime_obs
        accu, spec, sens = comp_classy( ddg_exp_true, ddg_cal_true )
        ACCU.append(accu); SPEC.append(spec); SENS.append(sens)
    a_t = np.array(ACCU)
    b_t = np.array(SPEC)
    c_t = np.array(SENS)
    return a_t, b_t, c_t

def show_confusion(a_t, b_t, c_t):
    a_mean = a_t.mean() ; a_low, a_high = ci(a_t)
    b_mean = b_t.mean() ; b_low, b_high = ci(b_t)
    c_mean = c_t.mean() ; c_low, c_high = ci(c_t)
    print ('True Accuracy: %.3f [%.3f, %.3f] (95%%CI)' % (a_mean, a_low, a_high) )
    print ('True Specific: %.3f [%.3f, %.3f] (95%%CI)' % (b_mean, b_low, b_high) )
    print ('True Sensitiv: %.3f [%.3f, %.3f] (95%%CI)' % (c_mean, c_low, c_high) )

for physics in [ 'Prime' ]: #, 'FEP' ]:
    print ('Classification Results for %s: ' % (physics) )
    if physics == 'Prime':
        a_t, b_t, c_t = get_confusion(trace, 'DDG_prime_calc_true', kdraws)
        show_confusion(a_t, b_t, c_t)
    if physics == 'FEP':
        a_t, b_t, c_t = get_confusion(trace, 'DDG_FEP_calc_true', kdraws)
        show_confusion(a_t, b_t, c_t)
        





