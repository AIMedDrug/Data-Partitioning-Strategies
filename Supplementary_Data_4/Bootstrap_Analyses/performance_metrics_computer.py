#
#   Library of functions to compute performance metrics
#
import numpy as np
# compute quantitative performance metrics MUE and RMSE
def comp_quanty( expe, pred ):
    sum_dif, sum_dif2 = 0, 0
    for x,y in zip( expe, pred ):
       dif  = float(x) - float(y)
       dif2 = dif*dif
       sum_dif  = sum_dif  + abs(dif)
       sum_dif2 = sum_dif2 + dif2
    mue = sum_dif/float(len(expe))
    mse = sum_dif2/float(len(expe))
    rmse = np.sqrt(mse)
    return mue, mse, rmse

#

##########################################################
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
##########################################################
##########################################################
def comp_classy(Expe,Pred,cut=1.4,kprint='mute'):
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
  if kprint == 'matrix':
        print (Cut, 'S', 'R')
        print ('s', Ss, Rs)
        print ('r', Sr, Rr)
        print ("---------------")
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

