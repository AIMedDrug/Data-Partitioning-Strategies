#!/usr/bin/python
import sys
import numpy as np
import performance_metrics_computer as pmc
import datahandlertab2 as daha

#
#  Bootstrap with replacement statistics 
#

inpfile='./FreeEnergyResults.dat'

kut=1.36 ; kutoff = float(kut)

kprint = 'mute'#'matrix'

nbootstrap=int(1000)

drug=sys.argv[1]

method=sys.argv[2]#method=str("FEP+")


#
# get FEP data from input:
comput='class' # keep all data (incl. L248R/imatinib and T315A/dasatinib)
tki, mut, ic50, c_expe_obs, c_ddg_prime, c_fep_obs1, c_dfep_obs1, c_fep_obs2, c_dfep_obs2, c_fep_obs3, c_dfep_obs3 = daha.read_all_inpfile(inpfile,drug,comput)
if drug == 'all' or drug == 'xtals' or drug == 'imatinib' or drug == 'dasatinib':
    comput='quant' # excl L248R/imatinib and T315A/dasatinib
    tki, mut, ic50, q_expe_obs, q_ddg_prime, q_fep_obs1, q_dfep_obs1, q_fep_obs2, q_dfep_obs2, q_fep_obs3, q_dfep_obs3 = daha.read_all_inpfile(inpfile,drug,comput)
else:
    q_expe_obs, q_ddg_prime, q_fep_obs1, q_dfep_obs1, q_fep_obs2, q_dfep_obs2, q_fep_obs3, q_dfep_obs3 = c_expe_obs, c_ddg_prime, c_fep_obs1, c_dfep_obs1, c_fep_obs2, c_dfep_obs2, c_fep_obs3, c_dfep_obs3


# Get Average of 3 FEP+ runs or the Prime results
if method == 'FEP+':
    q_Pred_avg, q_Pred_std = daha.fep_avg_runs(q_fep_obs1,q_fep_obs2,q_fep_obs3)
    c_Pred_avg, c_Pred_std = daha.fep_avg_runs(c_fep_obs1,c_fep_obs2,c_fep_obs3)
elif method == 'Prime':
    q_Pred_avg = q_ddg_prime
    c_Pred_avg = c_ddg_prime


# get MUE_apparent, MSE_apparent, RMSE_apparent
mue_app, mse_app, rmse_app = pmc.comp_quanty( q_expe_obs, q_Pred_avg )
# get Accuracy, Specificity, Sensitivity apparent
accu_app, spec_app, sens_app = pmc.comp_classy( c_expe_obs, c_Pred_avg, kutoff, kprint )

#
#  BOOSTRAPPING...CLASSIFICATION (INCL. T315A/DASATINIB, L248R/IMATINIB)
#
ndata = int(len(c_Pred_avg))
accu, spec, sens = [], [], []
for boot in range(nbootstrap):
    # Generate bootstrap sample indices to account for finite dataset size
    boot_sample = np.random.choice(range(ndata), ndata)
    # Generate resampled data
    #   get pred values
    np_c_pred_obs = np.array(c_Pred_avg,dtype=float)
    c_pred_boot_obs = np_c_pred_obs[boot_sample]
    #   get expe values
    np_c_expe_obs = np.array(c_expe_obs,dtype=float)
    c_expe_boot_obs = np_c_expe_obs[boot_sample]
    #accuracy, specific, sensitiv
    iaccu, ispec, isens = pmc.comp_classy( c_expe_boot_obs, c_pred_boot_obs, kut )
    accu.append(iaccu) ; spec.append(ispec) ; sens.append(isens)


#
#  BOOSTRAPPING...QUANTITATIVE (EXCL. T315A/DASATINIB, L248R/IMATINIB)
#
ndata = int(len(q_Pred_avg))
mue, mse, rmse = [], [], []
for boot in range(nbootstrap):
    # Generate bootstrap sample indices to account for finite dataset size
    boot_sample = np.random.choice(range(ndata), ndata)
    # Generate resampled data
    #   get pred values
    np_q_pred_obs = np.array(q_Pred_avg,dtype=float)
    q_pred_boot_obs = np_q_pred_obs[boot_sample]
    #   get expe values
    np_q_expe_obs = np.array(q_expe_obs,dtype=float)
    q_expe_boot_obs = np_q_expe_obs[boot_sample]
    # Compute performance metrics
    imue, imse, irmse = pmc.comp_quanty( q_expe_boot_obs, q_pred_boot_obs )
    mue.append(imue) ; mse.append(imse) ; rmse.append(irmse)

#
#  Compute confidence intervals on the bootstrap statistics
#
mue.sort() ; mse.sort() ; rmse.sort()
accu.sort() ; spec.sort() ; sens.sort()

p_lower = int( 0.025 * float(nbootstrap) ) #- int(1)
p_upper = int( 0.975 * float(nbootstrap) ) #- int(1)

#print "Number of bootstrap resampling events: ", nbootstrap
#print "lower bound value: ", p_lower, float(p_lower)*100/nbootstrap, "-th %"
#print "UPPER bound value: ", p_upper, float(p_upper)*100/nbootstrap, "-th %"

str_mue_app=str( '{:.2f}'.format(mue_app)) ; str_mue_low=str( '{:.2f}'.format(mue[p_lower])) ; str_mue_upp=str( '{:.2f}'.format(mue[p_upper]))
str_mse_app=str( '{:.2f}'.format(mse_app)) ; str_mse_low=str( '{:.2f}'.format(mse[p_lower])) ; str_mse_upp=str( '{:.2f}'.format(mse[p_upper]))
str_rmse_app=str( '{:.2f}'.format(rmse_app)) ; str_rmse_low=str( '{:.2f}'.format(rmse[p_lower])) ; str_rmse_upp=str( '{:.2f}'.format(rmse[p_upper]))

str_accu_app=str( '{:.2f}'.format(accu_app)) ; str_accu_low=str( '{:.2f}'.format(accu[p_lower])) ; str_accu_upp=str( '{:.2f}'.format(accu[p_upper]))
str_spec_app=str( '{:.2f}'.format(spec_app)) ; str_spec_low=str( '{:.2f}'.format(spec[p_lower])) ; str_spec_upp=str( '{:.2f}'.format(spec[p_upper]))
str_sens_app=str( '{:.2f}'.format(sens_app)) ; str_sens_low=str( '{:.2f}'.format(sens[p_lower])) ; str_sens_upp=str( '{:.2f}'.format(sens[p_upper]))


#
#  Dataset		&Method		& N$_\mathrm{quant}$ 	&  MUE				& RMSE			& N$_\mathrm{class}$ 	& Accuracy			& Specificity		& Sensitivity			\\
#  all 			& FEP+ 		& 142 			&  $0.76_{0.65}^{0.89}$		&$1.05_{0.87}^{1.24}$ 	& 144 			&$0.89_{0.83}^{0.94}$		&$0.95_{0.91}^{0.98}$	&$0.47_{0.25}^{0.71}$ 	\\

N_quant=str(len(q_Pred_avg)); N_class=str(len(c_Pred_avg))

t_drug=str(drug) + str(' & ')
t_meth=str(method) + str(' & ')
t_N_qu=str(N_quant) + str(' & ')
t_MUE= str('$') + str_mue_app  + str('_{')+str_mue_low+str('}')  + str('^{')+str_mue_upp+str('}$') + str(' & ')
t_RMSE=str('$') + str_rmse_app + str('_{')+str_rmse_low+str('}') + str('^{')+str_rmse_upp+str('}$') + str(' & ')
t_N_cl=str(N_class) + str(' & ')
t_accu=str('$') + str_accu_app + str('_{')+str_accu_low+str('}') + str('^{')+str_accu_upp+str('}$') + str(' & ')
t_spec=str('$') + str_spec_app + str('_{')+str_spec_low+str('}') + str('^{')+str_spec_upp+str('}$') + str(' & ')
t_sens=str('$') + str_sens_app + str('_{')+str_sens_low+str('}') + str('^{')+str_sens_upp+str('}$')

if drug == 'axitinib' or drug == 'ponatinib':
    t_sens = '$NA$'

print (t_drug + t_meth + t_N_qu + t_MUE + t_RMSE + t_N_cl + t_accu + t_spec + t_sens + ' \\\\ ')


#####
#####
