#
#  library to handle Hauser-s data files...
#
import numpy as np

# read all_all_data
def read_all_inpfile(inpfile,drug,comput):
    #   0              1           2          3         4              5                6              7                8              9               10
    # TKI	mutation	IC50	DDG_exp	DDG_prime	DDG_fep1	dDDG_fep1	DDG_fep2	dDDG_fep2	DDG_fep3	dDDG_fep3
    tki, mut, ic50, ddg_exp, ddg_prime, ddg_fep1, dddg_fep1, ddg_fep2, dddg_fep2, ddg_fep3, dddg_fep3 = [], [], [], [], [], [], [], [], [], [], []
    with open(inpfile) as input:
        for line in input:
            try:
                data = [ item.strip() for item in line.split() ]
                if len(data) < 11:
                    continue # data for mutation is incomplete...
                elif data[0] == 'TKI':
                    continue # skip the header of the file...
                elif data[2] == '10000' and comput == 'quant':
                    continue # skip beyond assay concentration limit ONLY for MUE, RMSE
#               # TKI-by-TKI
                elif data[0] == drug: # or data[0] == drug2:
                    tki.append(data[0]); mut.append(data[1]); ic50.append(data[2])
                    ddg_exp.append(data[3]); ddg_prime.append(data[4])
                    ddg_fep1.append(data[5]); dddg_fep1.append(data[6])
                    ddg_fep2.append(data[7]); dddg_fep2.append(data[8])
                    ddg_fep3.append(data[9]); dddg_fep3.append(data[10])
#               # All data (xtals + Glide)
                elif drug == 'all':
                    tki.append(data[0]); mut.append(data[1]); ic50.append(data[2])
                    ddg_exp.append(data[3]); ddg_prime.append(data[4])
                    ddg_fep1.append(data[5]); dddg_fep1.append(data[6])
                    ddg_fep2.append(data[7]); dddg_fep2.append(data[8])
                    ddg_fep3.append(data[9]); dddg_fep3.append(data[10])
#               # XTALS (excl. erlotinib, gefitinib):
                elif drug == 'xtals':
                    if data[0] == 'gefitinib' or data[0] == 'erlotinib':
                        continue # skip over gefitinib, erlotinib, leave only xtals
                    else:
                        tki.append(data[0]); mut.append(data[1]); ic50.append(data[2])
                        ddg_exp.append(data[3]); ddg_prime.append(data[4])
                        ddg_fep1.append(data[5]); dddg_fep1.append(data[6])
                        ddg_fep2.append(data[7]); dddg_fep2.append(data[8])
                        ddg_fep3.append(data[9]); dddg_fep3.append(data[10])
#               # Glide (incl. erlotinib, gefitinib):
                elif drug == 'Glide':
                    if data[0] == 'gefitinib' or data[0] == 'erlotinib': 
                        tki.append(data[0]); mut.append(data[1]); ic50.append(data[2])
                        ddg_exp.append(data[3]); ddg_prime.append(data[4])
                        ddg_fep1.append(data[5]); dddg_fep1.append(data[6])
                        ddg_fep2.append(data[7]); dddg_fep2.append(data[8])
                        ddg_fep3.append(data[9]); dddg_fep3.append(data[10])
                    else:
                        continue # skip over IF NOT gefitinib, erlotinib
            except:
                pass
    return tki, mut, ic50, ddg_exp, ddg_prime, ddg_fep1, dddg_fep1, ddg_fep2, dddg_fep2, ddg_fep3, dddg_fep3 

# compute average of three FEP runs:
def fep_avg_runs(run1,run2,run3):
    Pred_avg, Pred_std = [], []
    for a,b,c in zip(run1,run2,run3):
        fep_list = []
        fep_list.append(a); fep_list.append(b); fep_list.append(c)
        j=np.array(fep_list,dtype=float)
        avg_fep = np.mean(j)
        Pred_avg.append( avg_fep )
        std_fep = np.std(j, ddof=1)
        Pred_std.append( std_fep )
    return Pred_avg, Pred_std

