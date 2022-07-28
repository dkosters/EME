#!/usr/bin/env python
# coding: utf-8
import os, sys, argparse
import numpy as np
import random
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from itertools import permutations
import csv

from matplotlib import pyplot as plt

"""
SIG_NAMES = {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
}
"""

SIG_NAMES = {
  '3l_512_256_128': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '2l_256_128': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_128': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_512': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_64': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_32': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_16': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_8': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
  '1l_4': {
                   "Ato4l": 1,
                   "hChToTauNu": 3,
                   "hToTauTau": 4,
                   "leptoquark": 5
   },
}    

run = 0

def performance(bkg_events, sig_events):
    # bkg_events is a 1D array of anomaly scores for the background dataset
    # sig_events is a 1D array of anomaly scores for the signal dataset
    # Returns: Area under the ROC curve, and signal efficiencies for three background efficiencies: 10^-2, 10^-3, 10^-4, 10^-5

    #Create background and signal labels
    bkg_labels = np.zeros(len(bkg_events))
    sig_labels = np.ones(len(sig_events))
    
    #stitch all results together
    events = np.append(bkg_events, sig_events)
    labels = np.append(bkg_labels, sig_labels)

    #Build ROC curve using sklearns roc_curve function
    FPR, TPR, thresholds = roc_curve(labels, events)

    #Calculate area under the ROC curve
    AUC = auc(FPR, TPR)

    #background efficiencies
    efficiency1 = 10.0**-2
    efficiency2 = 10.0**-3
    efficiency3 = 10.0**-4
    efficiency4 = 10.0**-5
    #epsilon values
    epsilon1 = 0.0
    epsilon2 = 0.0
    epsilon3 = 0.0
    epsilon4 = 0.0
    #flags to tell when done
    done1 = False
    done2 = False
    done3 = False
    done4 = False


    #iterate through bkg efficiencies and get as close as possible to the desired efficiencies.
    for i in range(len(FPR)):
        bkg_eff = FPR[i]
        if bkg_eff >= efficiency1 and done1 == False:
            epsilon1 = TPR[i]
            done1 = True
        if bkg_eff >= efficiency2 and done2 == False:
            epsilon2 = TPR[i]
            done2 = True
        if bkg_eff >= efficiency3 and done3 == False:
            epsilon3 = TPR[i]
            done3 = True
        if bkg_eff >= efficiency4 and done4 == False:
            epsilon4 = TPR[i]
            done4 = True

        if done1 and done2 and done3 and done4:
            break
            
    return AUC, epsilon1, epsilon2, epsilon3, epsilon4

# radius-bg-MSE0-run0_2b_89-1
# radius-type-target-run-channel-z-signal
def load_model(mode, target_val, channel, z, sig):
    
    try:
     sig = np.loadtxt('results/scores/best_scores_sgn_' + str(sig) + '_SVDD_' + str(channel) + '_bs_10000' + mode + '_ft_' +  str(target_val) + '_zdim_' + str(z) + '.txt')
     bg = np.loadtxt('results/scores/best_scores_bkg_SVDD_' + str(channel) + '_bs_10000' + mode + '_ft_' +  str(target_val) + '_zdim_' + str(z) + '.txt')
     return bg, sig
    except:
     return None, None
   

def combine_scores(mode, metric, dim_z, target_vals):


 mode = '_' + mode

 if metric == 'AUC':
   index = 0 
 elif metric == 'SR1' :
   index = 1
 elif metric == 'SR2' :
   index = 2
 elif metric == 'SR3' :
   index = 3
 elif metric == 'SR4' :
   index = 4
 else:
   sys.exit()


 #create folders to store results
 if not os.path.exists('results/combined_scores'):
    os.makedirs('results/combined_scores')

 f = open('results/combined_scores/scores' + mode + '.csv', 'w')

 writer = csv.writer(f, delimiter=',')
 writer.writerow(['Signal','Network','AUC_max','1e-2_max','1e-3_max','1e-4_max','1e-5_max','AUC_min', '1e-2_min','1e-3_min','1e-4_min','1e-5_min','AUC_avg','1e-2_avg','1e-3_avg','1e-4_avg','1e-5_avg','AUC_pr','1e-2_pr','1e-3_pr','1e-4_pr','1e-5_pr'])
 
 for channel in SIG_NAMES.keys():
    #print(SIG_NAMES[channel].keys())
    for sig_val in SIG_NAMES[channel].keys():
        # take a test point for the len
        bg, sig = load_model(mode, target_vals[0], channel, dim_z[0], sig_val)
        if not type(bg).__module__ == np.__name__ :
         continue
        tot_bg = np.zeros((len(dim_z)*len(target_vals), len(bg)))
        tot_sig = np.zeros((len(dim_z)*len(target_vals), len(sig)))
        tot_bg.fill(10000)
        tot_sig.fill(10000)
        for _target in range(len(target_vals)):
            for _z in range(len(dim_z)):
                bg, sig = load_model(mode, target_vals[_target], channel, dim_z[_z], sig_val)
                if not type(bg).__module__ == np.__name__ :
                 continue
                else:
                 tot_bg[_target*len(dim_z)+_z,:] = bg
                 tot_sig[_target*len(dim_z)+_z,:] = sig

        tot_bg = tot_bg[(tot_bg!=10000).all(axis=1)]
        tot_sig = tot_sig[(tot_sig!=10000).all(axis=1)]
        
        max_bg = np.max(tot_bg, axis=0)
        max_sig = np.max(tot_sig, axis=0)

        np.savetxt('results/combined_scores/max-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), max_bg)
        np.savetxt('results/combined_scores/max-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), max_sig)
        #print('max', max_bg, max_sig)
        pr_max = performance(max_bg, max_sig)
        print(channel, sig_val, 'max', pr_max[index])
        min_bg = np.min(tot_bg, axis=0)
        min_sig = np.min(tot_sig, axis=0)
        #print('min', min_bg, min_sig)
        np.savetxt('results/combined_scores/min-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), min_bg)
        np.savetxt('results/combined_scores/min-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), min_sig)
        pr_min = performance(min_bg, min_sig)
        print(channel, sig_val, 'min', pr_min[index])
        avg_bg = np.average(tot_bg, axis=0)
        avg_sig = np.average(tot_sig, axis=0)
        np.savetxt('results/combined_scores/avg-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), avg_bg)
        np.savetxt('results/combined_scores/avg-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), avg_sig)
        pr_avg = performance(avg_bg, avg_sig)
        print(channel, sig_val, 'avg', pr_avg[index])
        prod_bg = np.product(tot_bg, axis=0)
        prod_sig = np.product(tot_sig, axis=0)
        np.savetxt('results/combined_scores/prod-bg' + mode + '_' + metric + '_'+ channel + '-' + str(sig_val), prod_bg)
        np.savetxt('results/combined_scores/prod-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), prod_sig)
        pr_prod = performance(prod_bg, prod_sig)
        print(channel, sig_val, 'all', 'prod', pr_prod[index])
        writer.writerow([sig_val, channel, pr_max[0], pr_max[1], pr_max[2], pr_max[3], pr_max[4], pr_min[0], pr_min[1], pr_min[2], pr_min[3], pr_min[4], pr_avg[0], pr_avg[1], pr_avg[2], pr_avg[3], pr_avg[4], pr_prod[0], pr_prod[1], pr_prod[2], pr_prod[3], pr_prod[4]])
 
def combine_scores_random(mode, n_comb, N):

 print('Number of combinations ', n_comb)

 new_target_vals = random.sample(target_vals, n_comb)
 new_dim_z = random.sample(dim_z, n_comb)

 #If number of combinations is less than len()*len()
 #choose one or the another list and remove

 if n_comb < len(new_dim_z)*len(new_target_vals):
  value = random.randint(0, 1)
  if value == 0:
   fixed = new_target_vals 
   target = new_dim_z  
  else:
   target = new_target_vals
   fixed = new_dim_z
  for i in range(len(target)):
    if len(fixed)*len(target) > n_comb:
      target.pop(i)
  if value == 0:
   new_dim_z = target  
  else:
   new_target_vals = target
 
 
 f = open('results/combined_scores_rnd/scores_rnd' + mode + '_' + str(n_comb)  + '_' + str(N) + '.csv', 'w')

 writer = csv.writer(f, delimiter=',')
 writer.writerow(['Signal','Network','AUC_max','1e-2_max','1e-3_max','1e-4_max','AUC_min', '1e-2_min','1e-3_min','1e-4_min','AUC_avg','1e-2_avg','1e-3_avg','1e-4_avg','AUC_pr','1e-2_pr','1e-3_pr','1e-4_pr'])

 #counter = 0
 #done = False
 for channel in SIG_NAMES.keys():
    #print(SIG_NAMES[channel].keys())
    for sig_val in SIG_NAMES[channel].keys():
        # take a test point for the len
        #print(channel, sig_val)
        bg, sig = load_model(mode, new_target_vals[0], channel, new_dim_z[0], sig_val)

        tot_bg = np.zeros((len(new_dim_z)*len(new_target_vals), len(bg)))
        tot_sig = np.zeros((len(new_dim_z)*len(new_target_vals), len(sig)))
        for _target in range(len(new_target_vals)):
            for _z in range(len(new_dim_z)):
                bg, sig = load_model(mode, new_target_vals[_target], channel, new_dim_z[_z], sig_val)
                tot_bg[_target*len(new_dim_z)+_z,:] = bg
                tot_sig[_target*len(new_dim_z)+_z,:] = sig

        max_bg = np.max(tot_bg, axis=0)
        max_sig = np.max(tot_sig, axis=0)
        np.savetxt('results/combined_scores_rnd/max-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), max_bg)
        np.savetxt('results/combined_scores_rnd/max-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), max_sig)
        pr_max = performance(max_bg, max_sig)
        print(channel, sig_val, 'all', 'max', pr_max[index])
        min_bg = np.min(tot_bg, axis=0)
        min_sig = np.min(tot_sig, axis=0)
        np.savetxt('results/combined_scores_rnd/min-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), min_bg)
        np.savetxt('results/combined_scores_rnd/min-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), min_sig)
        pr_min = performance(min_bg, min_sig)
        print(channel, sig_val, 'all', 'min', pr_min[index])
        avg_bg = np.average(tot_bg, axis=0)
        avg_sig = np.average(tot_sig, axis=0)
        np.savetxt('results/combined_scores_rnd/avg-bg' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), avg_bg)
        np.savetxt('results/combined_scores_rnd/avg-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), avg_sig)
        pr_avg = performance(avg_bg, avg_sig)
        print(channel, sig_val, 'all', 'avg', pr_avg[index])
        prod_bg = np.product(tot_bg, axis=0)
        prod_sig = np.product(tot_sig, axis=0)
        np.savetxt('results/combined_scores_rnd/prod-bg' + mode + '_' + metric + '_'+ channel + '-' + str(sig_val), prod_bg)
        np.savetxt('results/combined_scores_rnd/prod-sig' + mode + '_' + metric + '_' + channel + '-' + str(sig_val), prod_sig)
        pr_prod = performance(prod_bg, prod_sig)
        print(channel, sig_val, 'all', 'prod', pr_prod[index])
        writer.writerow([sig_val, channel, pr_max[0], pr_max[1], pr_max[2], pr_max[3], pr_min[0], pr_min[1], pr_min[2], pr_min[3], pr_avg[0], pr_avg[1], pr_avg[2], pr_avg[3], pr_prod[0], pr_prod[1], pr_prod[2], pr_prod[3]])
        #counter += 1
        #if counter > n_comb:
        # done = True
        # break
    #if done:
    # break

def main(Flags):

 mode = Flags.mode
 metric = Flags.metric

 dim_z = Flags.dim  
 ft = Flags.fixed_target

 N = Flags.realization
 n_combs = Flags.combinations

 if n_combs > 0:
   for i in range(5):
    combine_scores_random(mode, n_combs, i+1) 
 else:
   combine_scores(mode, metric, dim_z, ft)

if __name__ == '__main__':

 parser = argparse.ArgumentParser() 

 parser.add_argument('--combinations', type=int, default=0, help='Number of combinations')

 parser.add_argument('--mode', type=str, default='ordered')

 parser.add_argument('--metric', type=str, default='AUC')

 parser.add_argument('--realization', type=int, default=1)

 parser.add_argument('--dim', nargs='+', type=int, default=[5,8,13,21,34,55,89,144,233], help='Latent space dim.')

 parser.add_argument('--fixed_target', nargs='+', type=int, default=[0,1,2,3,4,10,25])

 Flags, unparsed = parser.parse_known_args()

 main(Flags)



