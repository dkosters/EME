#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import glob
from tqdm import tqdm
import sys, argparse
    
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import csv
import pandas as pd

xcoord = ['Ato4l', 'hChToTauNu', 'hToTauTau', 'leptoquark']

def to_csv(Flags):
    
    files_all = []
    files = []

    mode = Flags.mode

    files_all = glob.glob('results/metrics/*') 

    sf = ''
    if mode == 'ordered':
     files = [s for s in files_all if "ordered" in s]
     sf = '_ordered'
    elif mode == 'categorical':
     files = [s for s in files_all if 'cat' in s]
     sf = '_cat' 
    elif mode == 'polina':
     files = [s for s in files_all if 'polina' in s]
     sf = '_polina' 
    else:
     files = [s for s in files_all if "cat" not in s and "ordered" not in s and "polina" not in s] 

    files_sufix= [] 
    for fname in tqdm(files):      
      suffix = fname.split('/')[-1]
      suffix = suffix.split('.txt')[0]
      suffix = suffix.split('_')[2:]
      files_sufix.append('_'.join(suffix))
    
    f = open('results/scores/scores' + sf + '.csv', 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Signal','Channel','Model','AUC','1e-2','1e-3','1e-4','1e-5']) 
    counter = 0
    files_sufix = list(set(files_sufix))
    for fname in tqdm(files_sufix):
      suffix = fname.split('_')
      if mode == 'ordered':
       suffix.remove('ordered')
      if mode == 'categorical': 
       suffix.remove('cat')
      if mode == 'polina': 
       suffix.remove('polina')
      if len(suffix) == 10:
       channel = '_'.join(suffix[:4])
       name = '_'.join(suffix[4:])
      elif len(suffix) == 9:
       channel = '_'.join(suffix[:3])
       name = '_'.join(suffix[3:])
      elif len(suffix) == 8:
       channel = '_'.join(suffix[:2])
       name = '_'.join(suffix[2:])
      elif len(suffix) == 7:
       channel = '_'.join(suffix[:1])
       name = '_'.join(suffix[1:])
      else:
       continue

      fp = open('results/metrics/AUC_SVDD_' + fname + '.txt', 'r')
      lines = fp.read().splitlines()
      auc = [float(i) for i in lines]
      fp = open('results/metrics/Epsilon1_SVDD_' + fname + '.txt', 'r')
      lines = fp.read().splitlines()
      epsilon1 = [float(i) for i in lines]
      fp = open('results/metrics/Epsilon2_SVDD_' + fname + '.txt', 'r')
      lines = fp.read().splitlines()
      epsilon2 = [float(i) for i in lines]
      fp = open('results/metrics/Epsilon3_SVDD_' + fname + '.txt', 'r')
      lines = fp.read().splitlines()
      epsilon3 = [float(i) for i in lines]
      fp = open('results/metrics/Epsilon4_SVDD_' + fname + '.txt', 'r')
      lines = fp.read().splitlines()
      epsilon4 = [float(i) for i in lines]

      for i in range(len(xcoord)):
         writer.writerow([xcoord[i], channel, name, auc[i], epsilon1[i], epsilon2[i], epsilon3[i], epsilon4[i]]) 


#      print (auc, epsilon1, epsilon2, epsilon3)

def make_subplots(best, auc):

    limit = 8
    
    ch = []
    lb = []
    for i in range(len(best)):  
     ch.append(best[i][0])
     lb.append(best[i][1])              
    xcoord = np.array(xcoord)

    for p in auc:
     values = p[limit:]
     #print (values)
     #plt.scatter(np.arange(8), values)
     ax = plt.gca()
     #print(xcoord.shape, values.shape)
     plt.scatter(xcoord, values)
     plt.ylabel("AUC", fontsize=16)
     plt.title(channel, fontsize=14)
     ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
     plt.xticks(rotation=90)
     #plt.minorticks_on()
     plt.ylim(0.5, 1)
     #plt.legend(ch, lb)
     #plt.text(0.9, 0.9, str(ch[0]) + ' ' + str(lb[0]) )
     
    plt.show()
    #plt.savefig('AUC_' + channel +  '.pdf', bbox_inches = "tight")  

def convert(channel):

    files = []

    #files = glob.glob('../results_'+ channel + '/*') 

    files = glob.glob('../results_' + channel + '/AUC_energy*') 
    auc = []
    #coords = []
    for filename in tqdm(files, leave=False):
       coords = []
       #print(filename)
       fp = open(filename, 'r')
       lines = fp.read().splitlines()
       lines = [float(i) for i in lines] 
       #line = fp.readline()
       #while line:
       # line = fp.readline()
       #print(lines)
       coord = filename.split('/')[2]
       coord = coord.split('.txt')[0]
       #print (coord)
       dim = float(coord.split('_')[2])
       lr = float(coord.split('_')[3])
       batch = float(coord.split('_')[4])
       alpha1 = float(coord.split('_')[5])
       alpha2 = float(coord.split('_')[6])
       epoch = float(coord.split('_')[7])
       dim1 = float(coord.split('_')[8])
       dim2 = float(coord.split('_')[9])
       coords.append(dim)
       coords.append(lr)
       coords.append(batch)
       coords.append(alpha1)
       coords.append(alpha2)
       coords.append(epoch)
       coords.append(dim1)
       coords.append(dim2)
       for elem in lines:
         coords.append(elem)  
       #print(coords)
       auc.append(coords)

    return np.array(auc) 

def make_box_plots(Flags):

 sufix = Flags.mode

 if sufix == "cat" or sufix == "ordered" or sufix == 'polina':
  sufix = '_' + sufix 

 SVDD = pd.read_csv('results/scores/scores' + sufix + '.csv')

 SVDDModels = list(SVDD['Model'].unique())
 
 names = ['Signal', 'Channel', 'Model', 'AUC', '1e-2', '1e-3', '1e-4', '1e-5']

 AllCombined = np.array(SVDD[names])

 AllCombined = pd.DataFrame(AllCombined, columns=names)

 AllSigs = list(AllCombined['Signal'].unique())

 counter = 0
 yvs = []
 ListAUC = []
 List1m2 = []
 List1m3 = []
 List1m4 = []
 List1m5 = []
 my_yticks = []
 my_ynames = []
 mycolors = []

 colordict = {'1l_4':'C0', '1l_8':'C1', '1l_16':'C2', '1l_32':'C3', '1l_64':'C4', '1l_128':'C5', '1l_512':'C6', '2l_256_128': 'C7', '3l_512_256_128': 'C8'}

 plt.figure(figsize=(9, 12))
 for sig in np.sort(AllSigs)[::-1]:
    tmp_df = AllCombined[AllCombined['Signal'] == sig]
    #print(sig, tmp_df)
    channels = list(tmp_df.Channel.unique())
    lc = len(channels)
    
    tmp_yn = []
    for i, chan in enumerate(channels):
     tmp_df = AllCombined[(AllCombined['Signal'] == sig) & (AllCombined['Channel'] == chan)]

     #print(chan)
     #print(tmp_df['AUC'].max())        
     ListAUC.append(tmp_df['AUC'])
     List1m2.append(tmp_df['1e-2'])
     List1m3.append(tmp_df['1e-3'])
     List1m4.append(tmp_df['1e-5'])
     yvs.append(counter)
     tmp_yn.append(counter)
     counter += 1
     mycolors.append(colordict[chan])
    counter += 4
    my_yticks.append(np.mean(tmp_yn))
    my_ynames.append(sig.replace('_', '$\_$'))

 #print(np.array(ListAUC).shape)

 plt.subplot(1, 4, 1)    
 bbox1 = plt.boxplot(np.array(ListAUC).T,
            vert=False,
            manage_ticks=False,
            patch_artist=True,
            medianprops={'color':'k'},
            positions=yvs
           )

 plt.xlim(0.5, 1)
 plt.yticks(my_yticks, my_ynames)
 plt.xlabel('AUC')

 for patch, flier, color in zip(bbox1['boxes'], bbox1['fliers'], mycolors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markersize(3)
    flier.set_markeredgecolor(color)

 plt.ylim(min(yvs)-1, max(yvs)+1)

 plt.subplot(1, 4, 2)    
 bbox1 = plt.boxplot(np.array(List1m2).T,
            vert=False,
            manage_ticks=False,
            patch_artist=True,
            medianprops={'color':'k'},
            positions=yvs
           )
 plt.xscale('log')
 plt.xlim(1e-4, 1)
 plt.yticks(my_yticks, [])
 plt.xlabel(r'$\epsilon_S(\epsilon_B = 10^{-2})$')
 for patch, flier, color in zip(bbox1['boxes'], bbox1['fliers'], mycolors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markersize(3)
    flier.set_markeredgecolor(color)
 plt.ylim(min(yvs)-1, max(yvs)+1)

 plt.subplot(1, 4, 3)    
 bbox1 = plt.boxplot(np.array(List1m3).T,
            vert=False,
            manage_ticks=False,
            patch_artist=True,
            medianprops={'color':'k'},
            positions=yvs
           )
 plt.xscale('log')
 plt.xlim(1e-4, 1)
 plt.yticks(my_yticks, [])
 plt.xlabel(r'$\epsilon_S(\epsilon_B = 10^{-3})$')
 for patch, flier, color in zip(bbox1['boxes'], bbox1['fliers'], mycolors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markersize(3)
    flier.set_markeredgecolor(color)
 plt.ylim(min(yvs)-1, max(yvs)+1)

 plt.subplot(1, 4, 4)    
 bbox1 = plt.boxplot(np.array(List1m4).T,
            vert=False,
            manage_ticks=False,
            patch_artist=True,
            medianprops={'color':'k'},
            positions=yvs
           )
 plt.xscale('log')
 plt.xlim(1e-4, 1)
 plt.yticks(my_yticks, [])
 for patch, flier, color in zip(bbox1['boxes'], bbox1['fliers'], mycolors):
    patch.set_facecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markerfacecolor(color)
    flier.set_markersize(3)
    flier.set_markeredgecolor(color)
 plt.xlabel(r'$\epsilon_S(\epsilon_B = 10^{-5})$')
 plt.ylim(min(yvs)-1, max(yvs)+1)


 legend_elements = [
                   Patch(facecolor='C0', edgecolor='k',
                         label='1l_4'),
                   Patch(facecolor='C1', edgecolor='k',
                         label='1l_8'),
                   Patch(facecolor='C2', edgecolor='k',
                         label='1l_16'),
                   Patch(facecolor='C3', edgecolor='k',
                         label='1l_32'),
                   Patch(facecolor='C4', edgecolor='k',
                         label='1l_64'),
                   Patch(facecolor='C5', edgecolor='k',
                         label='1l_128'),
                   Patch(facecolor='C6', edgecolor='k',
                         label='1l_512'),
                   Patch(facecolor='C7', edgecolor='k',
                         label='2l_256_128'),
                   Patch(facecolor='C8', edgecolor='k',
                         label='3l_512_256_128'),
                  ] 

 plt.legend(handles=legend_elements, 
            loc=(-4, -0.13), 
           labelspacing=1,
           ncol=7,
           facecolor='white',
           framealpha=1,
#            markerscale=100,
           frameon=True)
    
 plt.suptitle('Analysis of all models on all signals in the\n$40mhz$ $Challenge$ $Hackathon$ $Data$', y=0.92)
 plt.savefig('figures/AllModelsAllSignals' + sufix + '.pdf', bbox_inches='tight')
 #plt.show()

#Performance versus complexity
#y-axis net model, y-axis metrics => 4 panels 
#color different channel
def make_combined_plots(Flags):

 sufix = Flags.mode

 if sufix == "cat" or sufix == "ordered" or sufix == 'polina':
  sufix = '_' + sufix 

 SVDD = pd.read_csv('results/combined_scores/scores' + sufix + '.csv')

 #print(SVDD.head())

 #SVDDModels = list(SVDD['Model'].unique())

 #print(len(SVDDModels))
 
 names = ['Signal', 'Network', 'AUC_max', '1e-2_max', '1e-3_max', '1e-4_max', '1e-5_max', 'AUC_min', '1e-2_min', '1e-3_min', '1e-4_min', '1e-5_min', 'AUC_avg', '1e-2_avg', '1e-3_avg', '1e-4_avg', '1e-5_avg', 'AUC_pr', '1e-2_pr', '1e-3_pr', '1e-4_pr', '1e-5_pr']

 AllCombined = np.array(SVDD[names])

 #print(AllCombined.shape)
 AllCombined = pd.DataFrame(AllCombined, columns=names)

 AllSigs = list(AllCombined['Signal'].unique())

 counter = 0
 yvs = []
 ListAUC = []
 List1m2 = []
 List1m3 = []
 List1m4 = []
 List1m5 = []
 my_yticks = []
 my_ynames = []
 mycolors = []

 colordict = {'1l_4':'C0', '1l_8':'C1', '1l_16':'C2', '1l_32':'C3', '1l_64':'C4', '1l_128':'C5', '1l_512':'C6', '2l_256_128': 'C7', '3l_512_256_128': 'C8'}

 fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), sharex=False,sharey=False, constrained_layout=True)    
 inx = 0
 for sig in np.sort(AllSigs)[::-1]:
    #print(sig, inx)
    tmp_df = AllCombined[AllCombined['Signal'] == sig]
    tmp_df = tmp_df.sort_values(by = 'Network')
    x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
    auc_max = tmp_df[["AUC_max"]].to_numpy().squeeze()
    auc_min = tmp_df[["AUC_min"]].to_numpy().squeeze()
    auc_avg = tmp_df[["AUC_avg"]].to_numpy().squeeze()
    auc_pr = tmp_df[["AUC_pr"]].to_numpy().squeeze()
    sr1_max = tmp_df[["1e-2_max"]].to_numpy().squeeze()
    sr1_min = tmp_df[["1e-2_min"]].to_numpy().squeeze()
    sr1_avg = tmp_df[["1e-2_avg"]].to_numpy().squeeze()
    sr1_pr = tmp_df[["1e-2_pr"]].to_numpy().squeeze()
    sr2_max = tmp_df[["1e-3_max"]].to_numpy().squeeze()
    sr2_min = tmp_df[["1e-3_min"]].to_numpy().squeeze()
    sr2_avg = tmp_df[["1e-3_avg"]].to_numpy().squeeze()
    sr2_pr = tmp_df[["1e-3_pr"]].to_numpy().squeeze()
    sr3_max = tmp_df[["1e-5_max"]].to_numpy().squeeze()
    sr3_min = tmp_df[["1e-5_min"]].to_numpy().squeeze()
    sr3_avg = tmp_df[["1e-5_avg"]].to_numpy().squeeze()
    sr3_pr = tmp_df[["1e-5_pr"]].to_numpy().squeeze()
    axs[inx,0].scatter(x, auc_max, c='r',marker="o")
    axs[inx,0].scatter(x, auc_min, c='b',marker="o")
    axs[inx,0].scatter(x, auc_avg, c='g',marker="o")
    axs[inx,0].scatter(x, auc_pr, c='m',marker="o")
    axs[inx,0].set_ylim([0.8, 1])
    axs[inx,0].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,0].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,1].scatter(x,sr1_max, c='r',marker="o")
    axs[inx,1].scatter(x,sr1_min, c='b',marker="o")
    axs[inx,1].scatter(x,sr1_avg, c='g',marker="o")
    axs[inx,1].scatter(x,sr1_pr, c='m',marker="o")
    axs[inx,1].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,1].set_ylim([0.1,1])
    axs[inx,1].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,2].scatter(x,sr2_max, c='r',marker="o")
    axs[inx,2].scatter(x,sr2_min, c='b',marker="o")
    axs[inx,2].scatter(x,sr2_avg, c='g',marker="o")
    axs[inx,2].scatter(x,sr2_pr, c='m',marker="o")
    axs[inx,2].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,2].set_yscale('log')
    axs[inx,2].set_ylim([1e-2,1])
    axs[inx,2].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,3].scatter(x,sr3_max, c='r',marker="o")
    axs[inx,3].scatter(x,sr3_min, c='b',marker="o")
    axs[inx,3].scatter(x,sr3_avg, c='g',marker="o")
    axs[inx,3].scatter(x,sr3_pr, c='m',marker="o")
    axs[inx,3].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,3].set_yscale('log')
    axs[inx,3].set_ylim([1e-3,1])
    axs[inx,3].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)

    if inx == 0:
      axs[inx,0].set_title("AUC",fontsize=10)    
      axs[inx,1].set_title((r'$\epsilon_S(\epsilon_B = 10^{-2})$'),fontsize=10)  
      axs[inx,2].set_title((r'$\epsilon_S(\epsilon_B = 10^{-3})$'),fontsize=10)  
      axs[inx,3].set_title((r'$\epsilon_S(\epsilon_B = 10^{-5})$'),fontsize=10)  
    
    if inx==0:
      axs[inx,0].set_ylabel('leptoquark', fontsize=12)
    elif inx==1:
      axs[inx,0].set_ylabel('hToTauTau', fontsize=12)
    elif inx==2:
      axs[inx,0].set_ylabel('hChToTauNu', fontsize=12)
    else:
      axs[inx,0].set_ylabel('Ato4l', fontsize=12) 
      axs[inx,0].set(xlabel='Network')  
      axs[inx,1].set(xlabel='Network')    
      axs[inx,2].set(xlabel='Network') 
      axs[inx,3].set(xlabel='Network') 

    inx += 1
    
 fig.suptitle('SVDD combined scores for 40 mhz challenge', y=1., fontsize=12)

 legend_elements = [Patch(facecolor='r', edgecolor='k',
                         label='OR'),
                   Patch(facecolor='b', edgecolor='k',
                         label='AND'),
                   Patch(facecolor='g', edgecolor='k',
                         label='AVG'),
                   Patch(facecolor='m', edgecolor='k',
                         label='PROD')
                  ]
 plt.subplots_adjust(top=0.95, left=0.1, right=0.98, bottom=0.1, wspace = 0.3)  # create some space below
 
 plt.legend(handles=legend_elements, 
            loc=(-2.5, -0.5), 
#           loc='lower center',
#           bbox_to_anchor=(0.5, 0), 
           labelspacing=1,
           ncol=4,
           facecolor='white',
           framealpha=1,
#            markerscale=100,
           frameon=True)
 

# plt.tight_layout()
 plt.savefig('figures/Combined_AllModelsAllSignals' + sufix + '.pdf', bbox_inches='tight')
 #plt.show()
    

def make_combined_rnd_plots(Flags):

 sufix = Flags.mode
 n_comb = Flags.n_comb

 if sufix == "cat" or sufix == "ordered" or sufix == 'polina':
  sufix = '_' + sufix 

 SVDD = []

 #print(SVDD.head())

 #SVDDModels = list(SVDD['Model'].unique())

 #print(len(SVDDModels))
 AllCombined = []
 names = ['Signal', 'Network', 'AUC_max', '1e-2_max', '1e-3_max', '1e-4_max', '1e-5_max', 'AUC_min', '1e-2_min', '1e-3_min', '1e-4_min', '1e-5_min', 'AUC_avg', '1e-2_avg', '1e-3_avg', '1e-4_avg', '1e-5_avg', 'AUC_pr', '1e-2_pr', '1e-3_pr', '1e-4_pr','1e-5_pr']

 for i in range(5): 
  SVDD = pd.read_csv('results/combined_scores_rnd/scores_rnd' + sufix + '_' + str(n_comb) + '_' + str(i+1) + '.csv')
  AllCombined.append(np.array(SVDD[names]))

 AllCombined = np.array(AllCombined)

 #print(AllCombined.shape)
 AllCombined_l = []
 for i in range(5):
  AllCombined_l.append(pd.DataFrame(AllCombined[i], columns=names))
  
 AllSigs = list(AllCombined_l[0]['Signal'].unique())

 counter = 0
 yvs = []
 ListAUC = []
 List1m2 = []
 List1m3 = []
 List1m4 = []
 List1m5 = []
 my_yticks = []
 my_ynames = []
 mycolors = []
 
 colordict = {'1l_4':'C0', '1l_8':'C1', '1l_16':'C2', '1l_32':'C3', '1l_64':'C4', '1l_128':'C5', '1l_512':'C6', '2l_256_128': 'C7', '3l_512_256_128': 'C8'}

 fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), sharex=False,sharey=False, constrained_layout=True)    

 inx = 0
 for sig in np.sort(AllSigs)[::-1]:
    #print(sig, inx)
    auc_max = []
    auc_min = []
    auc_avg = []
    auc_pr = []
    sr1_max = []
    sr1_min = []
    sr1_avg = []
    sr1_pr = []
    sr2_max = []
    sr2_min = []
    sr2_avg = []
    sr2_pr = []
    sr3_max = []
    sr3_min = []
    sr3_avg = []
    sr3_pr = []
    #Loop over realizations
    for i in range(5):
     tmp_df = AllCombined_l[i][AllCombined_l[i]['Signal'] == sig]
     x = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
     auc_max.append(tmp_df[["AUC_max"]].to_numpy().squeeze())
     auc_min.append(tmp_df[["AUC_min"]].to_numpy().squeeze())
     auc_avg.append(tmp_df[["AUC_avg"]].to_numpy().squeeze())
     auc_pr.append(tmp_df[["AUC_pr"]].to_numpy().squeeze())
     sr1_max.append(tmp_df[["1e-2_max"]].to_numpy().squeeze())
     sr1_min.append(tmp_df[["1e-2_min"]].to_numpy().squeeze())
     sr1_avg.append(tmp_df[["1e-2_avg"]].to_numpy().squeeze())
     sr1_pr.append(tmp_df[["1e-2_pr"]].to_numpy().squeeze())
     sr2_max.append(tmp_df[["1e-3_max"]].to_numpy().squeeze())
     sr2_min.append(tmp_df[["1e-3_min"]].to_numpy().squeeze())
     sr2_avg.append(tmp_df[["1e-3_avg"]].to_numpy().squeeze())
     sr2_pr.append(tmp_df[["1e-3_pr"]].to_numpy().squeeze())
     sr3_max.append(tmp_df[["1e-4_max"]].to_numpy().squeeze())
     sr3_min.append(tmp_df[["1e-4_min"]].to_numpy().squeeze())
     sr3_avg.append(tmp_df[["1e-4_avg"]].to_numpy().squeeze())
     sr3_pr.append(tmp_df[["1e-4_pr"]].to_numpy().squeeze())
    auc_max = np.array(auc_max)
    auc_min = np.array(auc_min)
    auc_avg = np.array(auc_avg)
    auc_pr = np.array(auc_pr)
    sr1_max = np.array(sr1_max)
    sr1_min = np.array(sr1_min)
    sr1_avg = np.array(sr1_avg)
    sr1_pr = np.array(sr1_pr)
    sr2_max = np.array(sr2_max)
    sr2_min = np.array(sr2_min)
    sr2_avg = np.array(sr2_avg)
    sr2_pr = np.array(sr2_pr)
    sr3_max = np.array(sr3_max)
    sr3_min = np.array(sr3_min)
    sr3_avg = np.array(sr3_avg)
    sr3_pr = np.array(sr3_pr)
    auc_max_mean = np.mean(auc_max, axis = 0)
    auc_max_std = [np.std(auc_max[:,i]) for i in range(9)]
    auc_min_mean = np.mean(auc_min, axis = 0)
    auc_min_std = [np.std(auc_min[:,i]) for i in range(9)]
    auc_avg_mean = np.mean(auc_avg, axis = 0)
    auc_avg_std = [np.std(auc_avg[:,i]) for i in range(9)]
    auc_pr_mean = np.mean(auc_pr, axis = 0)
    auc_pr_std = [np.std(auc_pr[:,i]) for i in range(9)]
    sr1_max_mean = np.mean(sr1_max, axis = 0)
    sr1_max_std = [np.std(sr1_max[:,i]) for i in range(9)]
    sr1_min_mean = np.mean(sr1_min, axis = 0)
    sr1_min_std = [np.std(sr1_min[:,i]) for i in range(9)]
    sr1_avg_mean = np.mean(sr1_avg, axis = 0)
    sr1_avg_std = [np.std(sr1_avg[:,i]) for i in range(9)]
    sr1_pr_mean = np.mean(sr1_pr, axis = 0)
    sr1_pr_std = [np.std(sr1_pr[:,i]) for i in range(9)]
    sr2_max_mean = np.mean(sr2_max, axis = 0)
    sr2_max_std = [np.std(sr2_max[:,i]) for i in range(9)]
    sr2_min_mean = np.mean(sr2_min, axis = 0)
    sr2_min_std = [np.std(sr2_min[:,i]) for i in range(9)]
    sr2_avg_mean = np.mean(sr2_avg, axis = 0)
    sr2_avg_std = [np.std(sr2_avg[:,i]) for i in range(9)]
    sr2_pr_mean = np.mean(sr2_pr, axis = 0)
    sr2_pr_std = [np.std(sr2_pr[:,i]) for i in range(9)]
    sr3_max_mean = np.mean(sr3_max, axis = 0)
    sr3_max_std = [np.std(sr3_max[:,i]) for i in range(9)]
    sr3_min_mean = np.mean(sr3_min, axis = 0)
    sr3_min_std = [np.std(sr3_min[:,i]) for i in range(9)]
    sr3_avg_mean = np.mean(sr3_avg, axis = 0)
    sr3_avg_std = [np.std(sr3_avg[:,i]) for i in range(9)]
    sr3_pr_mean = np.mean(sr3_pr, axis = 0)
    sr3_pr_std = [np.std(sr3_pr[:,i]) for i in range(9)]
    #I have to reverse order to plot it
    auc_max_mean = np.flip(auc_max_mean)
    auc_max_std = np.flip(auc_max_std)
    auc_min_mean = np.flip(auc_min_mean)
    auc_min_std = np.flip(auc_min_std)
    auc_avg_mean = np.flip(auc_avg_mean)
    auc_avg_std = np.flip(auc_avg_std)
    auc_pr_mean = np.flip(auc_pr_mean)
    auc_pr_std = np.flip(auc_pr_std)
    sr1_max_mean = np.flip(sr1_max_mean) 
    sr1_max_std = np.flip(sr1_max_std)
    sr1_min_mean = np.flip(sr1_min_mean) 
    sr1_min_std = np.flip(sr1_min_std)
    sr1_avg_mean = np.flip(sr1_avg_mean) 
    sr1_avg_std = np.flip(sr1_avg_std) 
    sr1_pr_mean = np.flip(sr1_pr_mean) 
    sr1_pr_std = np.flip(sr1_pr_std) 
    sr2_max_mean = np.flip(sr2_max_mean) 
    sr2_max_std = np.flip(sr2_max_std) 
    sr2_min_mean = np.flip(sr2_min_mean) 
    sr2_min_std = np.flip(sr2_min_std)
    sr2_avg_mean = np.flip(sr2_avg_mean) 
    sr2_avg_std = np.flip(sr2_avg_std) 
    sr2_pr_mean = np.flip(sr2_pr_mean) 
    sr2_pr_std = np.flip(sr2_pr_std) 
    sr3_max_mean = np.flip(sr3_max_mean) 
    sr3_max_std = np.flip(sr3_max_std)
    sr3_min_mean = np.flip(sr3_min_mean) 
    sr3_min_std = np.flip(sr3_min_std) 
    sr3_avg_mean = np.flip(sr3_avg_mean) 
    sr3_avg_std = np.flip(sr3_avg_std) 
    sr3_pr_mean = np.flip(sr3_pr_mean) 
    sr3_pr_std = np.flip(sr3_pr_std)
    axs[inx,0].errorbar(x, auc_max_mean, auc_max_std, linestyle='None', c='r', marker='o')
    axs[inx,0].errorbar(x, auc_min_mean, auc_min_std, linestyle='None', c='b', marker='o')
    axs[inx,0].errorbar(x, auc_avg_mean, auc_avg_std, linestyle='None', c='g', marker='o')
    axs[inx,0].errorbar(x, auc_pr_mean, auc_pr_std, linestyle='None', c='m', marker='o')
    axs[inx,0].set_ylim([0.8, 1])
    axs[inx,0].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,0].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,1].errorbar(x, sr1_max_mean, sr1_max_std, linestyle='None', c='r', marker='o')
    axs[inx,1].errorbar(x, sr1_min_mean, sr1_min_std, linestyle='None', c='b', marker='o')
    axs[inx,1].errorbar(x, sr1_avg_mean, sr1_avg_std, linestyle='None', c='g', marker='o')
    axs[inx,1].errorbar(x, sr1_pr_mean, sr1_pr_std, linestyle='None', c='m', marker='o')
    axs[inx,1].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,1].set_ylim([0.1,1])
    axs[inx,1].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,2].errorbar(x, sr2_max_mean, sr2_max_std, linestyle='None', c='r', marker='o')
    axs[inx,2].errorbar(x, sr2_min_mean, sr2_min_std, linestyle='None', c='b', marker='o')
    axs[inx,2].errorbar(x, sr2_avg_mean, sr2_avg_std, linestyle='None', c='g', marker='o')
    axs[inx,2].errorbar(x, sr2_pr_mean, sr2_pr_std, linestyle='None', c='m', marker='o')
    axs[inx,2].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,2].set_yscale('log')
    axs[inx,2].set_ylim([1e-2,1])
    axs[inx,2].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)
    axs[inx,3].errorbar(x, sr3_max_mean, sr3_max_std, linestyle='None', c='r', marker='o')
    axs[inx,3].errorbar(x, sr3_min_mean, sr3_min_std, linestyle='None', c='b', marker='o')
    axs[inx,3].errorbar(x, sr3_avg_mean, sr3_avg_std, linestyle='None', c='g', marker='o')
    axs[inx,3].errorbar(x, sr3_pr_mean, sr3_pr_std, linestyle='None', c='m', marker='o')
    axs[inx,3].set_xticks([1,2,3,4,5,6,7,8,9])
    axs[inx,3].set_yscale('log')
    axs[inx,3].set_ylim([1e-3,1])
    axs[inx,3].set_xticklabels(['4', '8', '16', '32', '64', '128', '512', '2l', '3l'], fontsize=7)

    if inx == 0:
      axs[inx,0].set_title("AUC",fontsize=10)    
      axs[inx,1].set_title((r'$\epsilon_S(\epsilon_B = 10^{-2})$'),fontsize=10)  
      axs[inx,2].set_title((r'$\epsilon_S(\epsilon_B = 10^{-3})$'),fontsize=10)  
      axs[inx,3].set_title((r'$\epsilon_S(\epsilon_B = 10^{-5})$'),fontsize=10)  
    
    if inx==0:
      axs[inx,0].set_ylabel('leptoquark', fontsize=12)
    elif inx==1:
      axs[inx,0].set_ylabel('hToTauTau', fontsize=12)
    elif inx==2:
      axs[inx,0].set_ylabel('hChToTauNu', fontsize=12)
    else:
      axs[inx,0].set_ylabel('Ato4l', fontsize=12) 
      axs[inx,0].set(xlabel='Network')  
      axs[inx,1].set(xlabel='Network')    
      axs[inx,2].set(xlabel='Network') 
      axs[inx,3].set(xlabel='Network') 

    inx += 1
    
 fig.suptitle('SVDD three combined scores for 40 mhz challenge', y=1., fontsize=12)

 legend_elements = [Patch(facecolor='r', edgecolor='k',
                         label='OR'),
                   Patch(facecolor='b', edgecolor='k',
                         label='AND'),
                   Patch(facecolor='g', edgecolor='k',
                         label='AVG'),
                   Patch(facecolor='m', edgecolor='k',
                         label='PROD')
                  ]
 plt.subplots_adjust(top=0.95, left=0.1, right=0.98, bottom=0.1, wspace = 0.3)  # create some space below
 
 plt.legend(handles=legend_elements, 
            loc=(-2.5, -0.5), 
#           loc='lower center',
#           bbox_to_anchor=(0.5, 0), 
           labelspacing=1,
           ncol=4,
           facecolor='white',
           framealpha=1,
#            markerscale=100,
           frameon=True)
 
# plt.tight_layout()
 plt.savefig('figures/Combined_three_ModelsASignals' + sufix + '.pdf', bbox_inches='tight')
              
def main(Flags):

 if not os.path.exists('figures'):
    os.makedirs('figures')     

 to_csv(Flags)
 
if __name__ == '__main__':

 parser = argparse.ArgumentParser() 

 parser.add_argument('--mode', type=str, default='ordered', help='Type of encoding')

 parser.add_argument('--convert_to_csv', type=str, default='True', help='Merge all the scores to a csv file for plotting')

 parser.add_argument('--make_box_plots', type=str, default='True', help='Make box and whisker plots')

 parser.add_argument('--make_combination_plots', type=str, default='False', help='Make scores combinations plots')

 parser.add_argument('--make_random_combination_plots', type=str, default='False', help='Make scores random combinations plots')

 parser.add_argument('--n_comb', type=int, default=2, help='Number of combinations')   
 
 Flags, unparsed = parser.parse_known_args()

 #to csv
 if Flags.convert_to_csv == 'True': 
  main(Flags)  

 if Flags.make_box_plots == 'True': 
  make_box_plots(Flags)

 if Flags.make_combination_plots == 'True': 
  make_combined_plots(Flags)  

 if Flags.make_random_combination_plots == 'True': 
  make_combined_rnd_plots(Flags)
