# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import glob
from tqdm import tqdm
from sklearn.neighbors import KernelDensity, DistanceMetric
import sys  
import re
import math
import itertools
from pathlib import Path

import tensorflow as tf

from keras.models import Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers
from scipy.stats import multivariate_normal
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import log_loss, roc_curve
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler 
from sklearn.metrics import roc_auc_score

import h5py

import numpy as np
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import random

from tensorflow.keras.utils import to_categorical

from joblib import Parallel, delayed
import multiprocessing

import time
import os, re, os.path

import ast

import model
from model import VariationalAutoencoderModel
from dataloader import unpack, unpack_ordered, unpack_polina

home = os.getcwd()


"""
Ato4l : 1
background : 2
hChToTauNu : 3
hToTauTau : 4
leptoquark : 5
"""

sgn_dict = {'Ato4l': 1, 'hChToTauNu': 3, 'hToTauTau': 4, 'leptoquark': 5}

def reset_keras():
    K.clear_session()
    #For TF < 1*
    #sess = K.get_session()
    #For TF 2*
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
#tf.compat.v1.keras.backend.get_session()
    return sess

def normalize(data, bg_norm_min = False, bg_norm_max = False):
    norm_min = []
    norm_max = []
    
    def norm_col(col):
        _min = 0
        _max = 0

        if bg_norm_min == False:
            _min = np.mean(data[:, col])
        else:
            _min = bg_norm_min[col]
            
        if bg_norm_max == False:
            _max = np.std(data[:, col])
        else:
            _max = bg_norm_max[col]
        if _max != 0 or _min != 0:
            data[:, col] = (data[:, col] - _min) / np.max((1,_max))
        return _min, _max
    
    # Do not normalize the weights
    for i in range(0, data.shape[1]):
        col_min, col_max = norm_col(i)
        norm_min.append(col_min)
        norm_max.append(col_max)
        
    return data, np.array(norm_min), np.array(norm_max)

def train(dataset_len, data_dim, training_data, Flags):

    cat = Flags.categorical

    mode = Flags.mode

    if mode == 'ordered' or mode == 'polina':
     name = Flags.name + '_bs_' + str(Flags.batch) + '_' + mode
    else:
     if cat == 'True': 
      name = Flags.name + '_bs_' + str(Flags.batch) + '_cat' 
     else:
      name = Flags.name + '_bs_' + str(Flags.batch)

    dim_z = Flags.dim  

    ft = Flags.fixed_target
    
    for c in tqdm(ft):
      for _z in tqdm(dim_z):
        print('training ft= ', c, ' dimz = ', _z )
        with tf.device("/gpu:0"):
            #sess = tf.Session()
            #K.set_session(sess)
            sess = tf.compat.v1.Session()
            tf.compat.v1.keras.backend.set_session(sess)

            model_name = name + '_ft_' + str(c) + '_zdim_' + str(_z)
            model = VariationalAutoencoderModel(model_name, data_dim, dataset_len, _z, c, mode=mode, verbose=True)
 
            model.train_model(training_data)

def test(dataset_len, data_dim, etype, training_data, test_data, Flags):

    cat = Flags.categorical

    mode = Flags.mode

    if mode == 'ordered' or mode == 'polina':
     name = Flags.name + '_bs_' + str(Flags.batch) + '_' + mode
    else:
     if cat == 'True': 
      name = Flags.name + '_bs_' + str(Flags.batch) + '_cat'  
     else:
      name = Flags.name + '_bs_' + str(Flags.batch)  

    dim_z = Flags.dim 
   
    ft = Flags.fixed_target

    for c in tqdm(ft):
      #name += 'c'
      for _z in dim_z:
        with tf.device("/gpu:0"):
            sess = tf.compat.v1.Session()
            tf.compat.v1.keras.backend.set_session(sess)

            model_name = name + '_ft_' + str(c) + '_zdim_' + str(_z)
            model = VariationalAutoencoderModel(model_name, data_dim, dataset_len, _z, c, mode=mode, verbose=True)
 
            model.load_weights('models/' + model_name + '.h5')

            #Evaluate radius for training with output r_max
            r_max = model.evaluate_radius_max(training_data)
            
            if r_max == 0.0:
              r_max = 1.e-35
            #
            scores = model.evaluate_radius(test_data, r_max)

            AUC = []
            Epsilon1 = []
            Epsilon2 = []
            Epsilon3 = []
            Epsilon4 = []
            mask_bkg = (etype == 2)

            np.savetxt('results/scores/best_scores_bkg_' + model_name + '.txt', scores[mask_bkg])
            for key in sgn_dict:

               mask_sgn = (etype == sgn_dict[key])               
               y_true = np.concatenate((np.zeros(scores[mask_bkg].shape[0], dtype="float32"), np.ones(scores[mask_sgn].shape[0], dtype="float32")))
               y_test = np.concatenate((scores[mask_bkg], scores[mask_sgn]))
               auc = roc_auc_score(y_true, y_test)
               eps1, eps2, eps3, eps4 = model.evaluate_efficiencies(y_true, y_test)
               AUC.append(auc)
               Epsilon1.append(eps1)
               Epsilon2.append(eps2)
               Epsilon3.append(eps3)
               Epsilon4.append(eps4)
               # Save scores individually
               np.savetxt('results/scores/best_scores_sgn_' + key + '_' + model_name + '.txt', scores[mask_sgn])

            np.savetxt('results/metrics/AUC_' + model_name + '.txt', AUC)
            np.savetxt('results/metrics/Epsilon1_' + model_name + '.txt', Epsilon1)
            np.savetxt('results/metrics/Epsilon2_' + model_name + '.txt', Epsilon2)
            np.savetxt('results/metrics/Epsilon3_' + model_name + '.txt', Epsilon3)
            np.savetxt('results/metrics/Epsilon4_' + model_name + '.txt', Epsilon4)

def main(Flags):
  
  #create folder to store svdd models
  if not os.path.exists('models'):
    os.makedirs('models')

  #create folders to store results
  if not os.path.exists('results'):
    os.makedirs('results')

  if not os.path.exists('results/scores'):
    os.makedirs('results/scores')

  if not os.path.exists('results/metrics'):
    os.makedirs('results/metrics')

  mode = Flags.mode

  if mode == 'ordered':
    h5_fname = 'h5_ordered'
  elif mode == 'polina':
    h5_fname = 'h5_polina'
  else:
    h5_fname = 'h5'
 
  filename = home + '/data/training.h5' 

  if mode == 'ordered':
   _, regression = unpack_ordered(filename, Flags)
  elif mode == 'polina':
   _, num_objects, regression = unpack_polina(filename, Flags)
  else:
   #load training and validation data
   _, num_objects, regression, classification = unpack(filename, Flags)

  scaler = StandardScaler()

  scaler.fit(regression)
  reg_normalized = scaler.transform(regression)

  if mode == 'ordered':
   training_data = reg_normalized[:]
  elif mode == 'polina':
   training_data = [num_objects[:], reg_normalized[:]]
  else:
   training_data = [classification[:], reg_normalized[:]]
  
  dataset_len = regression.shape[0]
  if mode == 'ordered' or mode == 'polina':
   data_dim = regression.shape[1]
  else:
   data_dim = classification.shape[1]

  if Flags.train == 'True':

   train(dataset_len, data_dim, training_data, FLAGS)

  if Flags.test == 'True':

   #read file and convert to dictionary
   #with open('process_dict/' + channel + '_process_id.txt') as fin:
   # rows = (line.rstrip('\n,').partition(' : ') for line in fin)
   # type_dict = {r[0]: ast.literal_eval(r[2]) for r in rows}
 
   filename = home + '/data/testing.h5' 

   if mode == 'ordered':
    etype, regression = unpack_ordered(filename, Flags)
   elif mode == 'polina':
    etype, num_objects, regression = unpack_polina(filename, Flags)
   else:
    #load training and validation data
    #load data
    etype, num_objects, regression, classification = unpack(filename, Flags)

   reg_normalized = scaler.transform(regression)   

   if mode == 'ordered':
    testing_data = reg_normalized[:]
   elif mode == 'polina':
    testing_data = [num_objects[:], reg_normalized[:]]
   else:
    testing_data = [classification[:], reg_normalized[:]]

   test(dataset_len, data_dim, etype, training_data, testing_data, FLAGS)
       
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=10000, help='Mini batch size')
    parser.add_argument('--dim', nargs='+', type=int, default=[5,8,13,21,34,55,89,144,233], help='Latent space dim.')

    parser.add_argument('--name', type=str, default='SVDD_1l_8', help='Name of the output')

    parser.add_argument('--resume', type=str, default='False', help='Resume')

    parser.add_argument('--train', type=str, default='True', help='Resume')
    
    parser.add_argument('--test', type=str, default='False', help='Resume')

    #parser.add_argument('--fixed_target', type=list, default=[0,1,2,3,4,10,25])

    parser.add_argument('--fixed_target', nargs='+', type=int, default=[0,1,2,3,4,10,25])

    parser.add_argument('--categorical', type=str, default='False')

    parser.add_argument('--mode', type=str, default='ordered')

    FLAGS, unparsed = parser.parse_known_args()
   
    main(FLAGS)
