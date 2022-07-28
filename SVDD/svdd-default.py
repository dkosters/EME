from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
#os.add_dll_directory("~/usr/lib32")
#from ctypes import *
#cdll.LoadLibrary("~/usr/lib32/libcuda.so.1")


#####  Import the model.py file #####
import modeldefault as model
from modeldefault import VariationalAutoencoderModel
from dataloader import unpack, unpack_ordered, unpack_polina



#####  Python essentials #####
import numpy as np
import glob
from tqdm import tqdm
import sys  
import re
import math
import itertools
from pathlib import Path
import argparse
import time
import os.path
import pprint
home = os.getcwd()
sgn_dict = {'Ato4l': 1, 'hChToTauNu': 3, 'hToTauTau': 4, 'leptoquark': 5}


#####  Tensorflow related #####
import tensorflow as tf
from keras.models import Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers
from scipy.stats import multivariate_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import log_loss, roc_curve
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, DistanceMetric


#####  Unknown where used #####
import h5py
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import random
from joblib import Parallel, delayed
import multiprocessing
import ast




def test(dataset_len, data_dim, etype, training_data, test_data, Flags):



    #### READ FLAGS ######
    mode = Flags.mode
    str_hl = Flags.hidden_layers
    hl_list = str_hl.split()
    map_object = map(int, hl_list)
    hl_int_list = list(map_object)
    dim_z = Flags.dim 
    ft = Flags.fixed_target

    #### generate name from FLAGS ######
    if mode == 'ordered' or mode == 'polina':
      name = 'SVDD_{0}l'.format(len(hl_int_list))
      for i in hl_int_list:
        name += '_{0}'.format(i)
      name += '_bs_10000'  + '_' + mode
    else:
      name = Flags.name + '_bs_10000'

    for c in tqdm(ft):
      for _z in dim_z:
        with tf.device("/{0}".format(Flags.device)):
            sess = tf.compat.v1.Session()
            tf.compat.v1.keras.backend.set_session(sess)
            model_name = name + '_ft_' + str(c) + '_zdim_' + str(_z)

            #create empty SVDD model
            model = VariationalAutoencoderModel(hl_int_list, model_name, data_dim, dataset_len, _z, c, mode=mode, verbose=True)

            print("INFO:: loading model :")
            print("test data shape")
            print(test_data.shape)
            print(data_dim)
            print(dataset_len)
            print(_z)
            print(c)
            print(model_name)


            #loading weights on SVDD model
            model.load_weights('models/' + model_name + '.h5')

            #Evaluate radius for training with output r_max
            r_max = model.evaluate_radius_max(training_data,100000)

            print("radius max is:")
            print(r_max)
            
            if r_max == 0.0:
              r_max = 1.e-35
            
            #evaluate output vector of the SVDD and calculate the scores
            for i in range(0, Flags.iterations, 1):
                scores = model.evaluate_radius(test_data, r_max,Flags.batch)


def main(Flags):

    tf.keras.backend.set_floatx(Flags.precision)
    print("using %s precision" % (tf.keras.backend.floatx()))

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
        _, regression = unpack_ordered(filename, Flags,Flags.precision)
    elif mode == 'polina':
        _, num_objects, regression = unpack_polina(filename, Flags,Flags.precision)
    else:
        #load training and validation data
        _, num_objects, regression, classification = unpack(filename, Flags,Flags.precision)
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

    filename = home + '/data/testing.h5' 

    if mode == 'ordered':
        etype, regression = unpack_ordered(filename, Flags,Flags.precision)
    elif mode == 'polina':
        etype, num_objects, regression = unpack_polina(filename, Flags,Flags.precision)
    else:
        #load training and validation data
        #load data
        etype, num_objects, regression, classification = unpack(filename, Flags,Flags.precision)

    reg_normalized = scaler.transform(regression)   

    if mode == 'ordered':
        testing_data = reg_normalized[:]
    elif mode == 'polina':
        testing_data = [num_objects[:], reg_normalized[:]]
    else:
        testing_data = [classification[:], reg_normalized[:]]

    
    test(dataset_len, data_dim, etype, training_data, testing_data, Flags)
    print("finished inference")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=10000, help='Mini batch size')
    
    parser.add_argument('--dim', nargs='+', type=int, default=[5,8,13,21,34,55,89,144,233], help='Latent space dim.')

    parser.add_argument('--hidden_layers', type=str, default='8', help='Number of nodes in hidden layers')   
   
    parser.add_argument('--device', type=str, default='cpu', help='type of device cpu or gpu')

    parser.add_argument('--resume', type=str, default='False', help='Resume')

    parser.add_argument('--train', type=str, default='False', help='Resume')
    
    parser.add_argument('--test', type=str, default='True', help='Resume')

    parser.add_argument('--fixed_target', nargs='+', type=int, default=[0,1,2,3,4,10,25])

    parser.add_argument('--mode', type=str, default='ordered')

    parser.add_argument('--iterations', type=int, default=1, help="number of times the inference procedure is repeated")

    parser.add_argument('--precision', type=str, default="float32", help="set precision (default = float32)")


    FLAGS, unparsed = parser.parse_known_args()
   
    main(FLAGS)
