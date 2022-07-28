
import numpy as np
import math
import sys

from keras.layers import Lambda, Input, Dense, LeakyReLU, BatchNormalization, Concatenate, Reshape, Conv1D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers
from scipy.stats import multivariate_normal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf





from sklearn.metrics import log_loss, roc_curve

from glob import glob
from tqdm import tqdm_notebook as tqdm

from sklearn.metrics import mean_squared_error, roc_auc_score
from keras.datasets import mnist
import importlib
import h5py


def get_R(coords, center=None):
    return np.linalg.norm(coords,axis=1)

class VariationalAutoencoderModel():
    def __init__(self, hidden_layers, filename, D, dataset_len, dim_z, c, mode=None, verbose=False):
        self.D = D
        self.dataset_len = dataset_len
        self.dim_z = dim_z
        self.c = c
        self.verbose = verbose
        self.model_filename = 'models/' + filename + '.h5'
        self.filename = filename
        self.mode = mode

        self.hidden_layers = hidden_layers

        
        model = self.build_lhcdata_model()

        if self.verbose:
            model.summary()
            
        self.model = model
        return
    
    def z_log_var_activation(self, x):
        return K.sigmoid(x) * 10

    def MSE(self, c, dataset_len, dim_z):
        output_data = np.zeros((dataset_len, dim_z), np.int8)
        output_data.fill(c)

        return output_data

    def build_lhcdata_model(self):
        D = self.D
        if self.mode == 'ordered':
         in_regression = Input(shape=(D,), name='in_regression')
         inputs = in_regression
        elif self.mode == 'polina':
         in_nobj = Input(shape=(3,), name='in_nobj')
         in_regression = Input(shape=(D,), name='in_regression')
         inputs = Concatenate()([in_nobj, in_regression])       
        else:
         in_classification = Input(shape=(D,), name='in_classification')
         in_regression = Input(shape=(D*3+2,), name='in_regression')
         inputs = Concatenate()([in_classification, in_regression])

        x = Dense(self.hidden_layers[0], activation='elu')(inputs)
        if len(self.hidden_layers) > 1:
            for i, v in enumerate(self.hidden_layers):
                if (i > 0):
                    x = Dense(v, activation='elu')(x)
        """
        x = Dense(512, activation='elu')(inputs)
        x = Dense(256, activation='elu')(x)
        x = Dense(128, activation='elu')(x)
        #
        x = Dense(256, activation='elu')(inputs)
        x = Dense(128, activation='elu')(x)
        
        x = Dense(8, activation='elu')(inputs)
        """
        z_mean = Dense(self.dim_z, name='z_mean', activation='linear')(x)

        # instantiate encoder model
        if self.mode == 'ordered':
         self.encoder = Model([in_regression], [z_mean], name='encoder')
        elif self.mode == 'polina':
         self.encoder = Model([in_nobj, in_regression], [z_mean], name='encoder')
        else:
         self.encoder = Model([in_classification, in_regression], [z_mean], name='encoder')

        self.encoder.compile(optimizer='adam', loss='mean_squared_error')
        return self.encoder


    def load_weights(self, weight_filename):
        self.model.load_weights(weight_filename)
        
    
    def evaluate_radius_max(self, train_data, batch_size):
        latent_space = self.encoder.predict(train_data, batch_size=batch_size, verbose=self.verbose)

        train_scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))

        max_radius = np.max(train_scores)

        return max_radius

    def evaluate_radius(self, data, r_max, batch_size):
        latent_space = self.encoder.predict(data, batch_size=batch_size, verbose=self.verbose)

        scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))

        scores_r = scores / r_max

        return scores_r

    def evaluate_radius_noscores(self, data, r_max, batch_size):
        latent_space = self.encoder.predict(data, batch_size=batch_size, verbose=self.verbose)

        # scores = get_R(latent_space - self.MSE(self.c, latent_space.shape[0], self.dim_z))
 
        scores_r = 1

        return scores_r

    """
    def evaluate_radius(self, test_bg, test_sig, batch_size=100):
        latent_space_bg = self.encoder.predict(test_bg, batch_size=batch_size, verbose=self.verbose)
        latent_space_sig = self.encoder.predict(test_sig, batch_size=batch_size, verbose=self.verbose)

        test_bg_scores = get_R(latent_space_bg - self.loss_fn(latent_space_bg.shape[0], self.dim_z))
        test_sig_scores = get_R(latent_space_sig - self.loss_fn(latent_space_sig.shape[0], self.dim_z))
        
        self.radius_bg = test_bg_scores
        self.radius_sig = test_sig_scores
        
        self.test_bg_scores_r = test_bg_scores / self.max_radius
        self.test_sig_scores_r = test_sig_scores / self.max_radius
        return test_bg_scores, test_sig_scores
    """
