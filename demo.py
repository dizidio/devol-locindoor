from __future__ import print_function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from devol import DEvol, GenomeHandler
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
#from genome_handler import GenomeHandler
#import devol
#import genome_handler

num_rows = 10;#
num_cols = 15;#
min_val = -95;
max_val = -20;
n_bins = 25;#
num_classes = 71;

def get_loc(ponto):
    return locs_apk[locs_apk[:,0]==ponto, 1:]

# **Prepare dataset**
# This problem uses mnist, a handwritten digit classification problem used
# for many introductory deep learning examples. Here, we load the data and
# prepare it for use by the GPU. We also do a one-hot encoding of the labels.

#K.set_image_data_format("channels_last")

filename_train = 'hists_train_150518_15aps_25bins_10rows_diogo.csv' #
#filename_test = 'hists_test_050718_15aps_25bins_10rows_asus_clean.csv' #

locs_apk = np.genfromtxt('coords_app_020818.csv',delimiter=",");
data_final = np.genfromtxt(filename_train,delimiter=",");
data_l = data_final[:,:-1];
classes_l = data_final[:,-1];

cont = 0;
locs_l = np.empty([classes_l.shape[0], 3]);
for i in classes_l:
    locs_l[cont] = get_loc(i);
    cont = cont + 1;
    

data_l = data_l/num_rows;
            
data_train, data_test, class_train, class_test = train_test_split(data_l,classes_l,test_size = 0.2);

data_train = data_train.reshape(data_train.shape[0], num_cols, n_bins, 1);
data_test = data_test.reshape(data_test.shape[0], num_cols, n_bins, 1);
input_shape = (num_cols, n_bins,1);

class_train = class_train - 1;
class_test = class_test - 1;
class_train = to_categorical(class_train, num_classes);
class_test = to_categorical(class_test, num_classes);

dataset = ((data_train, class_train), (data_test, class_test))

# **Prepare the genome configuration**
# The `GenomeHandler` class handles the constraints that are imposed upon
# models in a particular genetic program. See `genome-handler.py`
# for more information.

genome_handler = GenomeHandler(max_conv_layers=0, 
                               max_dense_layers=6, # includes final dense layer
                               max_filters=512,
                               max_dense_nodes=1024,
                               input_shape=data_train.shape[1:],
                               n_classes=71)

# **Create and run the genetic program**
# The next, and final, step is create a `DEvol` and run it. Here we specify
# a few settings pertaining to the genetic program. The program
# will save each genome's encoding, as well as the model's loss and
# accuracy, in a `.csv` file printed at the beginning of program.
# The best model is returned decoded and with `epochs` training done.

devol = DEvol(genome_handler)
model = devol.run(dataset=dataset,
                  num_generations=20,
                  pop_size=30,
                  epochs=50)
print(model.summary())
