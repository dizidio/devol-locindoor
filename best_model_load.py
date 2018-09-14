from __future__ import print_function

import time
import numpy as np
#import pandas as pd
import os
import os.path as path
import keras
import struct
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, Reshape, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, LSTM
from keras.models import model_from_json
from keras import backend as K

import matplotlib.pyplot as plt
#import seaborn as sns

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import tensorflow as tf

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

##def mean_pred(y_true, y_pred):
##    points_test = loc_real.astype('float32');
##    result_test = model.predict(data_lt)
##    points_result = tf.matmul(result_test, locs_pts.astype('float32'));
##    distance_x = tf.square(tf.subtract(points_test[:,0],points_result[:,0]));
##    distance_y = tf.square(tf.subtract(points_test[:,1],points_result[:,1]));
##    distance_z = tf.square(tf.subtract(points_test[:,2],points_result[:,2]));
##    distance_total = distance_x + distance_y + distance_z;
##    return tf.reduce_mean(distance_total);

def get_loc(ponto):
    return locs_apk[locs_apk[:,0]==ponto, 1:]


sess = tf.Session()
K.set_session(sess)

num_rows = 10;#
num_cols = 15;#
min_val = -95;
max_val = -20;
n_bins = 25;#
N_VZS = 1;
epochs = 150;
learn_rate = 0.001;
    

timestr = time.strftime("%Y%m%d_%H%M%S")
stamp = '_{}_{}rows_{}aps_{}bins'.format(timestr, num_rows, num_cols, n_bins);

MODEL_NAME = 'DNN'+stamp;#

############################
####### TRAIN CONFIG #######
batch_size = 64;
num_classes = 71;

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out'+stamp, \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out{}/'.format(stamp) + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out{}/'.format(stamp) + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out{}/'.format(stamp) + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out{}/frozen_'.format(stamp) + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out{}/frozen_'.format(stamp) + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out{}/opt_'.format(stamp) + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
    
############################
##### PRE-PROCESSING  ######

filename_train = 'hists_train_150518_15aps_25bins_10rows_diogo.csv' #
filename_test = 'hists_test_050718_15aps_25bins_10rows_asus_clean.csv' #

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

timestr = time.strftime("%Y%m%d_%H%M%S")

data_train = data_train.reshape(data_train.shape[0], num_cols, n_bins, 1);
data_test = data_test.reshape(data_test.shape[0], num_cols, n_bins, 1);
input_shape = (num_cols, n_bins,1);

class_train = class_train - 1;
class_test = class_test - 1;
class_train = keras.utils.to_categorical(class_train, num_classes);
class_test = keras.utils.to_categorical(class_test, num_classes);

data_final_test = np.genfromtxt(filename_test,delimiter=",");#
data_lt = data_final_test[:,:-1];

classes_lt = data_final_test[:,-1];

cont = 0;
loc_real = np.empty([classes_lt.shape[0], 3]);
for i in classes_lt:
    loc_real[cont] = get_loc(i);
    cont = cont + 1;

#loc_real = data_final_test[:,-3:];
data_lt = data_lt/num_rows;
data_lt = data_lt.reshape(data_lt.shape[0], num_cols, n_bins, 1);

cont = 0;
locs_pts = np.empty([71, 3]);
for i in range(71):
    locs_pts[cont] = get_loc(i+1);
    cont = cont + 1;
    


#############################
########## MODEL ############

class calc_dist_test(keras.callbacks.Callback):

  def __init__(self, patience):
      self.patience = patience;
      
  def on_train_begin(self, logs={}):
      self.history_distance = [];
      self.cont_distance = 0;
          
  def on_epoch_end(self, batch, logs={}):
      score_results = self.model.predict(data_lt)
      points_test = loc_real
      points_result = np.dot(score_results, locs_pts)
      distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2);
      print("Dist. Media (test): ", np.mean(distance));
      if (self.history_distance and (np.mean(distance) >= min(self.history_distance))):
          self.cont_distance = self.cont_distance + 1;
      else:
          self.cont_distance = 0;
      self.history_distance.append(np.mean(distance));
      if (self.cont_distance>self.patience):
          self.model.stop_training = True
          print('Early Stopping');

c_test = calc_dist_test(patience=3)

model = load_model('best-model.h5');

print(model.summary())

model.summary()

history = model.fit(data_train, class_train,
            batch_size=batch_size, epochs=epochs, verbose=2,
            validation_data= [data_test, class_test], callbacks=[c_test]) #### data_test, class_test


score_results = model.predict(data_lt)
points_test = loc_real
points_result = np.dot(score_results, locs_pts)
distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2 );
##
##    if np.mean(distance) < best_distance:
##        best_model = model;
##        best_distance = np.mean(distance);
##        error_distance = distance;
##        best_history = history;
##        best_index = ik;

#K.set_learning_phase(0)  # all new operations will be in test mode from now on
#
## serialize the model and get its weights, for quick re-building
#weights = b_model.get_weights()
#config = [layer for layer in b_model.get_config() if layer['class_name']!='Dropout'];       
#
###K.clear_session()
###sess = tf.Session()
###K.set_session(sess)
#
##b_model2 = Sequential()
##b_model2 = Sequential.from_config(config)
##b_model2.set_weights(weights)
#
#
#b_model2 = Sequential()
#b_model2.add(Flatten(input_shape=input_shape))
#b_model2.add(Dense(128, activation='tanh', kernel_initializer = 'he_normal'))
#b_model2.add(Dense(64, activation='tanh', kernel_initializer = 'he_normal'))
#b_model2.add(Dense(num_classes, activation='softmax', kernel_initializer = 'he_normal'))
#
#b_model.set_weights(weights)
#
#export_model(tf.train.Saver(), b_model2,
#             ["{}_input".format(b_model2.get_config()[0]['config']['name'])],
#              "{}/Softmax".format(b_model2.get_config()[-1]['config']['name']));
#
#
##np.savetxt("./out"+stamp+"/error_DNN"+stamp+".txt", error_distance, '%5.2f', delimiter = ",")
#
#config_training = b_model2.get_config() + [type(optimizer)] + [optimizer.get_config()] + \
#                  ["learn_rate = {}; epochs = {}; n_bins = {}; num_rows = {}; num_cols = {}; batch_size = {}; N_VZS = {}".format(learn_rate, epochs, n_bins, num_rows, num_cols, batch_size, N_VZS)] + \
#                  [filename_train];
#
#np.savetxt("./out"+stamp+"/model_config"+stamp+".txt", config_training, '%s')


plt.figure()
sorted_ = np.sort(distance)
yvals = np.arange(len(sorted_))/float(len(sorted_))
plt.title('DNN'+stamp)
plt.ylabel('Probabilidade Acumulada')
plt.xlabel('Erro (m)')

plt.plot(sorted_, yvals)
plt.savefig('cumulated_distance_test'+stamp+'.pdf'.format(timestr, num_rows, num_cols, n_bins))

print(model.summary())
#print("Best Distance: {}".format(best_distance));

