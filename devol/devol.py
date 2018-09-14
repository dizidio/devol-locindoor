"""
Run a genetic algorithm to find an appropriate architecture for some image
classification task with Keras+TF.

To use, define a `GenomeHandler` defined in genomehandler.py. Then pass it, with
training data, to a DEvol instance to run the genetic algorithm. See the readme
for more detailed instructions.
"""

from __future__ import print_function
import random as rand
import csv
import operator
import gc
import os
import time
from datetime import datetime
from keras.callbacks import Callback
from keras.models import load_model
import keras.backend as K
from sklearn.metrics import log_loss, auc

import numpy as np
from keras.backend.tensorflow_backend import set_session
#from pyDOE import *

if K.backend() == 'tensorflow':
    import tensorflow as tf

__all__ = ['DEvol']

METRIC_OPS = [operator.__lt__, operator.__gt__]
METRIC_OBJECTIVES = [min, max]


locs_apk = np.genfromtxt('coords_app_020818.csv',delimiter=",");

def get_loc(ponto):
        return locs_apk[locs_apk[:,0]==ponto, 1:]

def get_cdf(pct, sorted_dist):
        return sorted_dist[int(np.round(pct*len(sorted_dist)/100))];
    
    
    
class calc_dist_test(Callback):

  def __init__(self, patience, x_test, y_test):
      self.patience = patience;
      self.history_auc = [];
      self.history_acc = []
      self.cont_distance = 0;
      self.min_acc = 0;
      self.x_test = x_test;
      self.y_test = y_test;
          
  def on_epoch_end(self, batch, logs={}):
      score_results = self.model.predict(data_lt)
      points_test = loc_real
      points_result = np.dot(score_results, locs_pts)
      distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2);
      #sorted_dist = np.sort(distance);
      #print("25% Dist: ", get_cdf(25, sorted_dist));
      #print("50% Dist: ", get_cdf(50, sorted_dist));
      #print("75% Dist: ", get_cdf(75, sorted_dist));
      #print("95% Dist: ", get_cdf(95, sorted_dist));
      
      sorted_ = np.sort(distance)
      yvals = np.arange(len(sorted_))/float(len(sorted_))
      sorted_ = np.append(sorted_, 100);
      yvals = np.append(yvals, 1);
      
      auc_value = auc(sorted_, yvals);
      print("Dist. Media: ", np.mean(distance));
      print("AUC (test): ", auc_value);
      
      if (self.history_auc and (auc_value <= max(self.history_auc))):
          self.cont_distance = self.cont_distance + 1;
      else:
          self.cont_distance = 0;

      self.history_auc.append(auc_value);
      self.history_acc.append(self.model.evaluate(self.x_test, self.y_test, verbose=0)[1]);
      index = np.argmax(self.history_auc);
      self.max_acc = self.history_acc[index];
      
      if (self.cont_distance>self.patience):
          self.model.stop_training = True
          print('Early Stopping');


num_rows = 10;#
num_cols = 15;#
min_val = -95;
max_val = -20;
n_bins = 25;#
num_classes = 71;

filename_test = 'hists_test_050718_15aps_25bins_10rows_asus_clean.csv' #
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

        

class DEvol:
    """
    Object which carries out genetic search and returns top performing model
    upon completion.
    """
    def __init__(self, genome_handler, data_path=""):
        """
        Initialize a DEvol object which carries out the training and evaluation
        of a genetic search.

        Args:
            genome_handler (GenomeHandler): the genome handler object defining
                    the restrictions for the architecture search space
            data_path (str): the file which the genome encodings and metric data
                    will be stored in
        """
        self.genome_handler = genome_handler
        timestr = time.strftime("%Y%m%d_%H%M%S")
        self.datafile = data_path or (timestr + '.csv')
        self._bssf = -1

        if os.path.isfile(data_path) and os.stat(data_path).st_size > 1:
            raise ValueError(('Non-empty file %s already exists. Please change'
                              'file path to prevent overwritten genome data.'
                              % data_path))

        print("Genome encoding and metric data stored at", self.datafile, "\n")
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            metric_cols = ["Val Loss", "Val Accuracy"]
            genome = genome_handler.genome_representation() + metric_cols
            writer.writerow(genome)

    def set_objective(self, metric):
        """
        Set the metric for optimization. Can also be done by passing to
        `run`.

        Args:
            metric (str): either 'acc' to maximize classification accuracy, or
                    else 'loss' to minimize the loss function
        """
        if metric == 'acc':
            metric = 'accuracy'
        if metric not in ['loss', 'accuracy']:
            raise ValueError(('Invalid metric name {} provided - should be'
                              '"accuracy" or "loss"').format(metric))
        self._metric = metric
        self._objective = "max" if self._metric == "accuracy" else "min"
        self._metric_index = 1 if self._metric == 'loss' else -1
        self._metric_op = METRIC_OPS[self._objective == 'max']
        self._metric_objective = METRIC_OBJECTIVES[self._objective == 'max']

    def run(self, dataset, num_generations, pop_size, epochs, fitness=None,
            metric='accuracy'):
        """
        Run genetic search on dataset given number of generations and
        population size

        Args:
            dataset : tuple or list of numpy arrays in form ((train_data,
                    train_labels), (validation_data, validation_labels))
            num_generations (int): number of generations to search
            pop_size (int): initial population size
            epochs (int): epochs for each model eval, passed to keras model.fit
            fitness (None, optional): scoring function to be applied to
                    population scores, will be called on a numpy array which is
                    a min/max scaled version of evaluated model metrics, so It
                    should accept a real number including 0. If left as default
                    just the min/max scaled values will be used.
            metric (str, optional): must be "accuracy" or "loss" , defines what
                    to optimize during search

        Returns:
            keras model: best model found with weights
        """
        self.set_objective(metric)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset

        # generate and evaluate initial population
        members = self._generate_random_population(pop_size)
        pop = self._evaluate_population(members,
                                        epochs,
                                        fitness,
                                        0,
                                        num_generations)

        # evolve
        for gen in range(1, num_generations):
            members = self._reproduce(pop, gen)
            pop = self._evaluate_population(members,
                                            epochs,
                                            fitness,
                                            gen,
                                            num_generations)
            ##### GARBAGE COLLECTOR #####
            K.clear_session()
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            config.gpu_options.visible_device_list = "0"
            set_session(tf.Session(config=config))
            #############################

        return load_model('best-model.h5')

    def _reproduce(self, pop, gen):
        members = []
        perct_cross = 0.50
        # 95% of population from crossover
        for _ in range(int(len(pop) * perct_cross)):
            members.append(self._crossover(pop.select(), pop.select()))

        # best models survive automatically
        members += pop.get_best(len(pop) - int(len(pop) * perct_cross))

        # randomly mutate
        for imem, mem in enumerate(members):
            members[imem] = self._mutate(mem, gen)
        return members

    def _evaluate(self, genome, epochs):
        model = self.genome_handler.decode(genome)
        print(model.summary());
        loss, accuracy = None, None
        cback_dist = calc_dist_test(patience=3, x_test = self.x_test, y_test = self.y_test);
        try:
            model.fit(self.x_train, self.y_train,
                      validation_data=(self.x_test, self.y_test),
                      epochs=epochs,
                      verbose=2,
                      callbacks=[cback_dist])
            #_, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
            #score_results = model.predict(data_lt)
            #points_test = loc_real
            #points_result = np.dot(score_results, locs_pts)
            #distance = np.sqrt((points_test[:,0]-points_result[:,0])**2 + (points_test[:,1]-points_result[:,1])**2 + (points_test[:,2]-points_result[:,2])**2);
            #print("Dist. Media (test): ", np.mean(distance));
            #print("Dist. Max (test): ", np.max(distance));
            
            max_auc = max(cback_dist.history_auc);
            max_acc = cback_dist.max_acc;
            accuracy = max_auc;
            print("AUC: {} - Val_Acc: {}".format(max_auc, max_acc));
            
            #print("AUC: {}".format(loss)); 
        except Exception as e:
            loss, accuracy = self._handle_broken_model(model, e)

        self._record_stats(model, genome, loss, accuracy)

        return model, loss, accuracy

    def _record_stats(self, model, genome, loss, accuracy):
        with open(self.datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = list(genome) + [loss, accuracy]
            writer.writerow(row)

        met = loss if self._metric == 'loss' else accuracy
        if (self._bssf is -1 or
                self._metric_op(met, self._bssf) and
                accuracy is not 0):
            try:
                os.remove('best-model.h5')
            except OSError:
                pass
            self._bssf = met
            model.save('best-model.h5')

    def _handle_broken_model(self, model, error):
        del model

        n = self.genome_handler.n_classes
        loss = log_loss(np.concatenate(([1], np.zeros(n - 1))), np.ones(n) / n)
        accuracy = 1 / n
        gc.collect()

        if K.backend() == 'tensorflow':
            K.clear_session()
            tf.reset_default_graph()

        print('An error occurred and the model could not train:')
        print(error)
        print(('Model assigned poor score. Please ensure that your model'
               'constraints live within your computational resources.'))
        return loss, accuracy

    def _evaluate_population(self, members, epochs, fitness, igen, ngen):
        fit = []
        for imem, mem in enumerate(members):
            self._print_evaluation(imem, len(members), igen, ngen)
            res = self._evaluate(mem, epochs)
            v = res[self._metric_index]
            del res
            fit.append(v)

        fit = np.array(fit)
        self._print_result(fit, igen)
        return _Population(members, fit, fitness, obj=self._objective)

    def _print_evaluation(self, imod, nmod, igen, ngen):
        fstr = '\nmodel {0}/{1} - generation {2}/{3}:\n'
        print(fstr.format(imod + 1, nmod, igen + 1, ngen))

    def _generate_random_population(self, size):
        return self.genome_handler.generate2(size)
        #return [self.genome_handler.generate() for _ in range(size)]


    def _print_result(self, fitness, generation):
        result_str = ('Generation {3}:\t\tbest {4}: {0:0.4f}\t\taverage:'
                      '{1:0.4f}\t\tstd: {2:0.4f}')
        print(result_str.format(self._metric_objective(fitness),
                                np.mean(fitness),
                                np.std(fitness),
                                generation + 1, self._metric))

    def _crossover(self, genome1, genome2):
        cross_ind = rand.randint(0, len(genome1))
        child = genome1[:cross_ind] + genome2[cross_ind:]
        return child

    def _mutate(self, genome, generation):
        # increase mutations as program continues
        num_mutations = max(3, generation // 4)
        return self.genome_handler.mutate(genome, num_mutations)


class _Population(object):

    def __len__(self):
        return len(self.members)

    def __init__(self, members, fitnesses, score, obj='max'):
        self.members = members
        scores = fitnesses - fitnesses.min()
        if scores.max() > 0:
            scores /= scores.max()
        if obj == 'min':
            scores = 1 - scores
        if score:
            self.scores = score(scores)
        else:
            self.scores = scores
        self.s_fit = sum(self.scores)

    def get_best(self, n):
        combined = [(self.members[i], self.scores[i])
                    for i in range(len(self.members))]
        sorted(combined, key=(lambda x: x[1]), reverse=True)
        return [x[0] for x in combined[:n]]

    def select(self):
        dart = rand.uniform(0, self.s_fit)
        sum_fits = 0
        for i in range(len(self.members)):
            sum_fits += self.scores[i]
            if sum_fits >= dart:
                return self.members[i]
