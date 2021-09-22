#%%
import sys
sys.path.append('c:\\Users\\marsa\\Documents\\projects\\tesis-project\\experiments')
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, MaxPooling1D
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import load

from preprocessing.datasets import get_lagged_dataset
from utils.utils import get_user_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

from utils.utils import file_exists

seed = 1
tf.random.set_seed(seed)


dim_num_filters = Integer(low=2, high=8, name='num_filters')
dim_num_kernels = Integer(low=2, high=4, name='num_kernels')
dim_conv_dropout = Real(low=.0, high=.8, name='conv_dropout')
dim_num_dense_nodes = Integer(low=0, high=6, name='num_dense_nodes')
dim_dense_dropout = Real(low=.0, high=.8, name='dense_dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [dim_num_filters,
              dim_num_kernels,
              dim_conv_dropout,
              dim_num_dense_nodes,
              dim_dense_dropout,
              dim_num_epochs,
              dim_batch_size]

def create_model_fn(num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        model = Sequential(name='cnn')
        model.add(Conv1D(filters=2**num_filters,
                         kernel_size=int(num_kernels),
                         activation='relu',
                         padding='causal'))
        model.add(Dropout(conv_dropout))    
        model.add(Flatten())
        if num_dense_nodes>0:
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            model.add(Dropout(dense_dropout))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='MSE',
                      optimizer='adam',
                      )
        return model
    return create_model

@use_named_args(dimensions=dimensions)
def fitness(num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout, num_epochs, batch_size):


    print('num_filters:', num_filters)
    print('num_kernels:', num_kernels)
    print('conv_dropout:', conv_dropout)
    print('num_dense_nodes:', num_dense_nodes)
    print('dense_dropout:', dense_dropout)
    print('num_epochs:', num_epochs)
    print('batch_size: ', batch_size)
    print()

    model_fn = create_model_fn(num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout)

    pe = ImpersonalExperiment(model_fn, 'cnn', 'regression', 32, 4, 1, 60, True)
    pe.run(2**num_epochs, 2**batch_size, verbose=1)
    score = pe.get_mean_score()
    del pe
    del model_fn
    return score

#%%
checkpoint_file = '../../pkl/tunning/checkpoint_cnn_32_imp.pkl' 
checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)

#%%
res = load(checkpoint_file)
x0 = res.x_iters
y0 = res.func_vals
print(len(x0))
sorted(zip(y0, x0))

#%%
if (file_exists(checkpoint_file)):
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                x0=x0,
                                y0=y0,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=100 - len(y0),
                                callback=[checkpoint_saver],
                                verbose=True,
                                n_random_starts=0,
                                random_state=seed)
else:
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='EI',  # Expected Improvement.
                                n_calls=50,  # - len(y0),
                                callback=[checkpoint_saver],
                                verbose=True,
                                random_state=seed)


print(search_result.fun)

sorted(zip(search_result.func_vals, search_result.x_iters))


    #%%
# personal / cnn / 34
# (0.2686, [8, 2, 0.0, 4, 0.8, 6, 3])

# personal / cnn / 32   
# [(0.3832, [7, 3, 0.04065571452744929, 3, 0.5893029464859189, 6, 3])

# impersonal / cnn / 34
# (0.21039999999999998, [8, 4, 0.8, 6, 0.3058565417428838, 6, 8])
# impersonal / cnn / 32
# (0.305, [2, 4, 0.0, 6, 0.8, 5, 8]),

# %%
