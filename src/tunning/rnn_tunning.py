# %%
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, LSTM, Dropout, BatchNormalization

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

dim_num_lstm_layers = Integer(low=1, high=2, name='num_lstm_layers')
dim_num_lstm_nodes = Integer(low=2, high=9, name='num_lstm_nodes')
dim_lstm_dropout = Real(low=.0, high=.8, name='lstm_dropout')
dim_num_dense_nodes = Integer(low=0, high=6, name='num_dense_nodes')
dim_dense_dropout = Real(low=.0, high=.8, name='dense_dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [dim_num_lstm_layers,
              dim_num_lstm_nodes,
              dim_lstm_dropout,
              dim_num_dense_nodes,
              dim_dense_dropout,
              dim_num_epochs,
              dim_batch_size]


def create_model_fn(num_lstm_layers, num_lstm_nodes, lstm_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        use_two_layers = num_lstm_layers == 2

        model = Sequential(name='rnn')
        model.add(LSTM(2**num_lstm_nodes, return_sequences=use_two_layers))
        model.add(Dropout(lstm_dropout))

        if use_two_layers:
            model.add(LSTM(2**(num_lstm_nodes-1), return_sequences=False))
            model.add(Dropout(lstm_dropout))

        if num_dense_nodes > 0:
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            model.add(Dropout(dense_dropout))

        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam',
                      loss='MSE')
        return model
    return create_model


@use_named_args(dimensions=dimensions)
def fitness(num_lstm_layers, num_lstm_nodes, lstm_dropout, num_dense_nodes, dense_dropout, num_epochs, batch_size):

    print('num_lstm_layers: ', num_lstm_layers)
    print('num_lstm_nodes: ', num_lstm_nodes)
    print('lstm_dropout: ', lstm_dropout)
    print('num_dense_nodes: ', num_dense_nodes)
    print('dense_dropout: ', dense_dropout)
    print('num_epochs: ', num_epochs)
    print('batch_size: ', batch_size)
    print()

    model_fn = create_model_fn(num_lstm_layers,
                               num_lstm_nodes, lstm_dropout, num_dense_nodes, dense_dropout)

    pe = ImpersonalExperiment(
        model_fn, 'rnn', 'regression', 32, 4, 1, 60, True)
    pe.run(2**num_epochs, 2**batch_size, verbose=1)
    score = pe.get_mean_score()
    del pe
    del model_fn
    return score


# %%
checkpoint_file = '../pkl/checkpoint_nn_32_imp.pkl'
checkpoint_saver = CheckpointSaver(checkpoint_file, compress=9)


# %%
res = load(checkpoint_file)
x0 = res.x_iters
y0 = res.func_vals
print(len(x0))
sorted(zip(y0, x0))


# %%
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


# %%
search_result.fun

# personal / rnn / 34
# (0.24459999999999998, [1, 2, 0.0, 6, 0.4474155673501172, 4, 3])
#
# personal / rnn / 32
# [(0.34219999999999995, [2, 7, 0.0, 6, 0.5608200519233141, 3, 3])
# impersonal / rnn / 34
#
# impersonal / rnn / 32
#
