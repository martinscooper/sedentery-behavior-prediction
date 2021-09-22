#%%
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import load

from utils.utils import file_exists

seed = 1
tf.random.set_seed(seed)


dim_num_dense_nodes = Integer(low=2, high=9, name='num_dense_nodes')
dim_num_dense_layers = Integer(low=1, high=8, name='num_dense_layers')
dim_use_batch_norms = Integer(low=0, high=1, name='use_batch_norm')
dim_dropout = Real(low=.0, high=.8, name='dropout')
dim_num_epochs = Integer(low=2, high=6, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [
    dim_num_dense_nodes,
    dim_num_dense_layers,
    dim_use_batch_norms,
    dim_dropout,
    dim_num_epochs,
    dim_batch_size
]

def create_model(num_dense_nodes, num_dense_layers, use_batch_norm, dropout):
    def create_model():
        model = Sequential(name='mlp')
        for _ in range(num_dense_layers):
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            if use_batch_norm == 1:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam',
                      loss='MSE',
                      metrics=keras.metrics.MSE)
        return model
    return create_model


@use_named_args(dimensions=dimensions)
def fitness(num_dense_nodes, num_dense_layers, use_batch_norm, dropout, num_epochs, batch_size):
    print('num_dense_nodes:', num_dense_nodes)
    print('num_dense_layers:', num_dense_layers)
    print('use_batch_norm:', use_batch_norm)
    print('dropout:', dropout)
    print('num_epochs: ', num_epochs)
    print('batch_size: ', batch_size)
    print()
    model_fn = create_model(
        num_dense_nodes, num_dense_layers, use_batch_norm, dropout)

    pe = ImpersonalExperiment(model_fn, 'mlp', 'regression', 32, 4, 1, 60, False)
    pe.run(2**num_epochs, 2**batch_size, verbose=1)
    score = pe.get_mean_score()
    del pe
    del model_fn
    return score


# %%
checkpoint_file = '../pkl/checkpoint_mlp_32_imp.pkl'
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
                                n_calls=50 - len(y0),
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
# personal / mlp / 34
# (0.2516, [8, 4, 1, 0.8, 6, 3])

# personal / mlp / 32
# (0.3606, [2, 2, 0, 0.368, 6, 3])

# impersonal / mlp / 34
# (0.2105, [5, 1, 0, 0.743, 6, 7])

# impersonal / mlp / 32
# 0.3084, [5, 1, 1, 0.373779406268599, 6, 7]