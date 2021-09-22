# %%
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
import tensorflow as tf
import numpy as np
import math

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tcn import TCN

import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt import load

from utils.utils import file_exists
#from numpy.random import seed
# seed(1)
seed = 1
tf.random.set_seed(seed)

dim_num_filters = Integer(low=2, high=6, name='num_filters')
dim_kernel_size = Integer(low=2, high=4, name='kernel_size')
dim_dropout = Real(low=.0, high=.8, name='dropout')
dim_use_skip_connections = Integer(low=0, high=1, name='use_skip_connections')
dim_use_batch_norm = Integer(low=0, high=1, name='use_batch_norm')
dim_num_epochs = Integer(low=2, high=8, name='num_epochs')
dim_batch_size = Integer(low=3, high=8, name='batch_size')

dimensions = [dim_num_filters,
              dim_kernel_size,
              dim_dropout,
              dim_use_skip_connections,
              dim_use_batch_norm,
              dim_num_epochs,
              dim_batch_size]


def create_model_fn(num_filters, kernel_size, dropout, use_skip_connections, use_batch_norm, nb_lags):
    nb_dilations = max(
        math.ceil(math.log(nb_lags/kernel_size, kernel_size)+1), 2)
    if use_skip_connections == 1 and nb_dilations == 1:
        nb_dilations += 1
    print('dilations:', nb_dilations)
    print('receptive field: ', (kernel_size**(nb_dilations-1))*kernel_size)

    def create_model():
        model = Sequential(name='tcn')
        dilations = [2 ** i for i in range(nb_dilations)]
        model.add(TCN(
            nb_filters=2**num_filters,
            kernel_size=int(kernel_size),
            nb_stacks=1,
            dilations=dilations,
            padding='causal',
            use_skip_connections=use_skip_connections,
            dropout_rate=dropout,
            return_sequences=False,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=use_batch_norm,
            use_layer_norm=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='MSE',
                      optimizer='adam', metrics=keras.metrics.MSE
                      )
        return model
    return create_model


@use_named_args(dimensions=dimensions)
def fitness(num_filters, kernel_size, dropout, use_skip_connections, use_batch_norm, num_epochs, batch_size):
    print('Model hyper-parameters')
    print('num_filters:', num_filters)
    print('kernel_size:', kernel_size)
    print('dropout:', dropout)
    print('use_skip_connections:', use_skip_connections)
    print('use_batch_norm:', use_batch_norm)
    print('num_epochs:', num_epochs)
    print('batch_size: ', batch_size)
    print()

    model_fn = create_model_fn(
        num_filters, kernel_size, dropout, use_skip_connections, use_batch_norm, 4)

    pe = PersonalExperiment(model_fn, 'tcn', 'regression', 32, 4, 1, 60, True)
    pe.run(2**num_epochs, 2**batch_size, verbose=1)
    score = pe.get_mean_score()
    del pe
    del model_fn
    return score


# %%
checkpoint_file = '../pkl/checkpoint_tcn_32_per.pkl'
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
# personal / tcn / 34
# (0.2578, [4, 4, 0.8, 1, 1, 8, 3])

# personal / tcn / 32
# (0.3836, [2, 2, 0.8, 0, 1, 8, 3]),

# impersonal / tcn / 34
# (0.21599999999999997, [6, 2, 0.8, 0, 1, 6, 7]),
# impersonal / tcn / 32
# (0.307, [2, 4, 0.505993020593099, 0, 1, 8, 7])
