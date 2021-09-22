import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from datetime import datetime
from tcn import TCN
import math
import inspect
import pandas as pd
from skopt import load
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from utils.utils import get_granularity_from_minutes,get_experiment_combinations,file_exists,get_list_of_users
from experiments.Experiment import PersonalExperiment, ImpersonalExperiment
from preprocessing.datasets import get_clean_dataset

def create_cnn_model_fn(num_filters, num_kernels, conv_dropout, num_dense_nodes, dense_dropout):
    def create_model():
        model = Sequential(name='cnn')
        model.add(Conv1D(filters=2**num_filters,
                         kernel_size=int(num_kernels),
                         activation='relu',
                         padding='causal'))
        model.add(Dropout(conv_dropout))
        model.add(Flatten())
        if num_dense_nodes > 0:
            model.add(Dense(2**num_dense_nodes, activation='relu'))
            model.add(Dropout(dense_dropout))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='MSE',
                      optimizer='adam', metrics=[keras.metrics.MSE]
                      )
        return model
    return create_model


def create_mlp_model_fn(num_dense_nodes, num_dense_layers, use_batch_norm, dropout):
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
                      metrics=[keras.metrics.MSE])
        return model
    return create_model


def create_rnn_model_fn(num_lstm_layers, num_lstm_nodes, lstm_dropout, num_dense_nodes, dense_dropout):
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
                      loss='MSE', metrics=[keras.metrics.MSE])
        return model
    return create_model


def create_tcn_model_fn(num_filters, kernel_size, dropout, use_skip_connections, use_batch_norm, nb_lags):
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
                      optimizer='adam', metrics=[keras.metrics.MSE]
                      )
        return model
    return create_model


def get_model(arch, *args,):
    model_fn_dict = {'mlp': create_mlp_model_fn,
                     'cnn': create_cnn_model_fn,
                     'rnn': create_rnn_model_fn,
                     'tcn': create_tcn_model_fn}
    fn = model_fn_dict[arch]
    print(inspect.signature(fn))
    model_fn = fn(*args)
    return model_fn


def get_model_info(arch, centroid, model_type):
    checkpoint_file = f'../pkl/tunning/checkpoint_{arch}_{centroid}_{model_type}.pkl'
    res = load(checkpoint_file)
    return sorted(zip(res.func_vals, res.x_iters))[0][1]


def get_closests():
    '''
    Returns a dictionary with the closest centroid for each user
    '''
    df = get_clean_dataset()
    d = df.groupby(level=0)['slevel'].agg(['count', 'mean', 'std'])
    k = 2
    kmeans = KMeans(n_clusters=k).fit(d)
    y = pd.DataFrame(data={'group': kmeans.predict(
        d), 'user': d.reset_index().iloc[:, 0].values})
    user_34_group = int(y.loc[y.user == 34, 'group'])
    user_32_group = int(y.loc[y.user == 32, 'group'])
    y.loc[y.group == user_34_group, 'group'] = 34
    y.loc[y.group == user_32_group, 'group'] = 32
    y = y.set_index('user').iloc[:, 0].to_dict()

    return y


def run_experiment(poi, arch, user, gran, nb_lags, period, closest = None, task_type= 'regresshionn', times=[], **kargs):
    name = f'_{task_type}_gran{get_granularity_from_minutes(gran)}_period{period}_lags{nb_lags}_model-{arch}_user{user}_{poi}'
    file_name = f'../pkl/experiments/{name}.pkl'
    print(datetime.now())
    
    if closest is None: closest = get_closests()
    
    if not file_exists(file_name):
        need_3d_input = (arch != 'mlp')
        closest_centroid = closest[user]
        model_info = get_model_info(
            arch, closest_centroid, poi)
        [nb_epochs, batch_size] = model_info[-2:]
        if arch != 'tcn':
            model = get_model(arch, *model_info[:-2])
        else:
            model = get_model(
                arch, *(model_info[:-2]+[nb_lags]))
        print(model_info)
        if poi == 'per':
            experiment = PersonalExperiment(
                model, arch, task_type, user, nb_lags, period, gran, need_3d_input)
        else:
            experiment = ImpersonalExperiment(
                model, arch, task_type, user, nb_lags, period, gran, need_3d_input)
        experiment.run(2**nb_epochs, 2 **
                        batch_size, **kargs)
        if not experiment.déjà_fait:
            experiment.save()
        times.append(experiment.get_total_time())
        del experiment
        plt.plot(times)
        plt.show()
        plt.close()
    else: print(file_name)


def run_all_experiments(reverse_order:bool=False, **kargs):
    task_type = 'regression'
    closest = get_closests()
    c = 0
    combs = get_experiment_combinations(reverse_order)
    cant_experiments = len(combs)
    times = []
    for poi, arch, user, gran, nb_lags, period in combs:
        run_experiment(poi, arch, user, gran, nb_lags, period, closest, task_type, times, **kargs)
        c += 1
        print(
            f'{c}/{cant_experiments} fishished experiments')
        print(('#' * 10 + '\n') * 3)
        print('')