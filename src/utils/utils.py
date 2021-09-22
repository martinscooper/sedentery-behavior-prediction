import numpy
import pandas as pd
import os
import itertools


pd.options.mode.chained_assignment = None

numpy.random.seed(7)


def get_user_data(data, userId):
    """
    Get data of a specific user

    """
    result = data.loc[data.index.get_level_values(0) == userId]
    assert (result.shape[0] != 0), 'The user does not exist.'
    return result


def get_not_user_data(data, userId):
    """

    :return: all the data except that of the user specidied

    """
    return data.loc[data.index.get_level_values(0) != userId].sort_index(level=1)


def file_exists(file):
    return os.path.exists(file)


def get_granularity_from_minutes(nb_min):
    if nb_min % 60 == 0:
        gran = f'{int(nb_min/60)}h'
    else:
        gran = f'{nb_min}min'
    return gran


def add_per_to_all_experiments():
    '''
    add _per to all experiment pkl in case I forget to do it
    '''
    path_to_file = './pkl/experiments'
    c = 0
    for fn in os.listdir(path_to_file):
        nfn = fn[:-4] + '_per' + fn[-4:]
        os.rename(f'{path_to_file}/{fn}', f'{path_to_file}/{nfn}')


def get_list_of_users():
    users_list = [ 0,  1,  2,  3,  4,  5,  7,  8,  9, 10, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 22, 23, 24, 25, 27, 30, 31, 32, 33, 34, 35, 36, 39, 41, 42,
            43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 56, 57, 58, 59]
    return users_list


def get_experiment_combinations(reverse_order=False):

    '''
    Get list of all experiments combinations given its caracteristics
    rever_order is used to run a second process so both do not do the same experiment and avoid conflicts
    '''
    pois = ['per', 'imp']
    archs = ['rnn', 'cnn', 'tcn', 'mlp']
    users = get_list_of_users()
    grans = [60,30]
    lags = [1, 2, 4, 8]
    periods = [1, 2, 4]
    sets = [pois, archs,users, grans, lags, periods]
    combs = list(itertools.product(*sets))
    if reverse_order:
        combs.reverse()
    return combs