import pandas as pd
import numpy as np
from utils.utils import get_user_data, file_exists, get_granularity_from_minutes
from preprocessing.various import delete_user, makeDummies, addSedentaryLevel, delete_sleep_buckets, addSedentaryClasses, downgrade_datatypes
from preprocessing.studentlife_raw import get_studentlife_dataset


def shift_data(df, nb_lags, period, task_type, dropnan=False):
    '''
    Creates the lagged dataset calling shift_hours for every lag and then combines all the lagged datasets

    :param period: separation of lags, for example: if period = 3 and lag = 3, por a time t we will have features of t-3,
    t-6 and t-9.
    :return:
    '''
    target = 'slevel'
    if task_type == 'classification':
        target = 'sclass'
    # this range goes backwards so older lags are append at first
    lags = range(period * nb_lags, 0, -period)
    # print('Generating {0} time-lags with period equal {1} ...'.format(number_of_lags, period))
    # input sequence (t-n, ... t-1)
    # alors on dance
    result = pd.DataFrame(index=df.index)
    for i in lags:
        to_shift = df.shift(i)
        result = result.join(to_shift, rsuffix=f'(t-{i})')
    result.columns = [
        f'{col}(t-{lags[0]})' if not col.endswith(')') else col for col in result.columns]
    result = result.join(df.loc[:, target])

    # drop rows with NaN values
    if dropnan:
        result.dropna(inplace=True)
    return result


def generate_lagged_dataset(file_name, task_type, nb_lags, period, nb_min):
    '''
    Calls series_to_supervised for every user (otherwise user information would be merged) and then combines it.
    The resulting dataset is saved in the path 'pkl/datasets/gran{}_period{}_lags{}.pkl'

    :param gran: granularity. e.g. '1h', '30m', '2h', etc
    '''

    assert (task_type == 'regression' or task_type ==
            'classification'), 'Not a valid model type.'
    df = get_clean_dataset(nb_min=nb_min)

    #if included_data == 'wos':
    #    df = delete_sleep_buckets(df)
    #if task_type == 'classification':
    #    df = addSedentaryClasses(df)

    list_dfs = [shift_data(get_user_data(df, i), nb_lags, period, task_type)
                for i in df.index.get_level_values(0).drop_duplicates()]

    result = pd.concat(list_dfs, axis=0)
    result.dropna(inplace=True)
    print('Lagged dataset generation finished. Saving and returning it.')
    downgrade_datatypes(result).to_pickle(file_name)
    del result


def get_lagged_dataset(task_type='regression', user=-1, nb_lags=1, period=1, nb_min=60):
    '''
    Get a specific and already generated dataset based on nb_lags, period, gran.
    If personal is true, only returns the specific users data

    '''
    gran = get_granularity_from_minutes(nb_min)
    filename = f'../pkl/lagged_datasets/{task_type}_gran{gran}_period{period}_lags{nb_lags}.pkl'
    if not file_exists(filename):
        print('Lagged dataset not found.')
        print(
            f'Generating lagged dataset with period gran: {gran}, period: {period} and nb_lags:{nb_lags} for a {task_type} model.)')

        generate_lagged_dataset(filename, task_type, nb_lags, period, nb_min)
    data = downgrade_datatypes(pd.read_pickle(filename))
    if user != -1:
        return get_user_data(data, user)
    return data


def generate_clean_dataset(nb_min, file_name, dropna, delete_inconcitencies, with_dummies, from_disc):
    df = get_studentlife_dataset(nb_min)
    if dropna:
        df.dropna(inplace=True)
    if delete_inconcitencies:
        df = delete_user(df, 52)
    if with_dummies:
        df = makeDummies(df)
    df = addSedentaryLevel(df)
    if from_disc:
        downgrade_datatypes(df).to_pickle(file_name)
        print('Clean dataset generation finished. Saving and returning it.')
    return df


def get_clean_dataset(nb_min=60, dropna=False, delete_inconcitencies=True, with_dummies=True, from_disc=True):
    '''
        Creates a dataset with granularity gran. It uses the preprocesed dataset with the same granularity and makes
        some preprocessing steps (delete the user 52, make dummy variables, drops nans rows and calculate de sLevel feature).


        from_disc: if True will return the dataset located in storage. If there is no one,
        it will create and save it. If False, it will create and return the dataset with 
        the specified options, without saving it.
    '''
    gran = get_granularity_from_minutes(nb_min)
    file_name = f'../pkl/datasets/dataset_gran{gran}.pkl'
    if not from_disc:
        print('Creating clean dataset on the fly.')
        return generate_clean_dataset(nb_min, file_name, dropna, delete_inconcitencies, with_dummies, from_disc)
    elif not file_exists(file_name) and from_disc:
        print('Clean dataset does not exist.')
        print(f'Generating clean dataset with gran: {gran}')
        generate_clean_dataset(nb_min, file_name, dropna,
                               delete_inconcitencies, with_dummies, from_disc)
    return downgrade_datatypes(pd.read_pickle(file_name))
