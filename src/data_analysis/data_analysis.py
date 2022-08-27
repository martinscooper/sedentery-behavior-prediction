from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing.studentlife_raw import get_sensor_data, get_studentlife_dataset
from preprocessing.datasets import get_clean_dataset, get_lagged_dataset, get_user_data
from scipy.stats import pearsonr
from pandas.plotting import autocorrelation_plot
from utils.utils_graphic import plot_by_week
from utils.utils import get_user_data

sns.set_style("whitegrid")

def plot_activity_logs_per_user(only_unknowns=True):
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')

    df_all = df.groupby(['userId','time'], as_index=False).size().to_frame('count')
    df_all.reset_index(inplace=True)

    if only_unknowns:
        df = df[df['inference'] == 3]
        title = 'Registros de actividad desconocidos por usuario'
    else:
        df = df[df['inference'] != 3]
        title = 'Registros de actividad no desconocidos por usuario'

    df = df.groupby(['userId','time']).size().to_frame('count')
    df.reset_index(inplace=True)

    plt.close()
    fig, (ax,ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    sns.barplot(x="userId", y="count", data=df, ax=ax,
                estimator=np.mean, ci=None,
                palette=sns.color_palette("Paired", 10))
    plt.xticks(rotation='vertical')
    ax.set_xlabel("")
    ax.set_ylabel("Media")
    ax.set_title(title)

    sns.barplot(x="userId", y="count", data=df, ax=ax2,
                estimator=np.std, ci=None,
                palette=sns.color_palette("Paired", 10))
    ax2.set_ylabel("Desviación estándar")
    ax2.set_xlabel("User ID")
    plt.show()


def plot_activity_unknown_cumsum():
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df = df[df['inference'] == 3]
    min_date = datetime.fromtimestamp(df.time.min())
    min_date = min_date.replace(minute=0, second=0, microsecond=0)
    max_date = datetime.fromtimestamp(df.time.max())
    max_date = max_date.replace(minute=0, second=0, microsecond=0)
    print(min_date)
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                 pd.date_range(min_date,
                                                                               max_date,
                                                                               freq='1h')],
                                                      names=['userId', 'time']))
    print(s)
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')
    df_size = df.groupby(['userId', 'time']).size()
    print(df_size)
    s.loc[:, 'inference'] = df_size
    s = s.fillna(0)
    vals = s.to_numpy()
    vals = np.cumsum(vals)

    plt.close()
    fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.plot(vals)
    # ax.set_xlabel('Horas')
    # ax.set_ylabel('Suma acumulativa de valores desconocidos')

    ax2 = fig.add_subplot(111)
    ax2.plot(vals)
    ax2.set_xlabel('Buckets de una hora')
    ax2.set_ylabel('Registros desconocidos')
    ax2.set_title('Suma acumulativa de registros desconocidos por hora')
    plt.subplots_adjust()

    fig.show()


def print_basic_info(sensor):
    '''
    plots info of sensor data such as :
    - nb_reg
    :return:
    '''
    print(f'basic info for {sensor}')
    df = get_sensor_data(sensor)
    nb_regs = df.shape[0]
    print(f'Cantidad total de registros: {nb_regs}')

    '''
    nb_unknown = df.loc[df["inference"]==3].shape
    print(f'Cantidad de unknowns: {nb_unknown}')
    min_date = datetime.fromtimestamp(min(df.time))
    print(f'min date: {min_date}')
    max_date = datetime.fromtimestamp(max(df.time))
    print(f'max date: {max_date}')
    '''
    del df


def print_tsd_info(sensor, freq):
    '''
    print info related to the frequency of the sensor data

    :param sensor:
    :param freq:
    :return:
    '''
    print(f'freq info for {sensor} and {freq}')
    df = get_sensor_data(sensor)

    if 'time' in df.columns:
        time = 'time'
    else: time = 'timestamp'

    df = df.loc[df[' activity inference'] != 3]

    min_date = datetime.fromtimestamp(min(df[time]))
    max_date = datetime.fromtimestamp(max(df[time]))
    s = pd.DataFrame(index=pd.MultiIndex.from_product(iterables=[df['userId'].drop_duplicates(),
                                                                         pd.date_range(min_date,
                                                                                       max_date,
                                                                                       freq=freq)],
                                                              names=['userId', time]))
    print(f'Total number of 1h blocks : {s.shape[0]}')

    df[time] = pd.to_datetime(df[time], unit='s').dt.floor(freq)
    grouped_per_hour_and_user = df.groupby(['userId', time]).size()
    print(f'Total number of {freq} blocks with data available: {len(grouped_per_hour_and_user)}')
    vals = grouped_per_hour_and_user.to_numpy()
    print(f'Average and std available logs for {freq}: {vals.mean()} / {vals.std()}')
    print(f'Min and max available logs for {freq}: {vals.min()} / {vals.max()}')
    print('/n')

    del vals
    del df
    del grouped_per_hour_and_user


def print_tsi_info(sensor, freq):
    '''
    print info related to the frequency of the sensor data

    :param sensor:
    :param freq:
    :return:
    '''
    print(f'freq info for {sensor} and {freq}')
    df = get_sensor_data(sensor)

    if 'start' in df.columns:
        start = 'start'
        end = 'end'
    else:
        start = 'start_timestamp'
        end = ' end_timestamp'
    
    df['diff'] = df[end] - df[start]

    print(f'Promedio y desviación estándar del tamaño de los intervalos: '
          f'{ round(df["diff"].mean() / 3600, 3) } / {round(df["diff"].std() / 3600, 3) }')
    print(f'Maximo y minimo del tamaño de los intervalos: '
          f'{round(df["diff"].min() / 3600, 3)} / {round(df["diff"].max() / 3600, 3)}')
    
    def custom_func(df):
        return df.loc[:,end].iloc[-1] - df.loc[:,start].iloc[0] 

    total_time_elapsed = df.groupby('userId').apply(custom_func).sum()
    approximated_interval_time = df['diff'].sum()

    print('')


def print_all_info():

    TSD = ['activity', 'audio', 'gps', 'bt',
           'wifi', 'calendar', 'wifi_location']
    TSI = ['dark', 'phonelock',
            'phonecharge', 'conversation']

    # for t in TSD:
    #     print_basic_info(t)
    #     for f in ['1h','30min']:
    #         print_tsd_info(t, f)

    for t in TSI:
        print_basic_info(t)
        for f in ['1h','30min']:
            print_tsi_info(t, f)


def plot_portion_of_activity(only_unknowns=True):
    df = get_sensor_data('activity')
    df.columns = ['time', 'inference', 'userId']
    df['time'] = pd.to_datetime(df['time'], unit='s').dt.floor('1h')

    df_all = df.groupby(['userId','time'], as_index=False).size().to_frame('count')
    df_all.reset_index(inplace=True)

    if only_unknowns:
        df = df[df['inference'] == 3]
        title = 'Registros de actividad desconocidos por usuario'
    else:
        df = df[df['inference'] != 3]
        title = 'Registros de actividad no desconocidos por usuario'

    df = df.groupby(['userId','time']).size().to_frame('count')
    df.reset_index(inplace=True)

    plt.close()
    fig, (ax,ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex='all')
    sns.barplot(x="userId", y="count", data=df, ax=ax,
                estimator=np.mean, ci=None,
                palette=sns.color_palette("Paired", 10))
    plt.xticks(rotation='vertical')
    ax.set_xlabel("")
    ax.set_ylabel("Media")
    ax.set_title(title)

    sns.barplot(x="userId", y="count", data=df, ax=ax2,
                estimator=np.std, ci=None,
                palette=sns.color_palette("Paired", 10))
    ax2.set_ylabel("Desviación estándar")
    ax2.set_xlabel("User ID")
    plt.show()


def numerical_data_distribution():
    # %%
    df = get_clean_dataset()
    # %%
    df.info()
    # %%
    df = get_lagged_dataset()
    # %%
    df.info()
    # %%
    df_num = df.select_dtypes(exclude='object')
    #%%
    df.shape

    # %%
    df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    # %%
    df['location_mean(t-1)']

    #%%
    df_num_corr = df_num.corr()['slevel'][:-1] # 
    golden_features_list = df_num_corr.sort_values(ascending=False)
    print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
    # %%
    for i in range(0, len(df_num.columns), 5):
        sns.pairplot(data=df_num,
                    x_vars=df_num.columns[i:i+5],
                    y_vars=['slevel'])



def some_f():
    df = get_studentlife_dataset()
    lista = []
    for i in df.index.get_level_values(0).drop_duplicates():
        '''
        Creates 3 plots for each user:
        - The hours in which there is no information about physical activity
        - Autocorrelation plot
        - Plot a week of the user activity (see plot_by_week code)
        
        '''
        min_date = min(df.index.get_level_values(1))
        max_date = max(df.index.get_level_values(1))
        dfu = get_user_data(df, i).droplevel(0).loc[:, 'slevel']
        idx = pd.date_range(min_date, max_date, freq='h')
        d = pd.DataFrame(index=idx)
        d['slevel'] = dfu

        a = d.isna().sum()
        lista.append(a)
        nulls = d.isna()
        plt.close()
        plt.scatter(list(range(len(idx))), nulls, 0.1, marker='x')
        plt.show()

        # Fill nan with previous value
        d.ffill(inplace=True)

        # Autocorrelation Plot
        plt.close()
        plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120})
        autocorrelation_plot(d.slevel)
        plt.title('{0},{1}'.format(str(i), a))
        plt.show()

        plot_by_week(i)



def raw_data_stadistics(freq='1h'):
    sdata = get_sensor_data('activity')
    sdata.columns = ['time', 'activityId', 'userId']
    sdata = sdata.loc[sdata['activityId'] != 3]
    sdata['time'] = pd.to_datetime(sdata['time'], unit='s').dt.floor(freq)
    #return a = sdata.groupby(['userId', 'time']).count()

def generate_MET_stadistics(df):
    '''
    Generates a dataframe with some useful information about all the users
    columns: 'user', 'met', 'std', 'corr', 'nb_nulls'

    '''
    things = list()
    for u in df.index.get_level_values(0).drop_duplicates():
        dfuser = get_user_data(df, u)
        aux = dfuser.droplevel(0).loc[:, 'slevel']
        idx = pd.date_range('2013-03-27 04:00:00', '2013-06-01 3:00:00', freq='h')
        d = pd.DataFrame(index=idx)
        d['slevel'] = aux
        n = d.isna().sum()[0]
        dfuser['hourofday'] = dfuser.index.get_level_values(1).hour
        dfuser['dayofweek'] = dfuser.index.get_level_values(1).dayofweek
        stats = dfuser.groupby(['dayofweek', 'hourofday'])['slevel'].agg(['mean', 'std']).dropna()
        corr = pearsonr(stats['mean'], stats['std'])[0]

        things.append([u, stats['mean'].mean(), stats['std'].mean(), corr, n])
        # corrs.append(corr)
    return pd.DataFrame(columns=['user', 'met', 'std', 'corr', 'nb_nulls'], data=things).sort_values('met')
